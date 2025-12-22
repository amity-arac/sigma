# Copyright Amity
"""
SFT (Supervised Fine-Tuning) format converter for trajectories.

SFT format creates one-to-many expansion where each trajectory generates 
multiple training records - one for each assistant turn.
"""
import json
from typing import Any, Dict, List, Optional

from .base import parse_tool_call_from_content, parse_tool_arguments


def convert_to_sft_format(trajectories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert trajectories to SFT (Supervised Fine-Tuning) training format.
    
    SFT format:
    - task_id: identifier for the task
    - conversations: list of messages up to the decision point
    - answer: list containing the single correct next action
    - rubric: evaluation criteria (optional)
    - reject_rubric: criteria for rejected actions (optional)
    - reject_answer_raw: raw thinking/reasoning for rejected action (if available)
    - chosen_answer_raw: raw thinking/reasoning for chosen action (if available)
    
    One-to-Many Expansion:
    Each trajectory generates multiple SFT records - one for each assistant turn.
    
    Args:
        trajectories: List of trajectory dictionaries
        
    Returns:
        List of SFT training records
    """
    sft_records = []
    
    print(f"[SFT] Processing {len(trajectories)} trajectories")
    
    for traj_idx, trajectory in enumerate(trajectories):
        messages = trajectory.get('messages', [])
        wiki = trajectory.get('wiki', '')
        session_id = trajectory.get('session_id', '')
        task_id = trajectory.get('task_id', traj_idx)
        
        # Build rejection map: index of rejection -> rejected data
        rejection_map = {}
        for i, msg in enumerate(messages):
            if msg.get('role') == 'rejected':
                rejection_map[i] = msg.get('rejected', {})
        
        print(f"[SFT] Trajectory {traj_idx} ({session_id[:8] if session_id else 'N/A'}...): {len(messages)} messages, {len(rejection_map)} rejected")
        
        # Build system prompt from wiki/policy
        system_content = _build_system_prompt(wiki)
        
        # Create one record per assistant turn
        sft_records_for_traj = _create_sft_records_for_trajectory(
            messages=messages,
            task_id=task_id,
            system_content=system_content,
            rejection_map=rejection_map
        )
        sft_records.extend(sft_records_for_traj)
    
    print(f"[SFT] Summary: {len(sft_records)} records created")
    
    return sft_records


def _build_system_prompt(wiki: str) -> str:
    """Build the system prompt for SFT format."""
    return f"""
<instructions>
You are a customer service agent that helps the user according to the <policy>
provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.
Try to be helpful and always follow the policy. Always make sure you generate
valid JSON only.
</instructions>
<policy>
{wiki}
</policy>
"""


def _create_sft_records_for_trajectory(
    messages: List[Dict[str, Any]],
    task_id: Any,
    system_content: str,
    rejection_map: Optional[Dict[int, Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """
    Create SFT records for a trajectory with one-to-many expansion.
    
    Creates a record at each assistant action point (tool call or response).
    If rejection_map is provided, includes rejected reasoning at relevant points.
    
    Args:
        messages: List of trajectory messages
        task_id: Task identifier
        system_content: System prompt content
        rejection_map: Optional dict mapping rejection indices to rejected data
        
    Returns:
        List of SFT records for this trajectory
    """
    sft_records = []
    rejection_map = rejection_map or {}
    
    # Start with system prompt and initial greeting
    base_conversations = [
        {
            "role": "system",
            "content": system_content.strip()
        },
        {
            "role": "assistant",
            "content": "Hi! How can I help you today?",
            "tool_calls": None
        }
    ]
    
    conversations = list(base_conversations)
    
    # Track if the previous message was a rejection
    pending_rejection = None
    
    for i, msg in enumerate(messages):
        role = msg.get('role', '')
        
        # Check if this is a rejection marker
        if role == 'rejected':
            # Store the rejection data - the next assistant action will be the "chosen" action
            pending_rejection = msg.get('rejected', {})
            continue
        
        if role in ['agent', 'tool']:
            # Check if this is an action point (agent response or tool call)
            converted = _convert_message_for_sft(msg)
            if converted:
                # Create SFT record with conversation up to this point as context
                sft_record = {
                    "task_id": task_id,
                    "conversations": list(conversations),
                    "answer": [converted],
                    "rubric": "",
                    "reject_rubric": ""
                }
                
                # Add chosen reasoning if available
                reasoning = msg.get('reasoning', '')
                if reasoning:
                    sft_record["chosen_answer_raw"] = reasoning
                
                # If there was a pending rejection, add the rejected reasoning
                if pending_rejection:
                    rejected_reasoning = pending_rejection.get('reasoning', '')
                    if rejected_reasoning:
                        sft_record["reject_answer_raw"] = rejected_reasoning
                    pending_rejection = None
                
                sft_records.append(sft_record)
                
                # Add this message to the conversation for subsequent records
                conversations.append(converted)
        elif role in ['user', 'tool-result']:
            # Add to conversation history
            converted = _convert_message_for_sft(msg)
            if converted:
                conversations.append(converted)
    
    return sft_records


def _convert_message_for_sft(msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert a trajectory message to SFT format."""
    role = msg.get('role', '')
    
    if role == 'user':
        return {
            "role": "user",
            "content": msg.get('content', ''),
            "tool_calls": None
        }
    elif role == 'agent':
        result = {
            "role": "assistant",
            "content": msg.get('content'),
            "tool_calls": None
        }
        
        # Check if this agent message has tool_calls
        tool_name = msg.get('tool_name')
        tool_arguments = msg.get('tool_arguments')
        if tool_name:
            args = parse_tool_arguments(tool_arguments)
            result["tool_calls"] = [{
                "id": f"chatcmpl-tool-{msg.get('id', '')}",
                "name": tool_name,
                "arguments": args,
                "requestor": "assistant"
            }]
        return result
    elif role == 'tool':
        # Tool call from agent - represents assistant making a tool call
        tool_name = msg.get('tool_name')
        tool_arguments = msg.get('tool_arguments')
        
        # If tool_name is not set, try to parse from content
        if not tool_name:
            content = msg.get('content', '')
            parsed = parse_tool_call_from_content(content)
            if parsed[0]:
                tool_name, tool_arguments = parsed
        
        if not tool_name:
            return None
        
        args = parse_tool_arguments(tool_arguments)
        
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": f"chatcmpl-tool-{msg.get('id', '')}",
                "name": tool_name,
                "arguments": args,
                "requestor": "assistant"
            }]
        }
    elif role == 'tool-result':
        # Tool response - goes back to the model
        return {
            "role": "tool",
            "content": msg.get('content', ''),
            "tool_calls": None
        }
    elif role == 'system':
        return {
            "role": "system",
            "content": msg.get('content', ''),
            "tool_calls": None
        }
    
    return None
