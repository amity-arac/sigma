# Copyright Amity
"""
DPO (Direct Preference Optimization) format converter for trajectories.

DPO format:
- prompt: list of messages representing the conversation history up to the decision point
- chosen: list containing the single correct action (what was actually done)
- rejected: list containing the single rejected action (what was proposed but rejected)
"""
import json
from typing import Any, Dict, List, Optional

from .base import parse_tool_call_from_content, serialize_tool_arguments


def convert_to_dpo_format(trajectories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert trajectories to DPO training format.
    
    This function looks for trajectories with rejected suggestions (role='rejected')
    and creates training pairs where the model learns to prefer the chosen action over rejected.
    
    Args:
        trajectories: List of trajectory dictionaries
        
    Returns:
        List of DPO training records
    """
    dpo_records = []
    
    print(f"[DPO] Processing {len(trajectories)} trajectories")
    
    for traj_idx, trajectory in enumerate(trajectories):
        messages = trajectory.get('messages', [])
        wiki = trajectory.get('wiki', '')
        session_id = trajectory.get('session_id', '')
        
        # Count rejected messages in this trajectory
        rejected_count = sum(1 for m in messages if m.get('role') == 'rejected')
        print(f"[DPO] Trajectory {traj_idx} ({session_id[:8] if session_id else 'N/A'}...): {len(messages)} messages, {rejected_count} rejected")
        
        # Build system message
        system_content = _build_system_prompt(wiki)
        
        # Find rejected suggestions and create DPO pairs
        for i, msg in enumerate(messages):
            if msg.get('role') == 'rejected':
                print(f"[DPO]   Found rejection at message {i}")
                # We found a rejection point - create a DPO pair
                rejected_data = msg.get('rejected', {})
                
                # Build conversation history (prompt) up to this point
                prompt = []
                prompt.append({
                    "role": "system",
                    "content": system_content.strip()
                })
                
                for prev_msg in messages[:i]:
                    if prev_msg.get('role') == 'rejected':
                        continue  # Skip previous rejections
                    
                    converted = _convert_message_for_dpo(prev_msg)
                    if converted:
                        prompt.append(converted)
                
                # The chosen action is the one that came after (find next non-rejected message)
                chosen_action = None
                for j in range(i + 1, len(messages)):
                    next_msg = messages[j]
                    if next_msg.get('role') != 'rejected':
                        chosen_action = _convert_message_for_dpo(next_msg)
                        break
                
                # Build rejected action from the rejected suggestion
                rejected_action = _build_rejected_action(rejected_data)
                
                # Only create DPO record if we have valid chosen and rejected actions
                if chosen_action and rejected_action:
                    dpo_records.append({
                        "prompt": prompt,
                        "chosen": [chosen_action],
                        "rejected": [rejected_action]
                    })
    
    return dpo_records


def _build_system_prompt(wiki: str) -> str:
    """Build the system prompt for DPO format."""
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


def _build_rejected_action(rejected_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Build the rejected action from rejected suggestion data."""
    if not rejected_data:
        return None
    
    tool_name = rejected_data.get('tool_name')
    tool_arguments = rejected_data.get('tool_arguments')
    content = rejected_data.get('content')
    reasoning = rejected_data.get('reasoning', '')
    
    # If tool_name is null, try to parse from content field
    if not tool_name and content:
        parsed_tool_name, parsed_args = parse_tool_call_from_content(content)
        if parsed_tool_name:
            tool_name = parsed_tool_name
            tool_arguments = parsed_args
    
    # Only create tool_calls if we have a valid tool_name
    if tool_name:
        args_str = serialize_tool_arguments(tool_arguments)
        
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "function": {
                    "name": tool_name,
                    "arguments": args_str
                },
                "type": "function"
            }],
            "reasoning_content": reasoning
        }
    elif content:
        return {
            "role": "assistant",
            "content": content,
            "reasoning_content": reasoning
        }
    
    return None


def _convert_message_for_dpo(msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert a trajectory message to DPO format."""
    role = msg.get('role', '')
    
    if role == 'user':
        return {
            "role": "user",
            "content": msg.get('content', '')
        }
    elif role == 'agent':
        result = {
            "role": "assistant",
            "content": msg.get('content')
        }
        if msg.get('reasoning'):
            result["reasoning_content"] = msg.get('reasoning')
        # Check if this agent message has tool_calls
        tool_name = msg.get('tool_name')
        tool_arguments = msg.get('tool_arguments')
        if tool_name:
            args_str = serialize_tool_arguments(tool_arguments)
            result["tool_calls"] = [{
                "function": {
                    "name": tool_name,
                    "arguments": args_str
                },
                "type": "function"
            }]
        return result
    elif role == 'tool':
        # Tool call from agent - this represents an assistant making a tool call
        tool_name = msg.get('tool_name')
        tool_arguments = msg.get('tool_arguments')
        
        # If tool_name is not set, try to parse it from the content field
        if not tool_name:
            content = msg.get('content', '')
            tool_name, tool_arguments = parse_tool_call_from_content(content)
        
        # Skip if we still have no tool_name - this would create corrupted data
        if not tool_name:
            return None
        
        args_str = serialize_tool_arguments(tool_arguments)
        
        result = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "function": {
                    "name": tool_name,
                    "arguments": args_str
                },
                "type": "function"
            }]
        }
        if msg.get('reasoning'):
            result["reasoning_content"] = msg.get('reasoning')
        return result
    elif role == 'tool-result':
        # Tool response - this goes back to the model as role "tool"
        tool_call_id = msg.get('id') or msg.get('tool_call_id') or str(msg.get('timestamp', ''))
        tool_name = msg.get('tool_name', '')
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": msg.get('content', '')
        }
    elif role == 'system':
        return {
            "role": "system",
            "content": msg.get('content', '')
        }
    
    return None
