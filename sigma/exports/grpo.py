# Copyright Amity
"""
GRPO (Group Relative Policy Optimization) format converter for trajectories.

Exports trajectories in a format matching tasks.json schema so it can be used 
directly as new tasks for training or evaluation.
"""
import json
from typing import Any, Dict, List, Optional


def convert_to_grpo_format(trajectories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert trajectories to GRPO training format matching tasks.json structure.
    
    The exported format matches the tasks.json schema:
    
    {
        "id": "string",
        "description": { "purpose": null, "relevant_policies": null, "notes": null },
        "user_scenario": {
            "persona": null,
            "instructions": {
                "task_instructions": "...",
                "domain": "retail",
                "reason_for_call": "...",
                "known_info": "...",
                "unknown_info": "..."
            }
        },
        "db": {
            "users": { "user_id": {...} },
            "orders": { "#W123": {...} },
            "products": { "prod_id": {...} }
        },
        "evaluation_criteria": {
            "actions": [...],  # Ground truth: actual tool calls from trajectory
            "communicate_info": [...],
            "nl_assertions": null
        }
    }
    
    Args:
        trajectories: List of trajectory dictionaries
        
    Returns:
        List of GRPO training records
    """
    grpo_records = []
    
    print(f"[GRPO] Processing {len(trajectories)} trajectories")
    
    for idx, trajectory in enumerate(trajectories):
        messages = trajectory.get('messages', [])
        task_instruction = trajectory.get('task_instruction', '')
        user_id = trajectory.get('user_id', '')
        env_name = trajectory.get('env_name', '')
        session_id = trajectory.get('session_id', trajectory.get('id', ''))
        reward = trajectory.get('reward', 0)
        persona = trajectory.get('persona', '')
        persona_data = trajectory.get('persona_data', {})
        
        # Extract tool call sequence from the actual trajectory (ground truth)
        actions = _extract_actions(messages, len(grpo_records))
        
        print(f"[GRPO] Trajectory {idx} ({session_id[:8] if session_id else 'N/A'}...): reward={reward}, {len(actions)} tool calls")
        
        # Skip trajectories without any actions (no function calls)
        if not actions:
            print(f"[GRPO] Skipping trajectory {idx} - no actions ground truth (no function calls)")
            continue
        
        # Extract user info and build known/unknown info strings
        known_info, unknown_info = _build_info_strings(persona_data)
        
        # Extract communicate_info from reward_info if available
        communicate_info = _extract_communicate_info(trajectory.get('reward_info', {}))
        
        # Build db data from persona_data for GRPO training
        db_data = _build_db_data(persona_data, user_id)
        
        # Create GRPO record matching tasks.json format
        record_id = str(len(grpo_records))
        grpo_record = {
            "id": record_id,
            "description": {
                "purpose": f"Generated from trajectory {session_id[:8] if session_id else 'unknown'}",
                "relevant_policies": None,
                "notes": f"Reward: {reward}" if reward is not None else None
            },
            "user_scenario": {
                "persona": None,
                "instructions": {
                    "task_instructions": persona or ".",
                    "domain": env_name or "retail",
                    "reason_for_call": task_instruction or "",
                    "known_info": known_info,
                    "unknown_info": unknown_info
                }
            },
            "db": db_data,
            "evaluation_criteria": {
                "actions": actions,
                "communicate_info": communicate_info,
                "nl_assertions": None
            }
        }
        
        grpo_records.append(grpo_record)
    
    skipped_count = len(trajectories) - len(grpo_records)
    print(f"[GRPO] Summary: {len(grpo_records)} records created, {skipped_count} skipped (no actions ground truth)")
    
    return grpo_records


def _extract_actions(messages: List[Dict[str, Any]], record_index: int) -> List[Dict[str, Any]]:
    """Extract tool call sequence from messages as ground truth actions."""
    actions = []
    action_counter = 0
    
    for msg in messages:
        if msg.get('role') == 'tool':
            tool_name = msg.get('tool_name')
            tool_args = msg.get('tool_arguments', {})
            
            # If tool_name is not set, try to parse from content
            if not tool_name and msg.get('content'):
                content = msg.get('content', '')
                if content.startswith('🔧 Calling '):
                    try:
                        # Extract tool name from first line
                        first_line = content.split('\n')[0]
                        tool_name = first_line.replace('🔧 Calling ', '').strip()
                        
                        # Extract JSON arguments from the rest
                        json_start = content.find('{')
                        if json_start != -1:
                            json_str = content[json_start:]
                            tool_args = json.loads(json_str)
                    except (json.JSONDecodeError, IndexError, ValueError):
                        pass
            
            if tool_name:
                action = {
                    "action_id": f"{record_index}_{action_counter}",
                    "name": tool_name,
                    "arguments": tool_args if tool_args else {},
                    "info": None
                }
                actions.append(action)
                action_counter += 1
    
    return actions


def _build_info_strings(persona_data: Optional[Dict[str, Any]]) -> tuple[str, str]:
    """Build known_info and unknown_info strings from persona_data."""
    if not persona_data:
        return "", ""
    
    user_info = persona_data.get('user', {})
    user_name = user_info.get('name', {})
    first_name = user_name.get('first_name', '')
    last_name = user_name.get('last_name', '')
    user_address = user_info.get('address', {})
    zip_code = user_address.get('zip', '')
    email = user_info.get('email', '')
    
    # Build known_info string
    known_info_parts = []
    if first_name and last_name:
        known_info_parts.append(f"You are {first_name} {last_name}")
        if zip_code:
            known_info_parts.append(f"in zip code {zip_code}")
    known_info = " ".join(known_info_parts) + "." if known_info_parts else ""
    
    # Build unknown_info (email is commonly "forgotten")
    unknown_info = "You do not remember your email address." if email else ""
    
    return known_info, unknown_info


def _extract_communicate_info(reward_info: Optional[Dict[str, Any]]) -> List[str]:
    """Extract communicate_info from reward_info if available."""
    if not reward_info or 'outputs' not in reward_info:
        return []
    
    outputs = reward_info.get('outputs', {})
    return list(outputs.keys()) if isinstance(outputs, dict) else []


def _build_db_data(persona_data: Optional[Dict[str, Any]], user_id: str) -> Optional[Dict[str, Any]]:
    """
    Build db data from persona_data for GRPO training.
    
    This includes user profile, orders/reservations, and augmented_data (products, etc.)
    so the simulator can insert this data before starting simulations.
    """
    if not persona_data:
        return None
    
    db_data = {}
    
    # Include user profile data
    if 'user' in persona_data:
        db_data['users'] = {
            persona_data['user'].get('user_id', user_id): persona_data['user']
        }
    
    # Include orders/reservations data (based on data_key from scenario)
    for data_key in ['orders', 'reservations', 'bookings', 'tickets']:
        if data_key in persona_data and persona_data[data_key]:
            db_data[data_key] = persona_data[data_key]
    
    # Include augmented_data (products, flights, etc. that were injected into DB)
    augmented_data = persona_data.get('augmented_data', {})
    if augmented_data:
        for collection_name, collection_data in augmented_data.items():
            if collection_data:
                db_data[collection_name] = collection_data
    
    return db_data if db_data else None
