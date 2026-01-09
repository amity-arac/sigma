# Copyright Amity
"""
Raw trajectory format converter.

Exports trajectories as-is in their original format, suitable for archival
or custom processing. Each trajectory is output as a single JSON object
per line (JSONL format).
"""
from typing import Any, Dict, List


def convert_to_raw_format(trajectories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert trajectories to raw format (pass-through).
    
    This is a minimal converter that outputs trajectories in their original
    format without any transformation. Useful for:
    - Archival purposes
    - Custom post-processing pipelines
    - Debugging and inspection
    - Data migration
    
    Args:
        trajectories: List of trajectory dictionaries
        
    Returns:
        List of raw trajectory records (same as input)
    """
    print(f"[RAW] Processing {len(trajectories)} trajectories")
    
    raw_records = []
    
    for idx, trajectory in enumerate(trajectories):
        session_id = trajectory.get('session_id', trajectory.get('id', ''))
        messages = trajectory.get('messages', [])
        reward = trajectory.get('reward')
        
        print(f"[RAW] Trajectory {idx} ({session_id[:8] if session_id else 'N/A'}...): "
              f"{len(messages)} messages, reward={reward}")
        
        # Pass through the trajectory as-is
        raw_records.append(trajectory)
    
    print(f"[RAW] Summary: {len(raw_records)} records exported")
    
    return raw_records
