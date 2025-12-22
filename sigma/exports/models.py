# Copyright Amity
"""
Pydantic models for trajectory export functionality.
"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ExportTrajectoryRequest(BaseModel):
    """Request to export trajectories as training data."""
    format: str  # 'dpo', 'grpo', or 'sft'
    env_name: Optional[str] = None  # Filter by environment
    trajectory_ids: Optional[List[str]] = None  # Optional: specific trajectories to export
    date_filter: Optional[str] = None  # Optional: filter by date (YYYY-MM-DD)


class ExportTrajectoryResponse(BaseModel):
    """Response from exporting trajectories."""
    success: bool
    format: str
    count: int  # Number of records exported
    data: str  # JSONL content as string
    error: Optional[str] = None
