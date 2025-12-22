# Copyright Amity
"""
Trajectory export functionality for training data generation.

This module provides converters for various training data formats:
- DPO (Direct Preference Optimization)
- GRPO (Group Relative Policy Optimization)  
- SFT (Supervised Fine-Tuning)

Usage:
    from sigma.exports import convert_trajectories, ExportTrajectoryRequest, ExportTrajectoryResponse
    
    records = convert_trajectories('dpo', trajectories)
"""
from typing import Any, Dict, List

from .models import ExportTrajectoryRequest, ExportTrajectoryResponse
from .dpo import convert_to_dpo_format
from .grpo import convert_to_grpo_format
from .sft import convert_to_sft_format

__all__ = [
    # Models
    "ExportTrajectoryRequest",
    "ExportTrajectoryResponse",
    # Main API
    "convert_trajectories",
    # Individual converters (for direct use)
    "convert_to_dpo_format",
    "convert_to_grpo_format", 
    "convert_to_sft_format",
]


# Registry of supported export formats
EXPORT_FORMATS = {
    "dpo": convert_to_dpo_format,
    "grpo": convert_to_grpo_format,
    "sft": convert_to_sft_format,
}


def convert_trajectories(format: str, trajectories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert trajectories to the specified training data format.
    
    Args:
        format: The export format ('dpo', 'grpo', or 'sft')
        trajectories: List of trajectory dictionaries
        
    Returns:
        List of training records in the specified format
        
    Raises:
        ValueError: If the format is not supported
    """
    converter = EXPORT_FORMATS.get(format.lower())
    if converter is None:
        supported = ", ".join(EXPORT_FORMATS.keys())
        raise ValueError(f"Unsupported export format: {format}. Supported formats: {supported}")
    
    return converter(trajectories)


def get_supported_formats() -> List[str]:
    """Get list of supported export formats."""
    return list(EXPORT_FORMATS.keys())
