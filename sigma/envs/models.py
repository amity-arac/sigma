# Copyright Amity
"""
Pydantic models for environment file management API.

These models define the request/response schemas for environment
file operations exposed through the API.
"""

from typing import List
from pydantic import BaseModel


class EnvironmentFileInfo(BaseModel):
    """Information about a file in an environment."""
    name: str
    type: str  # 'json', 'markdown', 'text', 'python'
    size: int
    display_name: str
    description: str


class EnvironmentFilesResponse(BaseModel):
    """Response containing list of environment files."""
    env_name: str
    files: List[EnvironmentFileInfo]


class EnvironmentFileContentResponse(BaseModel):
    """Response containing file content."""
    env_name: str
    filename: str
    content: str
    type: str


class UpdateEnvironmentFileRequest(BaseModel):
    """Request to update an environment file."""
    content: str


class EnvironmentInfo(BaseModel):
    """Information about an available environment."""
    name: str
    display_name: str
    description: str
