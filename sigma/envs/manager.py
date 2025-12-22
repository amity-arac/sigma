# Copyright Amity
"""
Environment file management operations.

This module provides functions for listing, reading, and updating
environment configuration files (db.json, tasks.json, policy.md, etc.).
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from sigma.envs.paths import DATA_ENVS_PATH, get_env_data_path, get_env_file_path
from sigma.envs.models import (
    EnvironmentFileInfo,
    EnvironmentFilesResponse,
    EnvironmentFileContentResponse,
)


# =============================================================================
# Editable Files Configuration
# =============================================================================

# Define which files are editable and their metadata
EDITABLE_ENV_FILES: Dict[str, Dict[str, str]] = {
    "db.json": {
        "type": "json",
        "display_name": "Database",
        "description": "Database containing users, products, and orders data"
    },
    "tasks.json": {
        "type": "json",
        "display_name": "Tasks",
        "description": "Tasks with user scenarios and evaluation criteria"
    },
    "policy.md": {
        "type": "markdown",
        "display_name": "Policy",
        "description": "Agent policy and behavioral rules"
    },
    "user_guidelines.md": {
        "type": "markdown",
        "display_name": "User Guidelines",
        "description": "User simulation guidelines"
    },
    "agent_guidelines.md": {
        "type": "markdown",
        "display_name": "Agent Guidelines",
        "description": "Agent-specific guidelines"
    },
    "tools.py": {
        "type": "python",
        "display_name": "Tools",
        "description": "Python tool implementations for the environment"
    },
}


def get_editable_files() -> Dict[str, Dict[str, str]]:
    """Get the configuration of editable environment files."""
    return EDITABLE_ENV_FILES.copy()


def is_file_editable(filename: str) -> bool:
    """Check if a file is in the editable files list."""
    return filename in EDITABLE_ENV_FILES


# =============================================================================
# File Listing
# =============================================================================

def list_env_files(env_name: str) -> Tuple[List[EnvironmentFileInfo], Optional[str]]:
    """
    List editable files in an environment.
    
    Args:
        env_name: Name of the environment
        
    Returns:
        Tuple of (list of file info, error message or None)
    """
    env_path = get_env_data_path(env_name)
    
    if not os.path.exists(env_path):
        return [], f"Environment '{env_name}' not found"
    
    files = []
    for filename, info in EDITABLE_ENV_FILES.items():
        file_path = os.path.join(env_path, filename)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            files.append(EnvironmentFileInfo(
                name=filename,
                type=info["type"],
                size=size,
                display_name=info["display_name"],
                description=info["description"]
            ))
    
    return files, None


def get_env_files_response(env_name: str) -> Tuple[Optional[EnvironmentFilesResponse], Optional[str]]:
    """
    Get environment files as a response object.
    
    Args:
        env_name: Name of the environment
        
    Returns:
        Tuple of (response object or None, error message or None)
    """
    files, error = list_env_files(env_name)
    if error:
        return None, error
    
    return EnvironmentFilesResponse(env_name=env_name, files=files), None


# =============================================================================
# File Reading
# =============================================================================

def get_env_file(env_name: str, filename: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Get content of an environment file.
    
    Args:
        env_name: Name of the environment
        filename: Name of the file to read
        
    Returns:
        Tuple of (content or None, file type or None, error message or None)
    """
    if not is_file_editable(filename):
        return None, None, f"File '{filename}' is not editable"
    
    file_path = get_env_file_path(env_name, filename)
    
    if not os.path.exists(file_path):
        return None, None, f"File '{filename}' not found in environment '{env_name}'"
    
    try:
        with open(file_path, "r") as f:
            content = f.read()
        file_type = EDITABLE_ENV_FILES[filename]["type"]
        return content, file_type, None
    except Exception as e:
        return None, None, str(e)


def get_env_file_response(
    env_name: str, 
    filename: str
) -> Tuple[Optional[EnvironmentFileContentResponse], Optional[str]]:
    """
    Get environment file content as a response object.
    
    Args:
        env_name: Name of the environment
        filename: Name of the file to read
        
    Returns:
        Tuple of (response object or None, error message or None)
    """
    content, file_type, error = get_env_file(env_name, filename)
    if error:
        return None, error
    
    return EnvironmentFileContentResponse(
        env_name=env_name,
        filename=filename,
        content=content,
        type=file_type
    ), None


# =============================================================================
# File Writing
# =============================================================================

def update_env_file(env_name: str, filename: str, content: str) -> Tuple[bool, Optional[str]]:
    """
    Update content of an environment file.
    
    Args:
        env_name: Name of the environment
        filename: Name of the file to update
        content: New content for the file
        
    Returns:
        Tuple of (success boolean, error message or None)
    """
    if not is_file_editable(filename):
        return False, f"File '{filename}' is not editable"
    
    env_path = get_env_data_path(env_name)
    
    if not os.path.exists(env_path):
        return False, f"Environment '{env_name}' not found"
    
    file_path = get_env_file_path(env_name, filename)
    file_info = EDITABLE_ENV_FILES[filename]
    
    try:
        # Validate JSON files
        if file_info["type"] == "json":
            try:
                json.loads(content)
            except json.JSONDecodeError as e:
                return False, f"Invalid JSON: {str(e)}"
        
        # Write the file
        with open(file_path, "w") as f:
            f.write(content)
        
        return True, None
    except Exception as e:
        return False, str(e)


# =============================================================================
# Environment Existence Check
# =============================================================================

def env_exists(env_name: str) -> bool:
    """Check if an environment exists."""
    env_path = get_env_data_path(env_name)
    return os.path.exists(env_path)


def list_available_envs() -> List[str]:
    """List all available environment names from the data folder."""
    if not os.path.exists(DATA_ENVS_PATH):
        return []
    
    envs = []
    for name in os.listdir(DATA_ENVS_PATH):
        env_path = os.path.join(DATA_ENVS_PATH, name)
        if os.path.isdir(env_path):
            # Check for required files
            db_path = os.path.join(env_path, "db.json")
            tasks_path = os.path.join(env_path, "tasks.json")
            if os.path.exists(db_path) and os.path.exists(tasks_path):
                envs.append(name)
    
    return sorted(envs)
