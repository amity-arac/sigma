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
        "description": "Agent policy, guidelines, and behavioral rules"
    },
    "user_guidelines.md": {
        "type": "markdown",
        "display_name": "User Guidelines",
        "description": "User simulation guidelines"
    },
    "scenario_generator_guidelines.md": {
        "type": "markdown",
        "display_name": "Scenario Generator Guidelines",
        "description": "Guidelines for generating valid, solvable test scenarios"
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
                description=info["description"],
                editable=True
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


# =============================================================================
# Environment Duplication and Renaming
# =============================================================================

def validate_env_name(env_name: str) -> Tuple[bool, Optional[str]]:
    """
    Validate an environment name.
    
    Args:
        env_name: The name to validate
        
    Returns:
        Tuple of (is_valid, error_message or None)
    """
    if not env_name:
        return False, "Environment name cannot be empty"
    
    # Only allow alphanumeric, underscore, and hyphen
    import re
    if not re.match(r'^[a-zA-Z][a-zA-Z0-9_-]*$', env_name):
        return False, "Environment name must start with a letter and contain only letters, numbers, underscores, and hyphens"
    
    # Check length
    if len(env_name) > 64:
        return False, "Environment name must be 64 characters or less"
    
    return True, None


def duplicate_env(source_env: str, target_env: str) -> Tuple[bool, Optional[str]]:
    """
    Duplicate an environment to a new name.
    
    Args:
        source_env: Name of the environment to duplicate
        target_env: Name for the new environment
        
    Returns:
        Tuple of (success boolean, error message or None)
    """
    import shutil
    
    # Validate target name
    is_valid, error = validate_env_name(target_env)
    if not is_valid:
        return False, error
    
    # Check source exists
    source_path = get_env_data_path(source_env)
    if not os.path.exists(source_path):
        return False, f"Source environment '{source_env}' not found"
    
    # Check target doesn't exist
    target_path = get_env_data_path(target_env)
    if os.path.exists(target_path):
        return False, f"Environment '{target_env}' already exists"
    
    try:
        # Copy the entire directory
        shutil.copytree(source_path, target_path)
        
        # Remove __pycache__ if it was copied
        pycache_path = os.path.join(target_path, "__pycache__")
        if os.path.exists(pycache_path):
            shutil.rmtree(pycache_path)
        
        return True, None
    except Exception as e:
        # Clean up on failure
        if os.path.exists(target_path):
            shutil.rmtree(target_path)
        return False, f"Failed to duplicate environment: {str(e)}"


def rename_env(old_name: str, new_name: str) -> Tuple[bool, Optional[str]]:
    """
    Rename an environment.
    
    Args:
        old_name: Current name of the environment
        new_name: New name for the environment
        
    Returns:
        Tuple of (success boolean, error message or None)
    """
    # Validate new name
    is_valid, error = validate_env_name(new_name)
    if not is_valid:
        return False, error
    
    # Check source exists
    old_path = get_env_data_path(old_name)
    if not os.path.exists(old_path):
        return False, f"Environment '{old_name}' not found"
    
    # Check target doesn't exist
    new_path = get_env_data_path(new_name)
    if os.path.exists(new_path):
        return False, f"Environment '{new_name}' already exists"
    
    try:
        os.rename(old_path, new_path)
        return True, None
    except Exception as e:
        return False, f"Failed to rename environment: {str(e)}"


def delete_env(env_name: str) -> Tuple[bool, Optional[str]]:
    """
    Delete an environment.
    
    Args:
        env_name: Name of the environment to delete
        
    Returns:
        Tuple of (success boolean, error message or None)
    """
    import shutil
    
    # Check if this is the only environment
    available_envs = list_available_envs()
    if len(available_envs) <= 1:
        return False, "Cannot delete the last remaining environment"
    
    # Check environment exists
    env_path = get_env_data_path(env_name)
    if not os.path.exists(env_path):
        return False, f"Environment '{env_name}' not found"
    
    try:
        shutil.rmtree(env_path)
        return True, None
    except Exception as e:
        return False, f"Failed to delete environment: {str(e)}"
