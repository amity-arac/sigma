# Copyright Amity
"""
Centralized path constants for environment management.

This module provides the single source of truth for all path constants
used across the sigma environment system.
"""

import os

# Root of the sigma package
_SIGMA_ROOT = os.path.dirname(os.path.dirname(__file__))

# Root of the project (parent of sigma/)
_PROJECT_ROOT = os.path.dirname(_SIGMA_ROOT)

# Path to data environments folder (data files: db.json, tasks.json, policy.md, etc.)
# This is at root/data/envs, separate from source code
DATA_ENVS_PATH = os.path.join(_PROJECT_ROOT, "data", "envs")

# Path to source environments folder (tools.py only)
# This is at sigma/envs/
SOURCE_ENVS_PATH = os.path.dirname(__file__)

# Legacy alias for backward compatibility
ENVS_PATH = DATA_ENVS_PATH


def get_env_data_path(env_name: str) -> str:
    """Get the data directory path for a specific environment."""
    return os.path.join(DATA_ENVS_PATH, env_name)


def get_env_source_path(env_name: str) -> str:
    """Get the source directory path for a specific environment."""
    return os.path.join(SOURCE_ENVS_PATH, env_name)


def get_env_file_path(env_name: str, filename: str) -> str:
    """Get the full path to a file in an environment's data directory."""
    return os.path.join(DATA_ENVS_PATH, env_name, filename)
