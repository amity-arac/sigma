# Copyright Amity
"""
Sigma Environments Package

This package provides environment management for the sigma simulator:
- Environment base classes and generic environment creation
- Path constants and file management
- Pydantic models for API operations
"""

from typing import Optional, Union

from sigma.envs.base import Env
from sigma.envs.user import UserStrategy

# Path constants
from sigma.envs.paths import (
    DATA_ENVS_PATH,
    SOURCE_ENVS_PATH,
    ENVS_PATH,
    get_env_data_path,
    get_env_source_path,
    get_env_file_path,
)

# Pydantic models for API
from sigma.envs.models import (
    EnvironmentFileInfo,
    EnvironmentFilesResponse,
    EnvironmentFileContentResponse,
    UpdateEnvironmentFileRequest,
    EnvironmentInfo,
)

# File management operations
from sigma.envs.manager import (
    EDITABLE_ENV_FILES,
    get_editable_files,
    is_file_editable,
    list_env_files,
    get_env_files_response,
    get_env_file,
    get_env_file_response,
    update_env_file,
    env_exists,
    list_available_envs,
)


def get_env(
    env_name: str,
    user_strategy: Union[str, UserStrategy],
    user_model: str,
    task_split: str,
    user_provider: Optional[str] = None,
    task_index: Optional[int] = None,
) -> Env:
    if env_name == "retail":
        from sigma.envs.retail import MockRetailDomainEnv

        return MockRetailDomainEnv(
            user_strategy=user_strategy,
            user_model=user_model,
            task_split=task_split,
            user_provider=user_provider,
            task_index=task_index,
        )
    elif env_name == "airline":
        from sigma.envs.airline import MockAirlineDomainEnv

        return MockAirlineDomainEnv(
            user_strategy=user_strategy,
            user_model=user_model,
            task_split=task_split,
            user_provider=user_provider,
            task_index=task_index,
        )
    else:
        raise ValueError(f"Unknown environment: {env_name}")


__all__ = [
    # Base classes
    "Env",
    "UserStrategy",
    # Path constants
    "DATA_ENVS_PATH",
    "SOURCE_ENVS_PATH",
    "ENVS_PATH",
    "get_env_data_path",
    "get_env_source_path",
    "get_env_file_path",
    # Pydantic models
    "EnvironmentFileInfo",
    "EnvironmentFilesResponse",
    "EnvironmentFileContentResponse",
    "UpdateEnvironmentFileRequest",
    "EnvironmentInfo",
    # File management
    "EDITABLE_ENV_FILES",
    "get_editable_files",
    "is_file_editable",
    "list_env_files",
    "get_env_files_response",
    "get_env_file",
    "get_env_file_response",
    "update_env_file",
    "env_exists",
    "list_available_envs",
    # Environment factory
    "get_env",
]
