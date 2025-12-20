# Copyright Amity
"""
Trajectory Storage Module

This module provides functionality to save simulation session trajectories
for later analytics and transformation into model training data.

Supports multiple storage backends:
- Azure Blob Storage (recommended for analytics/ML pipelines)
- Local filesystem (for development/testing)

Key design decisions:
1. Rejected suggestions are stored INLINE with messages (role='rejected') 
   to preserve the sequence and know exactly when they occurred
2. All reasoning is captured for each action
3. Schema is optimized for easy transformation to training data formats

Environment variables (loaded from .env file):
- For Blob Storage (recommended):
  - AZURE_STORAGE_CONNECTION_STRING or
  - AZURE_STORAGE_ACCOUNT_NAME + AZURE_STORAGE_ACCOUNT_KEY
  - AZURE_STORAGE_CONTAINER (default: "trajectories")
"""

import os
import json
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables from .env file
load_dotenv()

# Azure Blob Storage imports
try:
    from azure.storage.blob import BlobServiceClient
    BLOB_AVAILABLE = True
except ImportError:
    BLOB_AVAILABLE = False


# =============================================================================
# Data Models
# =============================================================================

class RejectedSuggestion(BaseModel):
    """
    Data for a rejected suggestion, using the same format as a normal message.
    """
    content: Optional[str] = None
    reasoning: Optional[str] = None
    tool_name: Optional[str] = None
    tool_arguments: Optional[Dict[str, Any]] = None


class TrajectoryMessage(BaseModel):
    """
    A single message/event in the trajectory.
    
    The role field determines the message type:
    - 'user': Message from the simulated user
    - 'agent': Response from the human agent (includes reasoning)
    - 'tool': Tool call made by the agent (includes tool_name and tool_arguments)
    - 'tool-result': Result returned from a tool call
    - 'system': System message
    - 'rejected': A suggestion that was rejected by the user (preserves sequence)
    
    For rejected suggestions (role='rejected'), the `rejected` field contains
    the suggested action in the same format as normal messages.
    """
    id: str
    role: Literal['user', 'agent', 'tool', 'tool-result', 'system', 'rejected']
    content: Optional[str] = None
    reasoning: Optional[str] = None
    timestamp: Optional[str] = None
    
    # For tool calls (role='tool')
    tool_name: Optional[str] = None
    tool_arguments: Optional[Dict[str, Any]] = None
    
    # For rejected suggestions (role='rejected')
    # Contains the rejected action in the same format as normal messages
    rejected: Optional[RejectedSuggestion] = None


class TrajectoryData(BaseModel):
    """
    Complete trajectory data to be saved.
    
    This schema is designed for:
    1. Analytics - easy to query and aggregate
    2. Training data transformation - convert to chat format, DPO pairs, etc.
    3. Debugging - full context of what happened in a session
    """
    id: Optional[str] = None
    session_id: str
    created_at: Optional[str] = None
    
    # Environment info
    env_name: str
    task_index: Optional[int] = None
    task_split: Optional[str] = None
    task_instruction: Optional[str] = None
    
    # User info (for GRPO training data)
    user_id: Optional[str] = None
    
    # Model info
    user_model: str
    user_provider: str
    agent_model: Optional[str] = None
    agent_provider: Optional[str] = None
    
    # Session data
    persona: str
    wiki: str
    
    # Conversation (includes rejected suggestions inline with role='rejected')
    messages: List[TrajectoryMessage]
    
    # Final result
    is_done: bool = False
    reward: Optional[float] = None
    reward_info: Optional[Dict[str, Any]] = None
    expected_actions: Optional[List[Dict[str, Any]]] = None


class TrajectoryStorageError(Exception):
    """Custom exception for trajectory storage errors."""
    pass


# =============================================================================
# Storage Backends
# =============================================================================

class StorageBackend(ABC):
    """Abstract base class for storage backends."""
    
    @abstractmethod
    def save(self, trajectory: TrajectoryData) -> str:
        """Save a trajectory and return its ID."""
        pass
    
    @abstractmethod
    def get(self, trajectory_id: str, env_name: str) -> Optional[Dict[str, Any]]:
        """Get a trajectory by ID."""
        pass
    
    @abstractmethod
    def list(self, env_name: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """List trajectories."""
        pass
    
    @abstractmethod
    def delete(self, trajectory_id: str, env_name: str) -> bool:
        """Delete a trajectory."""
        pass
    
    @abstractmethod
    def update(self, trajectory_id: str, env_name: str, updates: Dict[str, Any]) -> bool:
        """Update specific fields of a trajectory."""
        pass


class LocalStorageBackend(StorageBackend):
    """
    Local filesystem storage backend.
    
    Stores trajectories as JSON files organized by environment and date:
    {base_path}/{env_name}/{date}/{trajectory_id}.json
    
    Best for: Development, testing, small-scale usage
    """
    
    def __init__(self, base_path: str = "./trajectories"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def _get_path(self, trajectory_id: str, env_name: str) -> Path:
        """Get the file path for a trajectory."""
        dir_path = self.base_path / env_name
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path / f"{trajectory_id}.json"
    
    def save(self, trajectory: TrajectoryData) -> str:
        if not trajectory.id:
            trajectory.id = str(uuid.uuid4())
        if not trajectory.created_at:
            trajectory.created_at = datetime.utcnow().isoformat() + 'Z'
        
        file_path = self._get_path(trajectory.id, trajectory.env_name)
        
        with open(file_path, 'w') as f:
            json.dump(trajectory.model_dump(), f, indent=2)
        
        return trajectory.id
    
    def get(self, trajectory_id: str, env_name: str) -> Optional[Dict[str, Any]]:
        env_path = self.base_path / env_name
        if not env_path.exists():
            return None
        
        file_path = env_path / f"{trajectory_id}.json"
        if file_path.exists():
            with open(file_path, 'r') as f:
                return json.load(f)
        return None
    
    def list(self, env_name: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        results = []
        
        if env_name:
            env_paths = [self.base_path / env_name]
        else:
            env_paths = [p for p in self.base_path.iterdir() if p.is_dir()]
        
        for env_path in env_paths:
            if not env_path.exists():
                continue
            for file_path in sorted(env_path.glob("*.json"), reverse=True):
                if len(results) >= limit:
                    return results
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    results.append({
                        "id": data.get("id"),
                        "session_id": data.get("session_id"),
                        "env_name": data.get("env_name"),
                        "created_at": data.get("created_at"),
                        "is_done": data.get("is_done"),
                        "reward": data.get("reward"),
                        "user_id": data.get("user_id"),
                    })
        
        return results
    
    def delete(self, trajectory_id: str, env_name: str) -> bool:
        env_path = self.base_path / env_name
        if not env_path.exists():
            return False
        
        file_path = env_path / f"{trajectory_id}.json"
        if file_path.exists():
            file_path.unlink()
            return True
        return False
    
    def update(self, trajectory_id: str, env_name: str, updates: Dict[str, Any]) -> bool:
        """Update specific fields of a trajectory."""
        env_path = self.base_path / env_name
        if not env_path.exists():
            return False
        
        file_path = env_path / f"{trajectory_id}.json"
        if not file_path.exists():
            return False
        
        # Read existing data
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Apply updates
        for key, value in updates.items():
            data[key] = value
        
        # Write back
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return True


class BlobStorageBackend(StorageBackend):
    """
    Azure Blob Storage backend.
    
    Stores trajectories as JSON blobs organized by environment and date:
    {container}/{env_name}/{date}/{trajectory_id}.json
    
    Best for: Production, analytics pipelines, ML training data
    
    Advantages over CosmosDB:
    - Much cheaper for storage
    - Easy to process with pandas, DuckDB, Spark
    - Direct integration with Azure ML, Databricks
    - Simple to download/export for offline processing
    """
    
    def __init__(
        self,
        connection_string: Optional[str] = None,
        account_name: Optional[str] = None,
        account_key: Optional[str] = None,
        container_name: Optional[str] = None,
    ):
        if not BLOB_AVAILABLE:
            raise TrajectoryStorageError(
                "azure-storage-blob not available. Install with: pip install azure-storage-blob"
            )
        
        self.connection_string = connection_string or os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        self.account_name = account_name or os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
        self.account_key = account_key or os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
        self.container_name = container_name or os.getenv("AZURE_STORAGE_CONTAINER", "trajectories")
        
        self._container_client = None
    
    def _get_client(self):
        """Get or create blob container client."""
        if self._container_client is None:
            if self.connection_string:
                blob_service = BlobServiceClient.from_connection_string(self.connection_string)
            elif self.account_name and self.account_key:
                blob_service = BlobServiceClient(
                    account_url=f"https://{self.account_name}.blob.core.windows.net",
                    credential=self.account_key
                )
            else:
                raise TrajectoryStorageError(
                    "Azure Blob Storage not configured. Set AZURE_STORAGE_CONNECTION_STRING "
                    "or AZURE_STORAGE_ACCOUNT_NAME + AZURE_STORAGE_ACCOUNT_KEY"
                )
            
            self._container_client = blob_service.get_container_client(self.container_name)
            
            # Create container if it doesn't exist
            try:
                self._container_client.create_container()
            except Exception:
                pass  # Container already exists
        
        return self._container_client
    
    def _get_blob_name(self, trajectory_id: str, env_name: str) -> str:
        """Get the blob name for a trajectory."""
        return f"{env_name}/{trajectory_id}.json"
    
    def save(self, trajectory: TrajectoryData) -> str:
        if not trajectory.id:
            trajectory.id = str(uuid.uuid4())
        if not trajectory.created_at:
            trajectory.created_at = datetime.utcnow().isoformat() + 'Z'
        
        container = self._get_client()
        blob_name = self._get_blob_name(trajectory.id, trajectory.env_name)
        
        data = json.dumps(trajectory.model_dump(), indent=2)
        container.upload_blob(name=blob_name, data=data, overwrite=True)
        
        return trajectory.id
    
    def get(self, trajectory_id: str, env_name: str) -> Optional[Dict[str, Any]]:
        container = self._get_client()
        
        prefix = f"{env_name}/"
        for blob in container.list_blobs(name_starts_with=prefix):
            if trajectory_id in blob.name:
                blob_client = container.get_blob_client(blob.name)
                data = blob_client.download_blob().readall()
                return json.loads(data)
        
        return None
    
    def list(self, env_name: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        container = self._get_client()
        results = []
        
        prefix = f"{env_name}/" if env_name else ""
        
        for blob in container.list_blobs(name_starts_with=prefix):
            if len(results) >= limit:
                break
            if blob.name.endswith('.json'):
                parts = blob.name.replace('.json', '').split('/')
                if len(parts) >= 2:
                    results.append({
                        "id": parts[-1],
                        "env_name": parts[0],
                        "blob_name": blob.name,
                    })
        
        return results
    
    def delete(self, trajectory_id: str, env_name: str) -> bool:
        container = self._get_client()
        
        prefix = f"{env_name}/"
        for blob in container.list_blobs(name_starts_with=prefix):
            if trajectory_id in blob.name:
                container.delete_blob(blob.name)
                return True
        
        return False
    
    def update(self, trajectory_id: str, env_name: str, updates: Dict[str, Any]) -> bool:
        """Update specific fields of a trajectory."""
        container = self._get_client()
        
        prefix = f"{env_name}/"
        for blob in container.list_blobs(name_starts_with=prefix):
            if trajectory_id in blob.name:
                # Read existing data
                blob_client = container.get_blob_client(blob.name)
                data = json.loads(blob_client.download_blob().readall())
                
                # Apply updates
                for key, value in updates.items():
                    data[key] = value
                
                # Write back
                container.upload_blob(name=blob.name, data=json.dumps(data, indent=2), overwrite=True)
                return True
        
        return False


# =============================================================================
# Main Storage Interface
# =============================================================================

class TrajectoryStorage:
    """
    Main trajectory storage interface.
    
    Supports multiple backends:
    - 'blob': Azure Blob Storage (default when configured - recommended for production/analytics)
    - 'local': Local filesystem (default when blob not configured)
    
    Auto-detection priority:
    1. Azure Blob Storage (if AZURE_STORAGE_CONNECTION_STRING or AZURE_STORAGE_ACCOUNT_NAME set)
    2. Local filesystem (always available)
    
    Example:
        # Auto-detect from environment
        storage = TrajectoryStorage()
        
        # Explicit backend
        storage = TrajectoryStorage(backend='blob')
        storage = TrajectoryStorage(backend='local', local_path='./my_trajectories')
    """
    
    def __init__(
        self,
        backend: Optional[Literal['local', 'blob']] = None,
        local_path: str = "./trajectories",
        **kwargs
    ):
        # Auto-detect backend from environment if not specified
        if backend is None:
            if os.getenv("AZURE_STORAGE_CONNECTION_STRING") or os.getenv("AZURE_STORAGE_ACCOUNT_NAME"):
                backend = 'blob'
            else:
                backend = 'local'
        
        self.backend_type = backend
        
        if backend == 'local':
            self._backend = LocalStorageBackend(base_path=local_path)
        elif backend == 'blob':
            self._backend = BlobStorageBackend(**kwargs)
        else:
            raise ValueError(f"Unknown backend: {backend}. Supported: 'local', 'blob'")
    
    def save(self, trajectory: TrajectoryData) -> str:
        """Save a trajectory and return its ID."""
        return self._backend.save(trajectory)
    
    def get(self, trajectory_id: str, env_name: str) -> Optional[Dict[str, Any]]:
        """Get a trajectory by ID."""
        return self._backend.get(trajectory_id, env_name)
    
    def list(self, env_name: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """List trajectories."""
        return self._backend.list(env_name, limit)
    
    def delete(self, trajectory_id: str, env_name: str) -> bool:
        """Delete a trajectory."""
        return self._backend.delete(trajectory_id, env_name)
    
    def update(self, trajectory_id: str, env_name: str, updates: Dict[str, Any]) -> bool:
        """Update specific fields of a trajectory."""
        return self._backend.update(trajectory_id, env_name, updates)


# =============================================================================
# Helper Functions
# =============================================================================

# Singleton instance (lazy-loaded)
_storage_instance: Optional[TrajectoryStorage] = None


def get_trajectory_storage() -> TrajectoryStorage:
    """
    Get the trajectory storage instance (singleton).
    
    Returns:
        TrajectoryStorage instance (auto-configured from environment)
    """
    global _storage_instance
    
    if _storage_instance is None:
        _storage_instance = TrajectoryStorage()
    
    return _storage_instance


def is_storage_configured() -> bool:
    """Check if any storage backend is configured (always True since local is available)."""
    return True


def get_configured_backend() -> str:
    """Get the name of the configured backend."""
    if os.getenv("AZURE_STORAGE_CONNECTION_STRING") or os.getenv("AZURE_STORAGE_ACCOUNT_NAME"):
        return 'blob'
    else:
        return 'local'


def check_storage_configuration() -> Dict[str, Any]:
    """
    Check the storage configuration status.
    
    Returns:
        Dict with configuration status for all backends
    """
    return {
        "configured_backend": get_configured_backend(),
        "local": {
            "available": True,
        },
        "blob": {
            "sdk_available": BLOB_AVAILABLE,
            "connection_string_set": bool(os.getenv("AZURE_STORAGE_CONNECTION_STRING")),
            "account_name_set": bool(os.getenv("AZURE_STORAGE_ACCOUNT_NAME")),
            "container": os.getenv("AZURE_STORAGE_CONTAINER", "trajectories"),
        },
    }


# Legacy compatibility
def is_trajectory_storage_available() -> bool:
    """Check if trajectory storage is available (always True)."""
    return True
