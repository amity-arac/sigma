# Copyright Amity
"""
Trajectory Storage Module

This module provides functionality to save simulation trajectories to Azure Blob Storage.
It stores complete session data including:
- Task instructions
- User messages
- Environment information
- Agent responses with reasoning
- Tool requests and responses
- Rejected suggestions (inline with messages)

Usage:
    from sigma.trajectory_storage import TrajectoryStorage
    
    storage = TrajectoryStorage()
    trajectory_id = storage.save_trajectory(trajectory_data)
"""

import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Literal

from pydantic import BaseModel


class TrajectoryMessage(BaseModel):
    """A single message/event in the trajectory."""
    id: str
    role: Literal['user', 'agent', 'tool', 'tool-result', 'system', 'rejected']
    content: Optional[str] = None
    reasoning: Optional[str] = None
    timestamp: Optional[str] = None
    
    # For tool calls
    tool_name: Optional[str] = None
    tool_arguments: Optional[Dict[str, Any]] = None
    
    # For rejected suggestions (role='rejected')
    rejected_action_type: Optional[str] = None  # 'respond' or 'tool_call'
    rejected_content: Optional[str] = None      # The rejected response text
    rejected_tool_name: Optional[str] = None    # The rejected tool name
    rejected_tool_arguments: Optional[Dict[str, Any]] = None  # The rejected tool args


class TrajectoryData(BaseModel):
    """Complete trajectory data to be saved."""
    id: Optional[str] = None
    session_id: str
    created_at: Optional[str] = None
    
    # Environment info
    env_name: str
    task_index: Optional[int] = None
    task_split: Optional[str] = None
    task_instruction: Optional[str] = None
    
    # Model info
    user_model: str
    user_provider: str
    agent_model: Optional[str] = None
    agent_provider: Optional[str] = None
    
    # Session data
    persona: str
    wiki: str
    
    # Conversation (includes rejected suggestions inline)
    messages: List[TrajectoryMessage]
    
    # Final result
    is_done: bool = False
    reward: Optional[float] = None
    reward_info: Optional[Dict[str, Any]] = None
    expected_actions: Optional[List[Dict[str, Any]]] = None


class CosmosDBStorage:
    """
    CosmosDB storage handler for trajectory data.
    
    Environment variables required:
    - COSMOSDB_ENDPOINT: The CosmosDB account endpoint URL
    - COSMOSDB_KEY: The CosmosDB account key (or use DefaultAzureCredential)
    - COSMOSDB_DATABASE: The database name (default: 'sigma')
    - COSMOSDB_CONTAINER: The container name (default: 'trajectories')
    """
    
    def __init__(
        self,
        endpoint: Optional[str] = None,
        key: Optional[str] = None,
        database_name: Optional[str] = None,
        container_name: Optional[str] = None,
        use_default_credential: bool = False,
    ):
        """
        Initialize CosmosDB storage.
        
        Args:
            endpoint: CosmosDB endpoint URL (or set COSMOSDB_ENDPOINT env var)
            key: CosmosDB key (or set COSMOSDB_KEY env var)
            database_name: Database name (default: 'sigma')
            container_name: Container name (default: 'trajectories')
            use_default_credential: Use Azure DefaultAzureCredential instead of key
        """
        self.endpoint = endpoint or os.getenv("COSMOSDB_ENDPOINT")
        self.key = key or os.getenv("COSMOSDB_KEY")
        self.database_name = database_name or os.getenv("COSMOSDB_DATABASE", "sigma")
        self.container_name = container_name or os.getenv("COSMOSDB_CONTAINER", "trajectories")
        self.use_default_credential = use_default_credential
        
        self._client = None
        self._database = None
        self._container = None
    
    def _get_client(self):
        """Get or create CosmosDB client."""
        if self._client is None:
            try:
                from azure.cosmos import CosmosClient, PartitionKey
                from azure.cosmos.exceptions import CosmosResourceExistsError
            except ImportError:
                raise ImportError(
                    "azure-cosmos package is required. Install with: pip install azure-cosmos"
                )
            
            if not self.endpoint:
                raise ValueError(
                    "CosmosDB endpoint not configured. "
                    "Set COSMOSDB_ENDPOINT environment variable or pass endpoint parameter."
                )
            
            if self.use_default_credential:
                try:
                    from azure.identity import DefaultAzureCredential
                    credential = DefaultAzureCredential()
                    self._client = CosmosClient(self.endpoint, credential=credential)
                except ImportError:
                    raise ImportError(
                        "azure-identity package is required for DefaultAzureCredential. "
                        "Install with: pip install azure-identity"
                    )
            else:
                if not self.key:
                    raise ValueError(
                        "CosmosDB key not configured. "
                        "Set COSMOSDB_KEY environment variable or pass key parameter."
                    )
                self._client = CosmosClient(self.endpoint, credential=self.key)
            
            # Get or create database
            self._database = self._client.create_database_if_not_exists(id=self.database_name)
            
            # Get or create container with partition key
            try:
                self._container = self._database.create_container_if_not_exists(
                    id=self.container_name,
                    partition_key=PartitionKey(path="/env_name"),
                    offer_throughput=400
                )
            except CosmosResourceExistsError:
                self._container = self._database.get_container_client(self.container_name)
        
        return self._container
    
    async def save_trajectory(self, trajectory: TrajectoryData) -> str:
        """
        Save a trajectory to CosmosDB.
        
        Args:
            trajectory: The trajectory data to save
            
        Returns:
            The ID of the saved trajectory
        """
        container = self._get_client()
        
        # Generate ID and timestamp if not provided
        if not trajectory.id:
            trajectory.id = str(uuid.uuid4())
        if not trajectory.created_at:
            trajectory.created_at = datetime.utcnow().isoformat() + 'Z'
        
        # Convert to dict for storage
        data = trajectory.model_dump()
        
        # Upsert the document
        container.upsert_item(data)
        
        return trajectory.id
    
    def save_trajectory_sync(self, trajectory: TrajectoryData) -> str:
        """
        Synchronous version of save_trajectory.
        
        Args:
            trajectory: The trajectory data to save
            
        Returns:
            The ID of the saved trajectory
        """
        container = self._get_client()
        
        # Generate ID and timestamp if not provided
        if not trajectory.id:
            trajectory.id = str(uuid.uuid4())
        if not trajectory.created_at:
            trajectory.created_at = datetime.utcnow().isoformat() + 'Z'
        
        # Convert to dict for storage
        data = trajectory.model_dump()
        
        # Upsert the document
        container.upsert_item(data)
        
        return trajectory.id
    
    def get_trajectory(self, trajectory_id: str, env_name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a trajectory from CosmosDB.
        
        Args:
            trajectory_id: The ID of the trajectory
            env_name: The environment name (partition key)
            
        Returns:
            The trajectory data or None if not found
        """
        container = self._get_client()
        
        try:
            item = container.read_item(item=trajectory_id, partition_key=env_name)
            return item
        except Exception:
            return None
    
    def list_trajectories(
        self,
        env_name: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        List trajectories from CosmosDB.
        
        Args:
            env_name: Optional environment name to filter by
            limit: Maximum number of trajectories to return
            
        Returns:
            List of trajectory summaries
        """
        container = self._get_client()
        
        if env_name:
            query = f"SELECT c.id, c.session_id, c.env_name, c.created_at, c.is_done, c.reward FROM c WHERE c.env_name = @env_name ORDER BY c.created_at DESC OFFSET 0 LIMIT {limit}"
            parameters = [{"name": "@env_name", "value": env_name}]
        else:
            query = f"SELECT c.id, c.session_id, c.env_name, c.created_at, c.is_done, c.reward FROM c ORDER BY c.created_at DESC OFFSET 0 LIMIT {limit}"
            parameters = []
        
        items = list(container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True
        ))
        
        return items
    
    def delete_trajectory(self, trajectory_id: str, env_name: str) -> bool:
        """
        Delete a trajectory from CosmosDB.
        
        Args:
            trajectory_id: The ID of the trajectory
            env_name: The environment name (partition key)
            
        Returns:
            True if deleted, False if not found
        """
        container = self._get_client()
        
        try:
            container.delete_item(item=trajectory_id, partition_key=env_name)
            return True
        except Exception:
            return False


# Singleton instance for convenience
_storage_instance: Optional[CosmosDBStorage] = None


def get_cosmosdb_storage() -> CosmosDBStorage:
    """Get the singleton CosmosDB storage instance."""
    global _storage_instance
    if _storage_instance is None:
        _storage_instance = CosmosDBStorage()
    return _storage_instance


def is_cosmosdb_configured() -> bool:
    """Check if CosmosDB is configured via environment variables."""
    return bool(os.getenv("COSMOSDB_ENDPOINT"))
