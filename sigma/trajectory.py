# Copyright Amity
"""
Trajectory Module

This module provides the Trajectory class which represents a simulation trajectory.
A trajectory is the primary unit of work - it contains the conversation history,
environment state, and can be loaded/saved to storage.

The Trajectory class provides:
- Creation of new trajectories with scenario generation
- Loading existing trajectories from storage
- Adding messages to the conversation
- Saving/updating the trajectory
- Creating a simulator from the trajectory state

Usage:
    # Create a new trajectory
    trajectory = Trajectory.create(
        env_name="retail",
        user_model="gpt-4o",
        generate_scenario=True
    )
    
    # Load an existing trajectory
    trajectory = Trajectory.load(trajectory_id)
    
    # Get a simulator for the trajectory
    simulator = trajectory.get_simulator()
    
    # Add a message
    trajectory.add_message(role="agent", content="Hello!")
    
    # Save changes
    trajectory.save()
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from sigma.trajectory_storage import (
    TrajectoryData,
    TrajectoryMessage,
    RejectedSuggestion,
    get_trajectory_storage,
)
from sigma.env_registry import list_environments

if TYPE_CHECKING:
    from sigma.simulator_core import SimulatorCore


class TrajectoryError(Exception):
    """Custom exception for trajectory errors."""
    pass


class Trajectory:
    """
    Represents a simulation trajectory.
    
    A trajectory is the primary unit of work in the simulator. It contains:
    - Environment configuration
    - Persona and task information
    - Conversation history (messages)
    - Simulation state (is_done, reward, etc.)
    
    Trajectories can be created new (with optional scenario generation) or
    loaded from storage. They can be saved at any time to persist changes.
    """
    
    def __init__(self, data: TrajectoryData):
        """
        Initialize a Trajectory from TrajectoryData.
        
        Use Trajectory.create() or Trajectory.load() instead of calling this directly.
        """
        self._data = data
        self._simulator: Optional["SimulatorCore"] = None
        self._storage = get_trajectory_storage()
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def id(self) -> str:
        """Get the trajectory ID."""
        return self._data.id
    
    @property
    def env_name(self) -> str:
        """Get the environment name."""
        return self._data.env_name
    
    @property
    def messages(self) -> List[TrajectoryMessage]:
        """Get the conversation messages."""
        return self._data.messages
    
    @property
    def is_done(self) -> bool:
        """Check if the trajectory is complete."""
        return self._data.is_done
    
    @is_done.setter
    def is_done(self, value: bool):
        """Set the trajectory completion status."""
        self._data.is_done = value
    
    @property
    def reward(self) -> Optional[float]:
        """Get the final reward."""
        return self._data.reward
    
    @reward.setter
    def reward(self, value: Optional[float]):
        """Set the final reward."""
        self._data.reward = value
    
    @property
    def reward_info(self) -> Optional[Dict[str, Any]]:
        """Get the reward info."""
        return self._data.reward_info
    
    @reward_info.setter
    def reward_info(self, value: Optional[Dict[str, Any]]):
        """Set the reward info."""
        self._data.reward_info = value
    
    @property
    def persona(self) -> str:
        """Get the persona instruction."""
        return self._data.persona
    
    @property
    def wiki(self) -> str:
        """Get the wiki content."""
        return self._data.wiki
    
    @property
    def persona_data(self) -> Optional[Dict[str, Any]]:
        """Get the full persona data for environment state restoration."""
        return self._data.persona_data
    
    @property
    def task_instruction(self) -> Optional[str]:
        """Get the task instruction."""
        return self._data.task_instruction
    
    @property
    def user_model(self) -> str:
        """Get the user model."""
        return self._data.user_model
    
    @property
    def user_provider(self) -> str:
        """Get the user provider."""
        return self._data.user_provider
    
    @property
    def agent_model(self) -> Optional[str]:
        """Get the agent model."""
        return self._data.agent_model
    
    @property
    def agent_provider(self) -> Optional[str]:
        """Get the agent provider."""
        return self._data.agent_provider
    
    @property
    def expected_actions(self) -> Optional[List[Dict[str, Any]]]:
        """Get the expected actions for evaluation."""
        return self._data.expected_actions
    
    @expected_actions.setter
    def expected_actions(self, value: Optional[List[Dict[str, Any]]]):
        """Set the expected actions."""
        self._data.expected_actions = value
    
    @property
    def data(self) -> TrajectoryData:
        """Get the underlying TrajectoryData object."""
        return self._data
    
    # =========================================================================
    # Factory Methods
    # =========================================================================
    
    @classmethod
    def create(
        cls,
        env_name: str,
        user_model: str = "gpt-4o",
        user_provider: str = "openai",
        agent_model: Optional[str] = None,
        agent_provider: Optional[str] = None,
        persona: Optional[str] = None,
        task_index: Optional[int] = None,
        task_split: str = "test",
        generate_scenario: bool = True,
        task_ids: Optional[List[int]] = None,
    ) -> "Trajectory":
        """
        Create a new trajectory.
        
        This creates a new trajectory with optional scenario generation.
        The trajectory is automatically saved to storage.
        
        Args:
            env_name: Name of the environment (e.g., "retail", "airline")
            user_model: LLM model for user simulation
            user_provider: LLM provider for user simulation
            agent_model: LLM model for agent assistance (optional)
            agent_provider: LLM provider for agent assistance (optional)
            persona: Custom persona instruction
            task_index: Specific task index to use
            task_split: Task split ("test", "train", "dev")
            generate_scenario: If True, auto-generate a new scenario
            task_ids: Optional list of task IDs to sample from
            
        Returns:
            New Trajectory instance (already saved to storage)
        """
        from sigma.simulator_core import SimulatorCore
        
        # Create simulator which handles scenario generation
        simulator = SimulatorCore(
            env_name=env_name,
            user_model=user_model,
            user_provider=user_provider,
            agent_model=agent_model,
            agent_provider=agent_provider,
            persona=persona,
            task_index=task_index,
            task_split=task_split,
            generate_scenario=generate_scenario,
            task_ids=task_ids,
        )
        
        # Start simulation to get initial message
        initial_message = simulator.start()
        
        # Get task info
        task_instruction = None
        user_id = None
        if hasattr(simulator, 'env') and hasattr(simulator.env, 'task'):
            task_instruction = getattr(simulator.env.task, 'instruction', None)
            user_id = getattr(simulator.env.task, 'user_id', None)
        if not user_id and simulator.generated_scenario:
            user_id = getattr(simulator.generated_scenario, 'user_id', None)
        
        # Create initial message
        initial_messages = [
            TrajectoryMessage(
                id=str(uuid.uuid4()),
                role="user",
                content=initial_message,
                timestamp=datetime.utcnow().isoformat() + 'Z',
            )
        ]
        
        # Build trajectory data
        trajectory_id = simulator.session_id  # Use session_id as trajectory_id
        
        # Get seed_task_id from persona_data if available (for generated scenarios)
        seed_task_id = None
        if simulator.persona_data:
            seed_task_id = simulator.persona_data.get("seed_task_id")
        
        data = TrajectoryData(
            id=trajectory_id,
            session_id=trajectory_id,
            created_at=datetime.utcnow().isoformat() + 'Z',
            env_name=env_name,
            task_index=task_index,
            task_split=task_split,
            task_instruction=task_instruction,
            seed_task_id=seed_task_id,
            user_id=user_id,
            user_model=user_model,
            user_provider=user_provider,
            agent_model=agent_model,
            agent_provider=agent_provider,
            persona=simulator.current_persona or "",
            wiki=simulator.wiki or "",
            persona_data=simulator.persona_data,
            messages=initial_messages,
            is_done=False,
            expected_actions=simulator.state.expected_actions,
        )
        
        # Create trajectory instance
        trajectory = cls(data)
        trajectory._simulator = simulator
        
        # Save to storage
        trajectory.save()
        
        return trajectory
    
    @classmethod
    def load(cls, trajectory_id: str, env_name: Optional[str] = None) -> "Trajectory":
        """
        Load a trajectory from storage.
        
        Args:
            trajectory_id: The trajectory ID to load
            env_name: Optional environment name (will search all if not provided)
            
        Returns:
            Loaded Trajectory instance
            
        Raises:
            TrajectoryError: If trajectory not found
        """
        storage = get_trajectory_storage()
        
        # If env_name provided, try that first
        if env_name:
            data_dict = storage.get(trajectory_id, env_name)
            if data_dict:
                return cls._from_dict(data_dict)
        
        # Search all environments
        for env in list_environments():
            data_dict = storage.get(trajectory_id, env)
            if data_dict:
                return cls._from_dict(data_dict)
        
        raise TrajectoryError(f"Trajectory not found: {trajectory_id}")
    
    @classmethod
    def _from_dict(cls, data_dict: Dict[str, Any]) -> "Trajectory":
        """Create a Trajectory from a dictionary (loaded from storage)."""
        # Convert messages
        messages = []
        for msg in data_dict.get('messages', []):
            rejected = None
            if msg.get('rejected'):
                rejected = RejectedSuggestion(**msg['rejected'])
            messages.append(TrajectoryMessage(
                id=msg.get('id', ''),
                role=msg.get('role', 'user'),
                content=msg.get('content'),
                reasoning=msg.get('reasoning'),
                timestamp=msg.get('timestamp'),
                tool_name=msg.get('tool_name'),
                tool_arguments=msg.get('tool_arguments'),
                rejected=rejected,
            ))
        
        # Build TrajectoryData
        data = TrajectoryData(
            id=data_dict.get('id'),
            session_id=data_dict.get('session_id', data_dict.get('id')),
            created_at=data_dict.get('created_at'),
            env_name=data_dict.get('env_name'),
            task_index=data_dict.get('task_index'),
            task_split=data_dict.get('task_split', 'test'),
            task_instruction=data_dict.get('task_instruction'),
            seed_task_id=data_dict.get('seed_task_id'),
            user_id=data_dict.get('user_id'),
            user_model=data_dict.get('user_model', 'gpt-4o'),
            user_provider=data_dict.get('user_provider', 'openai'),
            agent_model=data_dict.get('agent_model'),
            agent_provider=data_dict.get('agent_provider'),
            persona=data_dict.get('persona', ''),
            wiki=data_dict.get('wiki', ''),
            persona_data=data_dict.get('persona_data'),
            messages=messages,
            is_done=data_dict.get('is_done', False),
            reward=data_dict.get('reward'),
            reward_info=data_dict.get('reward_info'),
            expected_actions=data_dict.get('expected_actions'),
        )
        
        return cls(data)
    
    # =========================================================================
    # Simulator Integration
    # =========================================================================
    
    def get_simulator(self) -> "SimulatorCore":
        """
        Get a simulator for this trajectory.
        
        If a simulator is already cached, returns it.
        Otherwise, creates a new simulator from the trajectory state.
        
        Returns:
            SimulatorCore instance ready for operations
        """
        if self._simulator is not None:
            return self._simulator
        
        from sigma.simulator_core import SimulatorCore
        
        # Create simulator from trajectory
        self._simulator = SimulatorCore.from_trajectory(self._data)
        
        return self._simulator
    
    def clear_simulator(self):
        """Clear the cached simulator (useful for memory management)."""
        self._simulator = None
    
    # =========================================================================
    # Message Management
    # =========================================================================
    
    def add_message(
        self,
        role: str,
        content: Optional[str] = None,
        reasoning: Optional[str] = None,
        tool_name: Optional[str] = None,
        tool_arguments: Optional[Dict[str, Any]] = None,
        rejected: Optional[Dict[str, Any]] = None,
    ) -> TrajectoryMessage:
        """
        Add a message to the trajectory.
        
        Args:
            role: Message role ('user', 'agent', 'tool', 'tool-result', 'rejected')
            content: Message content
            reasoning: Agent reasoning (for agent messages)
            tool_name: Tool name (for tool calls)
            tool_arguments: Tool arguments (for tool calls)
            rejected: Rejected suggestion data (for rejected messages)
            
        Returns:
            The created TrajectoryMessage
        """
        rejected_suggestion = None
        if rejected:
            rejected_suggestion = RejectedSuggestion(**rejected)
        
        message = TrajectoryMessage(
            id=str(uuid.uuid4()),
            role=role,
            content=content,
            reasoning=reasoning,
            timestamp=datetime.utcnow().isoformat() + 'Z',
            tool_name=tool_name,
            tool_arguments=tool_arguments,
            rejected=rejected_suggestion,
        )
        
        self._data.messages.append(message)
        return message
    
    def set_messages(self, messages: List[TrajectoryMessage]):
        """Replace all messages with a new list."""
        self._data.messages = messages
    
    def get_first_user_message(self) -> Optional[str]:
        """Get the first user message content."""
        for msg in self._data.messages:
            if msg.role == 'user':
                return msg.content
        return None
    
    # =========================================================================
    # Persistence
    # =========================================================================
    
    def save(self) -> str:
        """
        Save the trajectory to storage.
        
        Returns:
            The trajectory ID
        """
        # Sync state from simulator if available
        if self._simulator is not None:
            self._data.is_done = self._simulator.state.is_done
            self._data.reward = self._simulator.state.last_reward
            self._data.reward_info = self._simulator.state.reward_info
            self._data.expected_actions = self._simulator.state.expected_actions
        
        return self._storage.save(self._data)
    
    def delete(self) -> bool:
        """
        Delete the trajectory from storage.
        
        Returns:
            True if deleted successfully
        """
        return self._storage.delete(self._data.id, self._data.env_name)
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the trajectory to a dictionary."""
        return self._data.model_dump()
    
    def __repr__(self) -> str:
        return f"Trajectory(id={self.id}, env={self.env_name}, messages={len(self.messages)}, done={self.is_done})"
