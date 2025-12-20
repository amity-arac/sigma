# Copyright Amity
"""
Core Simulation Engine for tau-bench

This module provides the decoupled simulation logic that can be used by
different frontends (CLI, Web API, etc.).

The SimulatorCore class manages:
- Environment setup and state
- Conversation history
- Action execution
- LLM-based response generation
"""

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable

from litellm import completion

from sigma.envs.base import Env
from sigma.envs.user import UserStrategy


# Debug flag - set via environment variable
DEBUG_SIMULATOR = os.environ.get("DEBUG_SIMULATOR", "false").lower() == "true"


def _debug_log(message: str):
    """Print debug message with timestamp if DEBUG_SIMULATOR is enabled."""
    if DEBUG_SIMULATOR:
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] [SIMULATOR] {message}")
from sigma.types import Action, RESPOND_ACTION_NAME, EnvResponse

from sigma.env_registry import (
    get_environment_config,
    list_environments,
    get_all_environment_configs,
    EnvironmentConfig,
)
from sigma.scenario_generator import (
    ScenarioGenerator,
    GeneratedScenario,
    scenario_to_persona_data,
    log_scenario_to_console,
)


# Path to shared prompts directory
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


def load_base_prompt(filename: str) -> str:
    """Load a base prompt from the shared prompts directory."""
    prompt_path = os.path.join(PROMPTS_DIR, filename)
    if os.path.exists(prompt_path):
        with open(prompt_path, "r") as f:
            return f.read()
    raise FileNotFoundError(f"Base prompt not found: {prompt_path}")


def load_env_guidelines(env_name: str, filename: str) -> str:
    """Load environment-specific guidelines, or return empty string."""
    env_path = os.path.join(os.path.dirname(__file__), "envs", env_name)
    guidelines_path = os.path.join(env_path, filename)
    if os.path.exists(guidelines_path):
        with open(guidelines_path, "r") as f:
            return f.read()
    return ""


def build_agent_prompt(env_name: str, prompt_type: str) -> str:
    """
    Build agent prompt from base template.
    
    Args:
        env_name: Name of the environment (e.g., 'retail')
        prompt_type: Either 'response' or 'action'
    
    Note: Environment-specific policy is already included via {context} 
    which contains the wiki/policy from the environment.
    """
    base_filename = f"agent_{prompt_type}_base.md"
    return load_base_prompt(base_filename)


class ActionType(str, Enum):
    """Types of actions the agent can take."""
    RESPOND = "respond"
    TOOL_CALL = "tool_call"


@dataclass
class ConversationEntry:
    """A single entry in the conversation history."""
    role: str  # "user", "agent", "tool"
    content: Optional[str] = None
    tool_call: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None


@dataclass
class SimulationState:
    """Current state of a simulation session."""
    session_id: str
    env_name: str
    is_active: bool = True
    is_done: bool = False
    conversation_history: List[ConversationEntry] = field(default_factory=list)
    last_reward: Optional[float] = None
    reward_info: Optional[Dict[str, Any]] = None
    expected_actions: Optional[List[Dict[str, Any]]] = None


@dataclass
class ActionResult:
    """Result of executing an action."""
    success: bool
    observation: str
    done: bool
    reward: Optional[float] = None
    reward_info: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class ToolInfo:
    """Information about an available tool."""
    name: str
    description: str
    parameters: Dict[str, Any]
    required_params: List[str]


class SimulatorCore:
    """
    Core simulation engine that manages the interaction between
    a human agent and an LLM-simulated user.
    
    This class is UI-agnostic and can be used by CLI, Web API, or other frontends.
    """

    def __init__(
        self,
        env_name: str,
        user_model: str,
        user_provider: str,
        agent_model: Optional[str] = None,
        agent_provider: Optional[str] = None,
        persona: Optional[str] = None,
        persona_data: Optional[Dict[str, Any]] = None,
        task_index: Optional[int] = None,
        task_split: str = "test",
        session_id: Optional[str] = None,
        generate_scenario: bool = False,
        task_ids: Optional[List[int]] = None,
    ):
        """
        Initialize the simulator core.
        
        Args:
            env_name: Name of the environment (e.g., "retail", "airline")
            user_model: LLM model for user simulation
            user_provider: LLM provider for user simulation
            agent_model: LLM model for agent assistance (optional)
            agent_provider: LLM provider for agent assistance (optional)
            persona: Custom persona instruction for the user
            persona_data: Full persona data dict (from persona file)
            task_index: Specific task index to use
            task_split: Task split ("test", "train", "dev")
            session_id: Optional session ID (generated if not provided)
            generate_scenario: If True, auto-generate a new scenario inspired by existing tasks
            task_ids: Optional list of task IDs to sample from when generating scenarios
        """
        _debug_log(f"Creating session for env={env_name}, model={user_model}")
        init_start = time.time()
        
        self.env_name = env_name
        self.user_model = user_model
        self.user_provider = user_provider
        self.agent_model = agent_model or user_model
        self.agent_provider = agent_provider or user_provider
        self.persona = persona
        self.persona_data = persona_data
        self.task_index = task_index
        self.task_split = task_split
        self.session_id = session_id or str(uuid.uuid4())
        self.generated_scenario: Optional[GeneratedScenario] = None
        self.task_ids = task_ids  # Task IDs to filter for scenario generation
        
        # Auto-generate scenario if requested (and no persona_data provided)
        if generate_scenario and not persona_data:
            _debug_log("Generating new scenario...")
            scenario_start = time.time()
            self._generate_new_scenario()
            _debug_log(f"Scenario generation took {time.time() - scenario_start:.2f}s")
        
        # Build persona instruction if we have persona data but no direct persona
        if self.persona_data and not self.persona:
            self.persona = self._build_persona_instruction()
            _debug_log(f"Built persona instruction ({len(self.persona)} chars)")
        
        # Initialize environment
        _debug_log("Loading environment...")
        env_start = time.time()
        self.env = self._load_env()
        _debug_log(f"Environment loaded in {time.time() - env_start:.2f}s")
        
        # Inject persona data if available
        if self.persona_data:
            _debug_log("Injecting persona data...")
            self._inject_persona_data()
        
        # Cache tools info
        self._tools_info = self.env.tools_info
        
        # Initialize state
        self.state = SimulationState(
            session_id=self.session_id,
            env_name=env_name,
        )
        
        # Event callbacks for observers (optional)
        self._on_user_message: Optional[Callable[[str], None]] = None
        self._on_agent_message: Optional[Callable[[str], None]] = None
        self._on_tool_call: Optional[Callable[[str, Dict], None]] = None
        self._on_tool_result: Optional[Callable[[str], None]] = None
        self._on_simulation_end: Optional[Callable[[Dict], None]] = None
        
        _debug_log(f"Session {self.session_id[:8]}... initialized in {time.time() - init_start:.2f}s")

    def _generate_new_scenario(self) -> None:
        """Generate a new scenario using the ScenarioGenerator."""
        # Use the ScenarioGenerator's default model (gpt-5.2), not the user_model
        # The user_model is for simulating user conversations, not for generating scenarios
        generator = ScenarioGenerator(
            env_name=self.env_name,
            task_ids=self.task_ids,  # Pass task IDs for focused scenario generation
            # model and provider use defaults from ScenarioGenerator
        )
        
        # Generate the scenario
        self.generated_scenario = generator.generate()
        
        # Log the scenario to console for the human agent to see
        log_scenario_to_console(self.generated_scenario)
        
        # Convert to persona_data format
        self.persona_data = scenario_to_persona_data(self.generated_scenario)

    def _build_persona_instruction(self) -> str:
        """Build a detailed persona instruction from persona data."""
        if not self.persona_data:
            return ""
        
        instruction = self.persona_data.get("instruction", "")
        user_data = self.persona_data.get("user", {})
        
        # Add user profile context to instruction
        user_context_parts = ["\n\nYOUR PROFILE INFORMATION (use this when asked):"]
        
        name = user_data.get("name", {})
        if name:
            user_context_parts.append(f"- Full Name: {name.get('first_name', '')} {name.get('last_name', '')}")
        
        if user_data.get("email"):
            user_context_parts.append(f"- Email: {user_data['email']}")
        
        address = user_data.get("address", {})
        if address:
            addr_str = f"{address.get('address1', '')}"
            if address.get('address2'):
                addr_str += f", {address['address2']}"
            addr_str += f", {address.get('city', '')}, {address.get('state', '')} {address.get('zip', '')}"
            user_context_parts.append(f"- Address: {addr_str}")
            user_context_parts.append(f"- Zip Code: {address.get('zip', '')}")
        
        if user_data.get("dob"):
            user_context_parts.append(f"- Date of Birth: {user_data['dob']}")
        
        if user_data.get("membership"):
            user_context_parts.append(f"- Membership: {user_data['membership']}")
        
        # Get the data key from registry
        try:
            env_config = get_environment_config(self.env_name)
            data_key = env_config.data_key
        except ValueError:
            data_key = "orders" if self.env_name == "retail" else "reservations"
        
        # Add data items info
        items = self.persona_data.get(data_key, {})
        if items:
            user_context_parts.append(f"- Your {data_key.title()}: {', '.join(items.keys())}")
            for item_id, item in items.items():
                summary_parts = []
                if "status" in item:
                    summary_parts.append(item.get("status"))
                if "origin" in item and "destination" in item:
                    summary_parts.append(f"{item.get('origin')} → {item.get('destination')}")
                if "cabin" in item:
                    summary_parts.append(f"({item.get('cabin')})")
                if "items" in item:
                    item_names = [i.get("name", "item") for i in item.get("items", [])]
                    summary_parts.append(f"Items: {', '.join(item_names)}")
                
                summary = " - ".join(summary_parts) if summary_parts else ""
                user_context_parts.append(f"  - {item_id}: {summary}")
        
        # Add payment methods
        payment_methods = user_data.get("payment_methods", {})
        if payment_methods:
            pm_strs = []
            for pm_id, pm in payment_methods.items():
                source = pm.get("source", "unknown")
                if source == "credit_card":
                    pm_strs.append(f"Credit card ending in {pm.get('last_four', '****')}")
                elif source == "gift_card":
                    pm_strs.append(f"Gift card (balance: ${pm.get('balance', 0)})")
                elif source == "certificate":
                    pm_strs.append(f"Certificate (${pm.get('amount', 0)})")
                elif source == "paypal":
                    pm_strs.append("PayPal")
            if pm_strs:
                user_context_parts.append(f"- Payment Methods: {', '.join(pm_strs)}")
        
        return instruction + "\n".join(user_context_parts)

    def _inject_persona_data(self):
        """
        Inject persona data into the environment's data store.
        
        This includes:
        1. User data (user profile)
        2. Primary data items (orders/reservations)
        3. Augmented data (any additional data like products, flights, etc.)
        """
        if not self.persona_data:
            return
        
        user_data = self.persona_data.get("user", {})
        user_id = user_data.get("user_id")
        
        if not user_id:
            return
        
        # Inject user into environment data
        self.env.data["users"][user_id] = user_data
        _debug_log(f"Injected user: {user_id}")
        
        # Get the data key from registry
        try:
            env_config = get_environment_config(self.env_name)
            data_key = env_config.data_key
        except ValueError:
            data_key = "orders" if self.env_name == "retail" else "reservations"
        
        # Inject primary data items (orders/reservations)
        items = self.persona_data.get(data_key, {})
        if items and data_key in self.env.data:
            for item_id, item in items.items():
                self.env.data[data_key][item_id] = item
                _debug_log(f"Injected {data_key} item: {item_id}")
        
        # Inject augmented data (products, flights, etc.)
        augmented_data = self.persona_data.get("augmented_data", {})
        if augmented_data:
            for collection_name, collection_items in augmented_data.items():
                if collection_name in self.env.data and isinstance(collection_items, dict):
                    for item_id, item in collection_items.items():
                        self.env.data[collection_name][item_id] = item
                        _debug_log(f"Injected augmented {collection_name} item: {item_id}")

    def _load_env(self) -> Env:
        """Load the environment based on env_name using the registry."""
        try:
            env_config = get_environment_config(self.env_name)
            env_class = env_config.env_class_loader()
            
            return env_class(
                user_strategy=UserStrategy.LLM,
                user_model=self.user_model,
                user_provider=self.user_provider,
                task_split=self.task_split,
                task_index=self.task_index,
            )
        except ValueError as e:
            raise ValueError(f"Failed to load environment '{self.env_name}': {e}")

    # =========================================================================
    # Public Properties
    # =========================================================================

    @property
    def tools(self) -> List[ToolInfo]:
        """Get list of available tools with their info."""
        tools = []
        for tool in self._tools_info:
            func_info = tool.get("function", {})
            params = func_info.get("parameters", {})
            tools.append(ToolInfo(
                name=func_info.get("name", "Unknown"),
                description=func_info.get("description", ""),
                parameters=params.get("properties", {}),
                required_params=params.get("required", []),
            ))
        return tools

    @property
    def tools_raw(self) -> List[Dict[str, Any]]:
        """Get raw tools info (OpenAI function format)."""
        return self._tools_info

    @property
    def wiki(self) -> str:
        """Get the environment wiki/policy."""
        return self.env.wiki

    @property
    def current_persona(self) -> str:
        """Get the current user persona/instruction."""
        if self.persona:
            return self.persona
        return self.env.task.instruction

    @property
    def conversation_history(self) -> List[ConversationEntry]:
        """Get conversation history."""
        return self.state.conversation_history

    @property
    def is_done(self) -> bool:
        """Check if simulation is complete."""
        return self.state.is_done

    # =========================================================================
    # Event Registration
    # =========================================================================

    def on_user_message(self, callback: Callable[[str], None]):
        """Register callback for user messages."""
        self._on_user_message = callback

    def on_agent_message(self, callback: Callable[[str], None]):
        """Register callback for agent messages."""
        self._on_agent_message = callback

    def on_tool_call(self, callback: Callable[[str, Dict], None]):
        """Register callback for tool calls."""
        self._on_tool_call = callback

    def on_tool_result(self, callback: Callable[[str], None]):
        """Register callback for tool results."""
        self._on_tool_result = callback

    def on_simulation_end(self, callback: Callable[[Dict], None]):
        """Register callback for simulation end."""
        self._on_simulation_end = callback

    # =========================================================================
    # Simulation Control
    # =========================================================================

    def start(self) -> str:
        """
        Start the simulation and return the initial user message.
        
        Returns:
            The initial message from the simulated user.
        """
        _debug_log("Starting simulation...")
        start_time = time.time()
        
        if self.persona:
            # Override the user's instruction
            _debug_log(f"Using persona instruction ({len(self.persona)} chars)")
            _debug_log(f"Persona preview: {self.persona[:200]}...")
            self.env.user.reset(instruction=self.persona)
            _debug_log("Generating initial user message...")
            msg_start = time.time()
            initial_message = self.env.user.generate_next_message(self.env.user.messages)
            _debug_log(f"Initial message generated in {time.time() - msg_start:.2f}s")
        else:
            _debug_log(f"Using task index: {self.task_index}")
            reset_response = self.env.reset(task_index=self.task_index)
            initial_message = reset_response.observation
        
        # Record initial message
        self._add_conversation_entry("user", content=initial_message)
        
        if self._on_user_message:
            self._on_user_message(initial_message)
        
        _debug_log(f"Simulation started in {time.time() - start_time:.2f}s")
        _debug_log(f"Initial message: {initial_message[:100]}...")
        
        return initial_message

    def respond_to_user(self, message: str) -> ActionResult:
        """
        Send a text response to the user.
        
        Args:
            message: The response message to send.
            
        Returns:
            ActionResult with the user's next message or completion info.
        """
        if self.state.is_done:
            return ActionResult(
                success=False,
                observation="",
                done=True,
                error="Simulation is already complete"
            )
        
        # Record agent message
        self._add_conversation_entry("agent", content=message)
        
        if self._on_agent_message:
            self._on_agent_message(message)
        
        # Execute the response action
        action = Action(name=RESPOND_ACTION_NAME, kwargs={"content": message})
        response = self.env.step(action)
        
        return self._process_env_response(response)

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> ActionResult:
        """
        Call a tool with the given arguments.
        
        Args:
            tool_name: Name of the tool to call.
            arguments: Arguments for the tool.
            
        Returns:
            ActionResult with the tool execution result.
        """
        if self.state.is_done:
            return ActionResult(
                success=False,
                observation="",
                done=True,
                error="Simulation is already complete"
            )
        
        # Record tool call
        self._add_conversation_entry("agent", tool_call={"name": tool_name, "arguments": arguments})
        
        if self._on_tool_call:
            self._on_tool_call(tool_name, arguments)
        
        # Execute the tool
        action = Action(name=tool_name, kwargs=arguments)
        response = self.env.step(action)
        
        # Record tool result
        self._add_conversation_entry("tool", content=response.observation)
        
        if self._on_tool_result:
            self._on_tool_result(response.observation)
        
        return self._process_env_response(response, is_tool_call=True)

    def _process_env_response(self, response: EnvResponse, is_tool_call: bool = False) -> ActionResult:
        """Process environment response and update state."""
        if response.done:
            self.state.is_done = True
            self.state.last_reward = response.reward
            
            # Extract reward info
            if response.info.reward_info:
                self.state.reward_info = {
                    "reward": response.info.reward_info.reward,
                    "actions": [{"name": a.name, "kwargs": a.kwargs} for a in response.info.reward_info.actions],
                }
                if hasattr(response.info.reward_info.info, "r_actions"):
                    self.state.reward_info["r_actions"] = response.info.reward_info.info.r_actions
                if hasattr(response.info.reward_info.info, "r_outputs"):
                    self.state.reward_info["r_outputs"] = response.info.reward_info.info.r_outputs
                    if hasattr(response.info.reward_info.info, "outputs"):
                        self.state.reward_info["outputs"] = response.info.reward_info.info.outputs
            
            # Store expected actions
            self.state.expected_actions = [
                {"name": a.name, "kwargs": a.kwargs}
                for a in self.env.task.actions
            ]
            
            if self._on_simulation_end:
                self._on_simulation_end(self.get_final_result())
        
        elif not is_tool_call:
            # Record user's next message
            self._add_conversation_entry("user", content=response.observation)
            
            if self._on_user_message:
                self._on_user_message(response.observation)
        
        return ActionResult(
            success=True,
            observation=response.observation,
            done=response.done,
            reward=response.reward if response.done else None,
            reward_info=self.state.reward_info if response.done else None,
        )

    def _add_conversation_entry(
        self,
        role: str,
        content: Optional[str] = None,
        tool_call: Optional[Dict[str, Any]] = None,
    ):
        """Add an entry to conversation history."""
        entry = ConversationEntry(
            role=role,
            content=content,
            tool_call=tool_call,
        )
        self.state.conversation_history.append(entry)

    def undo_last_action(self) -> Dict[str, Any]:
        """
        Undo the last action by removing entries from conversation history.
        
        This removes entries backwards until we hit an 'agent' entry (the action),
        including any subsequent 'tool' or 'user' responses.
        
        Returns:
            Dict with removed_count and the removed entries.
        """
        if len(self.state.conversation_history) <= 1:
            return {
                "success": False,
                "error": "Cannot undo: only initial message remains",
                "removed_count": 0,
                "removed_entries": []
            }
        
        removed_entries = []
        
        # Remove entries from the end until we've removed an agent action
        # Pattern: we want to remove the last "action block" which could be:
        # - agent response -> user reply
        # - agent tool_call -> tool result
        # - agent tool_call -> tool result -> user reply (if tool triggered user response)
        
        found_agent_action = False
        while self.state.conversation_history and len(self.state.conversation_history) > 1:
            last_entry = self.state.conversation_history[-1]
            
            # If we already found and removed an agent action, stop when we hit another agent action
            if found_agent_action and last_entry.role == "agent":
                break
            
            # Remove the entry
            removed = self.state.conversation_history.pop()
            removed_entries.append({
                "role": removed.role,
                "content": removed.content[:100] if removed.content else None,
                "tool_call": removed.tool_call
            })
            
            if last_entry.role == "agent":
                found_agent_action = True
                break  # Stop after removing the agent action
        
        # Reset done state if we undid the final action
        if self.state.is_done and removed_entries:
            self.state.is_done = False
            self.state.last_reward = None
            self.state.reward_info = {}
        
        return {
            "success": True,
            "removed_count": len(removed_entries),
            "removed_entries": list(reversed(removed_entries)),  # Return in original order
            "remaining_count": len(self.state.conversation_history)
        }

    def rollback_to_index(self, target_index: int) -> Dict[str, Any]:
        """
        Rollback conversation history to a specific index.
        
        This removes all entries from target_index onwards (exclusive).
        
        Args:
            target_index: The index to rollback to. All entries at and after this index
                         will be removed.
        
        Returns:
            Dict with success status, removed_count and remaining_count.
        """
        if target_index < 1:
            return {
                "success": False,
                "error": "Cannot rollback: must keep at least the initial message",
                "removed_count": 0,
                "remaining_count": len(self.state.conversation_history)
            }
        
        if target_index >= len(self.state.conversation_history):
            return {
                "success": False,
                "error": "Invalid target index: nothing to remove",
                "removed_count": 0,
                "remaining_count": len(self.state.conversation_history)
            }
        
        # Remove entries from target_index onwards
        removed_entries = []
        while len(self.state.conversation_history) > target_index:
            removed = self.state.conversation_history.pop()
            removed_entries.append({
                "role": removed.role,
                "content": removed.content[:100] if removed.content else None,
                "tool_call": removed.tool_call
            })
        
        # Reset done state if we undid the final action
        if self.state.is_done and removed_entries:
            self.state.is_done = False
            self.state.last_reward = None
            self.state.reward_info = {}
        
        return {
            "success": True,
            "removed_count": len(removed_entries),
            "removed_entries": list(reversed(removed_entries)),
            "remaining_count": len(self.state.conversation_history)
        }

    def regenerate_user_response(self, additional_note: Optional[str] = None) -> Dict[str, Any]:
        """
        Regenerate the simulated user's response.
        
        This finds the last user message, removes it and all subsequent messages
        (rolling back to that state), then generates a new user response.
        
        Args:
            additional_note: Optional additional guidance for the user agent.
        
        Returns:
            Dict with success status and new observation (user response).
        """
        if len(self.state.conversation_history) < 2:
            return {
                "success": False,
                "error": "No user message to regenerate",
                "observation": None
            }
        
        # Find the last user message in the conversation
        user_message_index = None
        for i in range(len(self.state.conversation_history) - 1, -1, -1):
            if self.state.conversation_history[i].role == "user":
                user_message_index = i
                break
        
        if user_message_index is None:
            return {
                "success": False,
                "error": "No user message found in conversation",
                "observation": None
            }
        
        # Find the agent message before this user message
        agent_message_before = None
        for i in range(user_message_index - 1, -1, -1):
            if self.state.conversation_history[i].role == "agent":
                agent_message_before = self.state.conversation_history[i].content
                break
        
        if agent_message_before is None:
            return {
                "success": False,
                "error": "No agent message found before user message",
                "observation": None
            }
        
        # Rollback: Remove the user message and all subsequent messages
        removed_count = len(self.state.conversation_history) - user_message_index
        while len(self.state.conversation_history) > user_message_index:
            self.state.conversation_history.pop()
        
        # Reset done state if we removed messages
        if self.state.is_done:
            self.state.is_done = False
            self.state.last_reward = None
            self.state.reward_info = {}
        
        # Store additional note temporarily
        original_persona_note = getattr(self, '_additional_persona_note', None)
        if additional_note:
            self._additional_persona_note = additional_note
        
        try:
            # Regenerate user response to the agent message
            new_response = self._generate_user_response_to_message(
                agent_message_before,
                additional_note
            )
            if new_response:
                self._add_conversation_entry("user", new_response)
                return {
                    "success": True,
                    "observation": new_response,
                    "done": False,
                    "removed_count": removed_count
                }
            
            return {
                "success": False,
                "error": "Could not regenerate user response",
                "observation": None
            }
        finally:
            # Restore original note
            self._additional_persona_note = original_persona_note

    def _generate_user_response_to_message(
        self, 
        agent_message: str,
        additional_note: Optional[str] = None
    ) -> Optional[str]:
        """Generate a user response to an agent message."""
        # Build context for user response generation
        # Note: self.current_persona already contains the full persona including
        # the user's task/instruction, so we don't need to add it separately
        user_prompt_parts = [
            f"You are playing the role of a customer with the following persona:",
            f"{self.current_persona}",
            "",
            "The customer service agent just said:",
            f'"{agent_message}"',
            "",
            "Generate a natural response as this customer."
        ]
        
        if additional_note:
            user_prompt_parts.extend([
                "",
                f"IMPORTANT GUIDANCE: {additional_note}"
            ])
        
        user_prompt = "\n".join(user_prompt_parts)
        
        try:
            # Build completion kwargs
            completion_kwargs = {
                "model": self.user_model,
                "custom_llm_provider": self.user_provider,
                "messages": [{"role": "user", "content": user_prompt}],
            }
            if not self.user_model.startswith("gpt-5"):
                completion_kwargs["temperature"] = 0.7
            
            if DEBUG_SIMULATOR:
                print(f"\n{'='*80}")
                print(f"[DEBUG SIMULATOR] _generate_user_response - Sending to {self.user_model} via {self.user_provider}:")
                print(f"{'='*80}")
                for i, msg in enumerate(completion_kwargs["messages"]):
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')
                    print(f"\n--- Message {i+1} [{role.upper()}] ---")
                    print(content)  # Full content, no truncation
                print(f"{'='*80}\n")
            
            res = completion(**completion_kwargs)
            response = res.choices[0].message.content.strip()
            
            if DEBUG_SIMULATOR:
                print(f"\n{'='*80}")
                print(f"[DEBUG SIMULATOR] _generate_user_response - Response:")
                print(f"{'='*80}")
                print(response)  # Full response, no truncation
                print(f"{'='*80}\n")
            
            return response
        except Exception as e:
            return None

    # =========================================================================
    # Agent Assistance
    # =========================================================================

    def generate_response(self, prompt: str) -> Optional[str]:
        """
        Generate an agent response using LLM based on the prompt.
        
        Args:
            prompt: Instruction or description of what to say.
            
        Returns:
            Generated response text, or None if generation failed.
        """
        context = self._build_agent_context()
        
        # Load prompt template (base + env-specific guidelines)
        template = build_agent_prompt(self.env_name, "response")
        system_prompt = template.replace("{context}", context).replace("{prompt}", prompt)

        try:
            # Build completion kwargs - some models don't support temperature
            completion_kwargs = {
                "model": self.agent_model,
                "custom_llm_provider": self.agent_provider,
                "messages": [{"role": "user", "content": system_prompt}],
            }
            # Only add temperature for models that support it (not gpt-5 series)
            if not self.agent_model.startswith("gpt-5"):
                completion_kwargs["temperature"] = 0.7
            
            if DEBUG_SIMULATOR:
                print(f"\n{'='*80}")
                print(f"[DEBUG SIMULATOR] generate_response - Sending to {self.agent_model} via {self.agent_provider}:")
                print(f"{'='*80}")
                for i, msg in enumerate(completion_kwargs["messages"]):
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')
                    print(f"\n--- Message {i+1} [{role.upper()}] ---")
                    print(content)  # Full content, no truncation
                print(f"{'='*80}\n")
            
            res = completion(**completion_kwargs)
            response = res.choices[0].message.content.strip()
            
            if DEBUG_SIMULATOR:
                print(f"\n{'='*80}")
                print(f"[DEBUG SIMULATOR] generate_response - Response:")
                print(f"{'='*80}")
                print(response)  # Full response, no truncation
                print(f"{'='*80}\n")
            
            return response
        except Exception as e:
            return None

    def parse_natural_language_action(self, user_input: str) -> Optional[Dict[str, Any]]:
        """
        Parse natural language input into a structured action with reasoning.
        
        Args:
            user_input: Natural language description of the action.
            
        Returns:
            Parsed action dict with reasoning, action_type and relevant fields.
        """
        tools_summary = self._get_tools_summary()
        context = self._build_agent_context()
        
        # Load prompt template (base + env-specific guidelines)
        template = build_agent_prompt(self.env_name, "action")
        system_prompt = (template
            .replace("{tools_summary}", tools_summary)
            .replace("{context}", context)
            .replace("{user_input}", user_input))

        try:
            # Build completion kwargs - some models don't support temperature
            completion_kwargs = {
                "model": self.agent_model,
                "custom_llm_provider": self.agent_provider,
                "messages": [{"role": "user", "content": system_prompt}],
            }
            # Only add temperature for models that support it (not gpt-5 series)
            if not self.agent_model.startswith("gpt-5"):
                completion_kwargs["temperature"] = 0.3
            
            if DEBUG_SIMULATOR:
                print(f"\n{'='*80}")
                print(f"[DEBUG SIMULATOR] parse_natural_language_action - Sending to {self.agent_model} via {self.agent_provider}:")
                print(f"{'='*80}")
                for i, msg in enumerate(completion_kwargs["messages"]):
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')
                    print(f"\n--- Message {i+1} [{role.upper()}] ---")
                    print(content)  # Full content, no truncation
                print(f"{'='*80}\n")
            
            res = completion(**completion_kwargs)
            
            response_text = res.choices[0].message.content.strip()
            
            if DEBUG_SIMULATOR:
                print(f"\n{'='*80}")
                print(f"[DEBUG SIMULATOR] parse_natural_language_action - Response:")
                print(f"{'='*80}")
                print(response_text)  # Full response, no truncation
                print(f"{'='*80}\n")
            
            # Extract JSON from response
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end]
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end]
            
            return json.loads(response_text.strip())
            
        except json.JSONDecodeError as e:
            print(f"[parse_natural_language_action] JSON decode error: {e}")
            print(f"[parse_natural_language_action] Response text: {response_text[:500] if 'response_text' in dir() else 'N/A'}")
            return None
        except Exception as e:
            print(f"[parse_natural_language_action] Error: {type(e).__name__}: {e}")
            return None

    def _build_agent_context(self) -> str:
        """Build context string for agent LLM assistance."""
        context_parts = []
        
        # Add wiki/policy
        context_parts.append(f"# Agent Policy\n{self.env.wiki}\n")
        
        # Add conversation history
        context_parts.append("# Conversation History")
        for entry in self.state.conversation_history:
            if entry.role == "user":
                context_parts.append(f"CUSTOMER: {entry.content or ''}")
            elif entry.role == "agent":
                if entry.tool_call:
                    tc = entry.tool_call
                    context_parts.append(f"AGENT ACTION: Called tool '{tc['name']}' with arguments: {json.dumps(tc['arguments'], indent=2)}")
                else:
                    context_parts.append(f"AGENT RESPONSE: {entry.content or ''}")
            elif entry.role == "tool":
                result = entry.content or ""
                if len(result) > 1500:
                    result = result[:1500] + "...[truncated]"
                context_parts.append(f"TOOL RESULT: {result}")
        
        return "\n".join(context_parts)

    def _get_tools_summary(self) -> str:
        """Get a concise summary of available tools for LLM context."""
        tools_summary = []
        for tool in self._tools_info:
            func_info = tool.get("function", {})
            name = func_info.get("name", "Unknown")
            desc = func_info.get("description", "No description")
            params = func_info.get("parameters", {}).get("properties", {})
            required = func_info.get("parameters", {}).get("required", [])
            
            param_strs = []
            for p_name, p_info in params.items():
                p_type = p_info.get("type", "string")
                is_req = "required" if p_name in required else "optional"
                enum_vals = p_info.get("enum")
                if enum_vals:
                    param_strs.append(f"{p_name}: {p_type} [{is_req}] (options: {', '.join(enum_vals)})")
                else:
                    param_strs.append(f"{p_name}: {p_type} [{is_req}]")
            
            tools_summary.append(f"- {name}: {desc}\n  Parameters: {', '.join(param_strs) if param_strs else 'none'}")
        
        return "\n".join(tools_summary)

    # =========================================================================
    # Results and Export
    # =========================================================================

    def get_final_result(self) -> Dict[str, Any]:
        """Get the final simulation result."""
        return {
            "session_id": self.session_id,
            "env_name": self.env_name,
            "is_done": self.state.is_done,
            "reward": self.state.last_reward,
            "reward_info": self.state.reward_info,
            "expected_actions": self.state.expected_actions,
            "conversation_history": [
                {
                    "role": e.role,
                    "content": e.content,
                    "tool_call": e.tool_call,
                }
                for e in self.state.conversation_history
            ],
        }

    def export_trajectory(self) -> List[Dict[str, Any]]:
        """Export the conversation as a trajectory for training data."""
        trajectory = []
        for entry in self.state.conversation_history:
            if entry.role == "user":
                trajectory.append({"role": "user", "content": entry.content})
            elif entry.role == "agent":
                if entry.tool_call:
                    trajectory.append({
                        "role": "assistant",
                        "tool_calls": [{
                            "type": "function",
                            "function": entry.tool_call,
                        }],
                    })
                else:
                    trajectory.append({"role": "assistant", "content": entry.content})
            elif entry.role == "tool":
                trajectory.append({"role": "tool", "content": entry.content})
        return trajectory


# =============================================================================
# Session Manager for Multi-User Support
# =============================================================================

class SimulatorSessionManager:
    """
    Manages multiple simulator sessions for multi-user support.
    Useful for web server deployments.
    """

    def __init__(self):
        self._sessions: Dict[str, SimulatorCore] = {}

    def create_session(
        self,
        env_name: str,
        user_model: str,
        user_provider: str,
        agent_model: Optional[str] = None,
        agent_provider: Optional[str] = None,
        persona: Optional[str] = None,
        persona_data: Optional[Dict[str, Any]] = None,
        task_index: Optional[int] = None,
        task_split: str = "test",
        generate_scenario: bool = False,
        task_ids: Optional[List[int]] = None,
    ) -> SimulatorCore:
        """
        Create a new simulation session.
        
        Args:
            env_name: Environment name
            user_model: LLM model for user simulation
            user_provider: LLM provider
            agent_model: Optional agent model
            agent_provider: Optional agent provider
            persona: Custom persona instruction
            persona_data: Full persona data dict
            task_index: Specific task to use
            task_split: Task split (test/train/dev)
            generate_scenario: If True, auto-generate a new scenario
            task_ids: Optional list of task IDs to sample from when generating scenarios
        """
        simulator = SimulatorCore(
            env_name=env_name,
            user_model=user_model,
            user_provider=user_provider,
            agent_model=agent_model,
            agent_provider=agent_provider,
            persona=persona,
            persona_data=persona_data,
            task_index=task_index,
            task_split=task_split,
            generate_scenario=generate_scenario,
            task_ids=task_ids,
        )
        self._sessions[simulator.session_id] = simulator
        return simulator

    def get_session(self, session_id: str) -> Optional[SimulatorCore]:
        """Get an existing session by ID."""
        return self._sessions.get(session_id)

    def remove_session(self, session_id: str) -> bool:
        """Remove a session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def list_sessions(self) -> List[str]:
        """List all active session IDs."""
        return list(self._sessions.keys())

    def cleanup_completed_sessions(self) -> int:
        """Remove all completed sessions. Returns count of removed sessions."""
        to_remove = [
            sid for sid, sim in self._sessions.items()
            if sim.is_done
        ]
        for sid in to_remove:
            del self._sessions[sid]
        return len(to_remove)


# =============================================================================
# Helper Functions
# =============================================================================

def load_persona_file(filepath: str) -> Dict[str, Any]:
    """Load a persona from a JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def get_available_environments() -> List[Dict[str, str]]:
    """Get list of available environments with their info."""
    configs = get_all_environment_configs()
    return [
        {
            "name": name,
            "display_name": cfg.display_name,
            "description": cfg.description,
        }
        for name, cfg in configs.items()
    ]
