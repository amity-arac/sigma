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
from typing import Any, Dict, List, Optional, Callable, TYPE_CHECKING

from litellm import completion

from sigma.envs.base import Env
from sigma.envs.user import UserStrategy

# Type hints for TrajectoryData without circular imports
if TYPE_CHECKING:
    from sigma.trajectory_storage import TrajectoryData, TrajectoryMessage

# Debug flag - set via environment variable
DEBUG_SIMULATOR = os.environ.get("DEBUG_SIMULATOR", "false").lower() == "true"

# Agent generation logging - logs all agent-side LLM calls (parse action, generate response)
LOG_AGENT_GENERATION = os.environ.get("LOG_AGENT_GENERATION", "true").lower() == "true"


def _debug_log(message: str):
    """Print debug message with timestamp if DEBUG_SIMULATOR is enabled."""
    if DEBUG_SIMULATOR:
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] [SIMULATOR] {message}")


def _log_agent_generation(operation: str, model: str, provider: str, prompt_preview: str = None, response: str = None):
    """Log agent-side LLM generation calls."""
    if not LOG_AGENT_GENERATION:
        return
    
    timestamp = time.strftime("%H:%M:%S")
    print(f"\n{'='*60}")
    print(f"[{timestamp}] 🤖 AGENT GENERATION: {operation}")
    print(f"{'='*60}")
    print(f"Model: {model} ({provider})")
    
    if prompt_preview:
        # Show a preview of the prompt (last part is most relevant)
        preview_lines = prompt_preview.strip().split('\n')
        if len(preview_lines) > 10:
            print(f"Prompt (last 10 lines of {len(preview_lines)} total):")
            for line in preview_lines[-10:]:
                print(f"  {line[:100]}{'...' if len(line) > 100 else ''}")
        else:
            print(f"Prompt:")
            for line in preview_lines:
                print(f"  {line[:100]}{'...' if len(line) > 100 else ''}")
    
    if response:
        print(f"\nResponse:")
        response_lines = response.strip().split('\n')
        for line in response_lines[:20]:  # Show first 20 lines
            print(f"  {line}")
        if len(response_lines) > 20:
            print(f"  ... ({len(response_lines) - 20} more lines)")
    
    print(f"{'='*60}\n")


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
    from sigma.envs.paths import DATA_ENVS_PATH
    env_path = os.path.join(DATA_ENVS_PATH, env_name)
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
    id: Optional[str] = None  # Unique message ID for idempotent operations


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

    @classmethod
    def from_trajectory(
        cls,
        trajectory: "TrajectoryData",
        user_model: Optional[str] = None,
        user_provider: Optional[str] = None,
        agent_model: Optional[str] = None,
        agent_provider: Optional[str] = None,
    ) -> "SimulatorCore":
        """
        Create a SimulatorCore instance from saved trajectory data.
        
        This is the primary method for resuming simulations from saved state.
        It properly restores:
        - Environment configuration
        - Persona data (user profile, orders, augmented data)
        - Conversation history
        - Done state and rewards
        
        Args:
            trajectory: TrajectoryData object (from storage or constructed)
            user_model: Override user model (uses trajectory's model if not provided)
            user_provider: Override user provider
            agent_model: Override agent model
            agent_provider: Override agent provider
            
        Returns:
            Fully initialized SimulatorCore ready for continued simulation
        """
        # Import here to avoid circular imports
        from sigma.trajectory_storage import TrajectoryData
        
        _debug_log(f"Creating simulator from trajectory {trajectory.session_id[:8]}...")
        
        # Use trajectory's models or provided overrides
        effective_user_model = user_model or trajectory.user_model
        effective_user_provider = user_provider or trajectory.user_provider
        effective_agent_model = agent_model or trajectory.agent_model
        effective_agent_provider = agent_provider or trajectory.agent_provider
        
        # Create the simulator with full persona_data from trajectory
        # Use the trajectory ID as the session ID for continuity
        simulator = cls(
            env_name=trajectory.env_name,
            user_model=effective_user_model,
            user_provider=effective_user_provider,
            agent_model=effective_agent_model,
            agent_provider=effective_agent_provider,
            persona=trajectory.persona or None,
            persona_data=trajectory.persona_data,  # Full persona data including augmented_data
            task_index=trajectory.task_index,
            task_split=trajectory.task_split or "test",
            session_id=trajectory.id,  # Use trajectory ID as session ID for continuity
            generate_scenario=False,  # Don't generate new scenario when resuming
        )
        
        # Store the original trajectory ID for reference
        simulator._source_trajectory_id = trajectory.id
        
        # Note: wiki is loaded automatically by the environment from policy.md
        # No need to restore it from trajectory as it's environment-specific
        
        # Initialize user's message state with the persona instruction
        # This is critical for continued conversations - without this, the user
        # simulation won't have the system prompt or conversation context
        simulator.env.user.reset(instruction=simulator.persona or simulator.env.task.instruction)
        
        # Restore conversation history
        simulator.restore_conversation(trajectory.messages)
        
        # Restore done state
        simulator.state.is_done = trajectory.is_done
        if trajectory.is_done:
            simulator.state.last_reward = trajectory.reward
            simulator.state.reward_info = trajectory.reward_info
        
        # Restore expected actions if available
        if trajectory.expected_actions:
            simulator.state.expected_actions = trajectory.expected_actions
        
        _debug_log(f"Restored {len(trajectory.messages)} messages, is_done={trajectory.is_done}")
        
        return simulator

    def restore_conversation(self, messages: List["TrajectoryMessage"]) -> None:
        """
        Restore conversation history from trajectory messages.
        
        This clears any existing conversation and rebuilds from the saved messages.
        Also rebuilds the user simulation's internal message state so it can
        continue generating contextually appropriate responses.
        
        Args:
            messages: List of TrajectoryMessage objects from saved trajectory
        """
        from sigma.trajectory_storage import TrajectoryMessage
        
        # Clear existing conversation
        self.state.conversation_history.clear()
        
        for msg in messages:
            # Handle both TrajectoryMessage objects and dicts
            if isinstance(msg, TrajectoryMessage):
                role = msg.role
                content = msg.content
                tool_name = msg.tool_name
                tool_arguments = msg.tool_arguments
                msg_id = msg.id
            else:
                role = msg.get('role', '')
                content = msg.get('content', '')
                tool_name = msg.get('tool_name')
                tool_arguments = msg.get('tool_arguments')
                msg_id = msg.get('id')
            
            if role == 'user':
                self._add_conversation_entry('user', content=content, entry_id=msg_id)
                # Also add to user simulation's message history
                # The user's (customer's) messages are stored as "assistant" role in the LLM context
                # (since the LLM is playing the customer role)
                self.env.user.messages.append({"role": "assistant", "content": content})
            elif role == 'agent':
                if tool_name:
                    self._add_conversation_entry('agent', tool_call={'name': tool_name, 'arguments': tool_arguments}, entry_id=msg_id)
                else:
                    self._add_conversation_entry('agent', content=content, entry_id=msg_id)
                    # Agent text responses need to be added to user's context as "user" role
                    # (input to the LLM that plays the customer)
                    self.env.user.messages.append({"role": "user", "content": f"The customer service agent says: {content}\n\nGenerate your response as the customer."})
            elif role == 'tool':
                # Tool call entry (legacy format - tool calls are now in agent messages)
                if tool_name:
                    self._add_conversation_entry('agent', tool_call={'name': tool_name, 'arguments': tool_arguments}, entry_id=msg_id)
            elif role == 'tool-result':
                self._add_conversation_entry('tool', content=content, entry_id=msg_id)
            # Skip 'rejected' and 'system' messages as they don't affect simulation state

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
        
        # Get user's data items from augmented_data (orders/reservations)
        augmented_data = self.persona_data.get("augmented_data", {})
        
        # Check for orders or reservations in augmented_data
        for data_key in ["orders", "reservations", "bookings"]:
            items = augmented_data.get(data_key, {})
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
                break  # Only show one type of data items
        
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
        2. All augmented data (orders, reservations, products, flights, etc.)
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
        
        # Inject all augmented data (orders, reservations, products, flights, etc.)
        augmented_data = self.persona_data.get("augmented_data", {})
        if augmented_data:
            for collection_name, collection_items in augmented_data.items():
                if collection_name in self.env.data and isinstance(collection_items, dict):
                    for item_id, item in collection_items.items():
                        self.env.data[collection_name][item_id] = item
                        _debug_log(f"Injected {collection_name} item: {item_id}")

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
        entry_id: Optional[str] = None,
    ):
        """Add an entry to conversation history."""
        import time
        # Generate a unique ID using timestamp if not provided
        if entry_id is None:
            entry_id = str(time.time() * 1000)  # Millisecond timestamp as string
        
        entry = ConversationEntry(
            role=role,
            content=content,
            tool_call=tool_call,
            id=entry_id,
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

    def rollback_to_message_id(self, message_id: str) -> Dict[str, Any]:
        """
        Rollback conversation history to a specific message ID.
        
        This removes the message with the given ID and all entries after it.
        Uses message ID for idempotent operations.
        
        Args:
            message_id: The ID of the message to rollback to. This message and all
                       subsequent messages will be removed.
        
        Returns:
            Dict with success status, removed_count and remaining_count.
        """
        # Find the index of the message with the given ID
        target_index = None
        for i, entry in enumerate(self.state.conversation_history):
            if entry.id == message_id:
                target_index = i
                break
        
        if target_index is None:
            return {
                "success": False,
                "error": f"Message with ID '{message_id}' not found in conversation history",
                "removed_count": 0,
                "remaining_count": len(self.state.conversation_history)
            }
        
        return self.rollback_to_index(target_index)

    def rollback_to_index(self, target_index: int) -> Dict[str, Any]:
        """
        Rollback conversation history to a specific index.
        
        This removes all entries from target_index onwards (inclusive).
        
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
                "id": removed.id,
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

    def regenerate_user_response(
        self, 
        rejected_message: Optional[str] = None,
        feedback: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Regenerate the simulated user's response.
        
        This finds the last user message, removes it and all subsequent messages
        (rolling back to that state), then generates a new user response.
        
        Args:
            rejected_message: The rejected user message to improve upon.
            feedback: User's feedback on why the message was rejected.
        
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
        if feedback:
            self._additional_persona_note = feedback
        
        try:
            # Regenerate user response to the agent message
            new_response = self._generate_user_response_to_message(
                agent_message_before,
                rejected_message=rejected_message,
                feedback=feedback
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
        rejected_message: Optional[str] = None,
        feedback: Optional[str] = None
    ) -> Optional[str]:
        """Generate a user response to an agent message.
        
        Uses the same prompt structure as the normal user simulation to ensure
        consistent response style (short, natural, one thing at a time, etc.)
        This includes conversation history to maintain context.
        
        Args:
            agent_message: The agent message to respond to.
            rejected_message: Optional rejected response to improve upon.
            feedback: Optional feedback on why the response was rejected.
        """
        # Load user guidelines to match normal user generation behavior
        user_guidelines = load_env_guidelines(self.env_name, "user_guidelines.md")
        
        # Build system prompt similar to the normal user simulation
        system_prompt_parts = [
            "# User Simulation Prompt",
            "",
            "You are simulating a customer contacting customer service.",
            "",
            "## Your Task",
            f"{self.current_persona}",
            "",
        ]
        
        # Add environment-specific guidelines if available
        if user_guidelines:
            system_prompt_parts.extend([
                user_guidelines,
                "",
            ])
        
        system_prompt = "\n".join(system_prompt_parts)
        
        # Build message history similar to LLMUserSimulationEnv
        # In user simulation: "user" role = prompts to model, "assistant" role = customer responses
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        
        # Add conversation history (convert from our format to user simulation format)
        # Skip the last agent message since we'll add it explicitly at the end
        history_entries = list(self.state.conversation_history)
        # Find and remove the last agent entry (it will be added as the final prompt)
        last_agent_idx = None
        for i in range(len(history_entries) - 1, -1, -1):
            if history_entries[i].role == "agent" and history_entries[i].content:
                last_agent_idx = i
                break
        if last_agent_idx is not None:
            history_entries = history_entries[:last_agent_idx]
        
        for entry in history_entries:
            if entry.role == "user":
                # Customer's previous messages become "assistant" in user simulation
                messages.append({"role": "assistant", "content": entry.content})
            elif entry.role == "agent":
                # Agent's messages become "user" prompts in user simulation
                if entry.content:
                    messages.append({"role": "user", "content": f"Agent: {entry.content}"})
            # Skip tool calls and tool results - they're internal to agent
        
        # Build the final prompt with the agent message we're responding to
        final_prompt_parts = [f"Agent: {agent_message}"]
        
        # Add rejection context at the bottom if regenerating
        if rejected_message and feedback:
            final_prompt_parts.extend([
                "",
                "## IMPORTANT: Previous Response Rejected",
                f"Your previous response was: \"{rejected_message}\"",
                f"Feedback: {feedback}",
                "",
                "Generate a NEW response that addresses this feedback while maintaining your persona.",
            ])
        elif rejected_message:
            final_prompt_parts.extend([
                "",
                "## IMPORTANT: Previous Response Rejected", 
                f"Your previous response was: \"{rejected_message}\"",
                "",
                "Generate a different, improved response.",
            ])
        elif feedback:
            final_prompt_parts.extend([
                "",
                f"## IMPORTANT GUIDANCE FOR THIS RESPONSE: {feedback}"
            ])
        
        final_prompt_parts.append("\nGenerate your response as the customer.")
        
        messages.append({"role": "user", "content": "\n".join(final_prompt_parts)})
        
        try:
            # Build completion kwargs
            completion_kwargs = {
                "model": self.user_model,
                "custom_llm_provider": self.user_provider,
                "messages": messages,
            }
            if not self.user_model.startswith("gpt-5"):
                completion_kwargs["temperature"] = 0.7
            
            if DEBUG_SIMULATOR:
                print(f"\n{'='*80}")
                print(f"[DEBUG SIMULATOR] _generate_user_response - Sending to {self.user_model} via {self.user_provider}:")
                print(f"{'='*80}")
                for i, msg in enumerate(messages):
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
            
            # Log agent generation
            _log_agent_generation(
                operation="Generate Response",
                model=self.agent_model,
                provider=self.agent_provider,
                prompt_preview=f"Prompt: {prompt}"
            )
            
            res = completion(**completion_kwargs)
            response = res.choices[0].message.content.strip()
            
            # Log agent generation response
            _log_agent_generation(
                operation="Generate Response - Result",
                model=self.agent_model,
                provider=self.agent_provider,
                response=response
            )
            
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
        if self.state.is_done:
            return {
                "error": "Simulation is already complete",
                "action_type": None,
                "reasoning": "The conversation has ended."
            }
        
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
            
            # Log agent generation (user input preview)
            _log_agent_generation(
                operation="Parse Action",
                model=self.agent_model,
                provider=self.agent_provider,
                prompt_preview=f"User instruction: {user_input}"
            )
            
            res = completion(**completion_kwargs)
            
            response_text = res.choices[0].message.content.strip()
            
            # Log agent generation response
            _log_agent_generation(
                operation="Parse Action - Response",
                model=self.agent_model,
                provider=self.agent_provider,
                response=response_text
            )
            
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

    def regenerate_action_with_feedback(
        self,
        rejected_action: Dict[str, Any],
        feedback: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Regenerate an agent action considering a previously rejected action and user feedback.
        
        Args:
            rejected_action: The previously rejected action with action_type, content/tool_name, arguments, reasoning
            feedback: Optional user feedback on why the action was rejected
            
        Returns:
            New parsed action dict with reasoning, action_type and relevant fields.
        """
        tools_summary = self._get_tools_summary()
        context = self._build_agent_context()
        
        # Build rejected action description
        rejected_desc = []
        if rejected_action.get("action_type") == "respond":
            rejected_desc.append(f"Type: Text response to customer")
            rejected_desc.append(f"Response: {rejected_action.get('content', '')}")
        else:
            rejected_desc.append(f"Type: Tool call")
            rejected_desc.append(f"Tool: {rejected_action.get('tool_name', '')}")
            rejected_desc.append(f"Arguments: {json.dumps(rejected_action.get('arguments', {}), indent=2)}")
        if rejected_action.get("reasoning"):
            rejected_desc.append(f"Reasoning: {rejected_action.get('reasoning', '')}")
        
        rejected_str = "\n".join(rejected_desc)
        
        # Build feedback section
        feedback_section = ""
        if feedback:
            feedback_section = f"\n## Operator Feedback\n{feedback}\n"
        
        # Load the base prompt template and modify it
        template = build_agent_prompt(self.env_name, "action")
        
        # Add rejected action context before the output format section
        rejection_context = f"""
## Previously Rejected Action
The following action was rejected by the operator and should NOT be repeated:
{rejected_str}
{feedback_section}
Please generate a DIFFERENT and BETTER action that addresses the operator's concerns.
"""
        
        # Find where to insert the rejection context (before "## Output Format")
        output_format_idx = template.find("## Output Format")
        if output_format_idx > 0:
            template = template[:output_format_idx] + rejection_context + "\n" + template[output_format_idx:]
        else:
            # Fallback: append to end of template before replacements
            template = template + "\n" + rejection_context
        
        # Create system prompt with placeholders filled
        system_prompt = (template
            .replace("{tools_summary}", tools_summary)
            .replace("{context}", context)
            .replace("{user_input}", "Generate a better action based on the context and avoiding the rejected approach"))

        try:
            # Build completion kwargs - some models don't support temperature
            completion_kwargs = {
                "model": self.agent_model,
                "custom_llm_provider": self.agent_provider,
                "messages": [{"role": "user", "content": system_prompt}],
            }
            # Only add temperature for models that support it (not gpt-5 series)
            if not self.agent_model.startswith("gpt-5"):
                completion_kwargs["temperature"] = 0.5  # Slightly higher temp for more variety
            
            if DEBUG_SIMULATOR:
                print(f"\n{'='*80}")
                print(f"[DEBUG SIMULATOR] regenerate_action_with_feedback - Sending to {self.agent_model} via {self.agent_provider}:")
                print(f"{'='*80}")
                for i, msg in enumerate(completion_kwargs["messages"]):
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')
                    print(f"\n--- Message {i+1} [{role.upper()}] ---")
                    print(content)
                print(f"{'='*80}\n")
            
            # Log agent generation
            _log_agent_generation(
                operation="Regenerate Action",
                model=self.agent_model,
                provider=self.agent_provider,
                prompt_preview=f"Rejected: {rejected_action.get('action_type')}, Feedback: {feedback or 'None'}"
            )
            
            res = completion(**completion_kwargs)
            
            response_text = res.choices[0].message.content.strip()
            
            # Log agent generation response
            _log_agent_generation(
                operation="Regenerate Action - Response",
                model=self.agent_model,
                provider=self.agent_provider,
                response=response_text
            )
            
            if DEBUG_SIMULATOR:
                print(f"\n{'='*80}")
                print(f"[DEBUG SIMULATOR] regenerate_action_with_feedback - Response:")
                print(f"{'='*80}")
                print(response_text)
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
            print(f"[regenerate_action_with_feedback] JSON decode error: {e}")
            print(f"[regenerate_action_with_feedback] Response text: {response_text[:500] if 'response_text' in dir() else 'N/A'}")
            return None
        except Exception as e:
            print(f"[regenerate_action_with_feedback] Error: {type(e).__name__}: {e}")
            return None

    def check_policy_compliance(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if a proposed agent action complies with the policy.
        
        Uses Policy AI (always gpt-5.2 for best quality) to evaluate whether 
        the action confidently follows policy.
        Returns approval if confident, or flags for human review if uncertain.
        
        Args:
            action: The proposed action dict with action_type, content/tool_name, arguments, reasoning
            
        Returns:
            Dict with:
                - approved: bool - True if Policy AI confidently approves
                - confidence: str - "high", "medium", or "low"
                - reason: str - Explanation for the decision
                - policy_concerns: list[str] - Any specific policy concerns identified
                - model_used: str - The model used for evaluation
                - timestamp: str - When the check was performed
        """
        import datetime
        
        # Always use gpt-5.2 for approval AI for best quality
        APPROVER_MODEL = "gpt-5.2"
        APPROVER_PROVIDER = "openai"
        
        # Build action description for the policy check
        if action.get("action_type") == "respond":
            action_desc = f"Text response to customer:\n\"{action.get('content', '')}\""
        elif action.get("action_type") == "tool_call":
            tool_name = action.get("tool_name", "unknown")
            args = json.dumps(action.get("arguments", {}), indent=2)
            action_desc = f"Tool call: {tool_name}\nArguments:\n{args}"
        else:
            action_desc = f"Action: {json.dumps(action, indent=2)}"
        
        reasoning = action.get("reasoning", "No reasoning provided")
        
        # Build context with FULL conversation history (not just last 10)
        conv_history = []
        for entry in self.state.conversation_history:  # Full history for complete context
            if entry.role == "user":
                conv_history.append(f"CUSTOMER: {entry.content or ''}")
            elif entry.role == "agent":
                if entry.tool_call:
                    tc = entry.tool_call
                    tool_args = json.dumps(tc.get('arguments', {}))
                    conv_history.append(f"AGENT: [Called {tc['name']} with args: {tool_args}]")
                else:
                    conv_history.append(f"AGENT: {entry.content or ''}")
            elif entry.role == "tool":
                result = entry.content or ""
                conv_history.append(f"TOOL RESULT: {result}")
        
        conv_context = "\n".join(conv_history) if conv_history else "No conversation history yet."
        
        # Log conversation history size for debugging
        print(f"[check_policy_compliance] Conversation history: {len(self.state.conversation_history)} entries")
        
        system_prompt = f"""You are a Policy Compliance AI that reviews customer service agent actions in a SIMULATION environment.

Your SOLE JOB is to determine if the proposed agent action complies with the POLICY given the AVAILABLE TOOLS.
You must be CONFIDENT to approve. If there's ANY uncertainty about policy compliance, flag for human review.

# ⚠️ CRITICAL: WHAT YOU MUST EVALUATE ⚠️
- Does this action follow the PROCEDURES and RULES defined in the policy?
- Is this action appropriate given the conversation context and customer request?
- Is the agent using the available tools correctly according to policy?

# ⚠️ CRITICAL: WHAT YOU MUST NOT EVALUATE ⚠️
- DO NOT critique tool design, missing tool parameters, or tool limitations
- DO NOT suggest improvements to the tools themselves
- DO NOT flag issues about what tools "should" include or support
- DO NOT comment on the system architecture or implementation
- The tools are FIXED - evaluate only if the agent is USING them correctly per policy

# Policy
{self.env.wiki}

# Conversation History (for context only - all previous actions were already approved)
{conv_context}

# ⚠️ NEW PROPOSED ACTION TO EVALUATE ⚠️
{action_desc}

# Agent's Reasoning for this NEW action
{reasoning}

# Your Task
Evaluate ONLY whether the NEW PROPOSED ACTION complies with POLICY. Accept the tools AS-IS.

Focus your analysis STRICTLY on:
1. Does this action follow the required procedures in the policy?
2. Is this action appropriate given the current conversation state?
3. Is the agent using the correct tool with appropriate arguments for this situation?
4. Are there any POLICY violations (NOT tool design critiques)?

Respond with JSON only:
```json
{{
    "approved": true/false,
    "confidence": "high"/"medium"/"low",
    "reason": "Brief explanation focused on POLICY compliance only",
    "policy_concerns": ["List of POLICY concerns only - not tool design issues"],
    "analysis": "Analysis of policy compliance - not tool critique"
}}
```

Rules:
- ONLY evaluate policy compliance - NOT tool design or limitations
- Only approve with "high" confidence if the action clearly follows policy
- If confidence is "medium" or "low", set approved to false
- Be conservative - when in doubt, flag for human review
- Look for: missing confirmations, skipped authentication, wrong procedures, inappropriate responses
- DO NOT reject because a tool "doesn't include parameter X" - that's out of scope"""

        try:
            completion_kwargs = {
                "model": APPROVER_MODEL,
                "custom_llm_provider": APPROVER_PROVIDER,
                "messages": [{"role": "user", "content": system_prompt}],
            }
            if not APPROVER_MODEL.startswith("gpt-5"):
                completion_kwargs["temperature"] = 0.1  # Low temp for consistent evaluation
            
            # Log the complete prompt for debugging
            print("\n" + "="*80)
            print("[check_policy_compliance] COMPLETE PROMPT SENT TO APPROVAL AI:")
            print("="*80)
            print(system_prompt)
            print("="*80 + "\n")
            
            _log_agent_generation(
                operation="Policy Compliance Check",
                model=APPROVER_MODEL,
                provider=APPROVER_PROVIDER,
                prompt_preview=f"Checking action: {action.get('action_type', 'unknown')} | History: {len(self.state.conversation_history)} entries"
            )
            
            res = completion(**completion_kwargs)
            response_text = res.choices[0].message.content.strip()
            
            _log_agent_generation(
                operation="Policy Compliance Check - Result",
                model=APPROVER_MODEL,
                provider=APPROVER_PROVIDER,
                response=response_text
            )
            
            # Extract JSON from response
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end]
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end]
            
            result = json.loads(response_text.strip())
            
            # Ensure required fields exist with timestamp and model info
            return {
                "approved": result.get("approved", False) and result.get("confidence") == "high",
                "confidence": result.get("confidence", "low"),
                "reason": result.get("reason", "No reason provided"),
                "policy_concerns": result.get("policy_concerns", []),
                "analysis": result.get("analysis", "No detailed analysis provided"),
                "model_used": APPROVER_MODEL,
                "timestamp": datetime.datetime.now().isoformat(),
                "action_checked": action
            }
            
        except json.JSONDecodeError as e:
            print(f"[check_policy_compliance] JSON decode error: {e}")
            return {
                "approved": False,
                "confidence": "low",
                "reason": "Failed to parse policy check response",
                "policy_concerns": ["Technical error in policy evaluation"],
                "analysis": f"JSON parsing failed: {str(e)}",
                "model_used": APPROVER_MODEL,
                "timestamp": datetime.datetime.now().isoformat(),
                "action_checked": action
            }
        except Exception as e:
            print(f"[check_policy_compliance] Error: {type(e).__name__}: {e}")
            return {
                "approved": False,
                "confidence": "low",
                "reason": f"Policy check failed: {str(e)}",
                "policy_concerns": ["Technical error in policy evaluation"],
                "analysis": f"Error during evaluation: {type(e).__name__}: {str(e)}",
                "model_used": APPROVER_MODEL,
                "timestamp": datetime.datetime.now().isoformat(),
                "action_checked": action
            }

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

    def create_from_trajectory(
        self,
        trajectory: "TrajectoryData",
        user_model: Optional[str] = None,
        user_provider: Optional[str] = None,
        agent_model: Optional[str] = None,
        agent_provider: Optional[str] = None,
    ) -> SimulatorCore:
        """
        Create a new session from saved trajectory data.
        
        This is the recommended way to resume a simulation from saved state.
        It uses SimulatorCore.from_trajectory() to properly restore all state.
        
        Args:
            trajectory: TrajectoryData object (from storage or constructed)
            user_model: Override user model (uses trajectory's model if not provided)
            user_provider: Override user provider
            agent_model: Override agent model
            agent_provider: Override agent provider
            
        Returns:
            SimulatorCore instance ready for continued simulation
        """
        simulator = SimulatorCore.from_trajectory(
            trajectory=trajectory,
            user_model=user_model,
            user_provider=user_provider,
            agent_model=agent_model,
            agent_provider=agent_provider,
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
