# Copyright Amity
"""
Sigma - Scenario Simulator for tau-bench

A multi-interface simulation tool where humans can act as agents
interacting with LLM-simulated users.

Available interfaces:
- CLI: python -m sigma --env retail --user-model gpt-4o --user-provider openai
- Web API: python -m sigma.api_server --host 0.0.0.0 --port 8000

Core module (SimulatorCore) can be used programmatically for custom integrations.
"""

from sigma.cli_simulator import CLISimulator
from sigma.simulator_core import (
    SimulatorCore,
    SimulatorSessionManager,
    ActionResult,
    ConversationEntry,
    SimulationState,
    ToolInfo,
    ActionType,
    load_persona_file,
    get_available_environments,
)
from sigma.env_registry import (
    EnvironmentConfig,
    register_environment,
    get_environment_config,
    list_environments,
    get_all_environment_configs,
    create_environment_template,
)
from sigma.trajectory import Trajectory, TrajectoryError
from sigma.trajectory_storage import (
    TrajectoryData,
    TrajectoryMessage,
    RejectedSuggestion,
    get_trajectory_storage,
)

__all__ = [
    # CLI
    "CLISimulator",
    # Core
    "SimulatorCore",
    "SimulatorSessionManager",
    "ActionResult",
    "ConversationEntry",
    "SimulationState",
    "ToolInfo",
    "ActionType",
    "load_persona_file",
    "get_available_environments",
    # Registry
    "EnvironmentConfig",
    "register_environment",
    "get_environment_config",
    "list_environments",
    "get_all_environment_configs",
    "create_environment_template",
    # Trajectory
    "Trajectory",
    "TrajectoryError",
    "TrajectoryData",
    "TrajectoryMessage",
    "RejectedSuggestion",
    "get_trajectory_storage",
]
