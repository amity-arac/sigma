# Copyright Amity
"""
Generic Environment Loader

This module provides shared loading functions for all environments.

Data files are stored in data/envs/<env_name>/:
- db.json: Database with environment data
- tasks.json: Tasks in the standard format
- policy.md: Agent policy (includes behavioral rules)
- user_guidelines.md: (optional) User simulation guidelines
- tools.py: Python tool implementations

All environment files including tools.py are now stored in data/envs/<env_name>/.
"""

import importlib.util
import json
import os
from typing import Any, Dict, List, Optional, Union

from sigma.envs.base import Env
from sigma.envs.user import UserStrategy
from sigma.envs.paths import DATA_ENVS_PATH, SOURCE_ENVS_PATH, ENVS_PATH
from sigma.types import Task, Action


def load_env_data(env_name: str) -> Dict[str, Any]:
    """Load database from db.json for an environment."""
    env_path = os.path.join(ENVS_PATH, env_name)
    db_path = os.path.join(env_path, "db.json")
    with open(db_path, "r") as f:
        return json.load(f)


def load_env_tasks(env_name: str, split: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load tasks from tasks.json for an environment.
    
    Args:
        env_name: Name of the environment
        split: Ignored (kept for API compatibility)
    """
    env_path = os.path.join(ENVS_PATH, env_name)
    tasks_path = os.path.join(env_path, "tasks.json")
    
    with open(tasks_path, "r") as f:
        return json.load(f)


def load_env_policy(env_name: str) -> str:
    """Load policy from policy.md for an environment."""
    env_path = os.path.join(ENVS_PATH, env_name)
    policy_path = os.path.join(env_path, "policy.md")
    
    if os.path.exists(policy_path):
        with open(policy_path, "r") as f:
            return f.read()
    
    return ""


def get_task_instruction(task: Dict[str, Any]) -> str:
    """
    Extract instruction string from a task.
    
    Supports both new format (user_scenario.instructions) and legacy format (instruction field).
    """
    # Check for new format first
    user_scenario = task.get("user_scenario", {})
    instructions = user_scenario.get("instructions", {})
    
    if instructions:
        # New format - combine instruction parts
        parts = []
        
        known_info = instructions.get("known_info", "")
        if known_info:
            parts.append(known_info)
        
        reason = instructions.get("reason_for_call", "")
        if reason:
            parts.append(reason)
        
        unknown_info = instructions.get("unknown_info", "")
        if unknown_info:
            parts.append(unknown_info)
        
        task_instructions = instructions.get("task_instructions", "")
        if task_instructions:
            parts.append(task_instructions)
        
        if parts:
            return " ".join(parts)
    
    # Legacy format
    if "instruction" in task:
        return task["instruction"]
    
    return ""


def get_task_actions(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract actions from a task."""
    # New format
    evaluation = task.get("evaluation_criteria", {})
    if "actions" in evaluation:
        return evaluation["actions"]
    
    # Legacy format
    if "actions" in task:
        return task["actions"]
    
    return []


def get_task_user_id(task: Dict[str, Any]) -> str:
    """Extract user_id from a task."""
    # Direct user_id field
    if "user_id" in task:
        return task["user_id"]
    
    # Try to extract from known_info
    user_scenario = task.get("user_scenario", {})
    instructions = user_scenario.get("instructions", {})
    known_info = instructions.get("known_info", "")
    
    if "You are " in known_info:
        parts = known_info.split()
        for i, part in enumerate(parts):
            if part == "are" and i + 1 < len(parts):
                return parts[i + 1]
    
    return ""


def convert_tasks_to_task_objects(raw_tasks: List[Dict[str, Any]]) -> List[Task]:
    """Convert raw task dicts to Task objects."""
    tasks = []
    
    for raw_task in raw_tasks:
        instruction = get_task_instruction(raw_task)
        
        raw_actions = get_task_actions(raw_task)
        actions = [
            Action(
                name=action.get("name", ""),
                kwargs=action.get("arguments", action.get("kwargs", {})),
            )
            for action in raw_actions
        ]
        
        # Get outputs
        evaluation = raw_task.get("evaluation_criteria", {})
        outputs = evaluation.get("communicate_info", raw_task.get("outputs", []))
        
        user_id = get_task_user_id(raw_task)
        
        task = Task(
            user_id=user_id,
            instruction=instruction,
            actions=actions,
            outputs=outputs,
        )
        tasks.append(task)
    
    return tasks


def load_tools_from_file(env_name: str) -> List:
    """
    Load tools from tools.py in the data/envs/<env_name>/ folder.
    
    Uses importlib.util to dynamically load the Python file from the data folder.
    """
    tools_path = os.path.join(DATA_ENVS_PATH, env_name, "tools.py")
    
    if not os.path.exists(tools_path):
        raise FileNotFoundError(f"Tools file not found: {tools_path}")
    
    # Create a unique module name to avoid conflicts
    module_name = f"sigma_env_tools_{env_name}"
    
    spec = importlib.util.spec_from_file_location(module_name, tools_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load tools from {tools_path}")
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    if not hasattr(module, "ALL_TOOLS"):
        raise AttributeError(f"tools.py must define ALL_TOOLS list")
    
    return module.ALL_TOOLS


def create_generic_env_class(env_name: str):
    """
    Create a generic environment class for the given environment.
    
    This dynamically creates an Env subclass that loads data from the
    environment's data files in data/envs/<env_name>/ including tools.py.
    """
    # Load tools from env's tools.py in data directory
    ALL_TOOLS = load_tools_from_file(env_name)
    
    # Get the path to this environment's data folder
    env_path = os.path.join(DATA_ENVS_PATH, env_name)
    
    class GenericDomainEnv(Env):
        """Generic environment that loads data from JSON/MD files."""
        
        ENV_NAME = env_name
        ENV_PATH = env_path
        
        def __init__(
            self,
            user_strategy: Union[str, UserStrategy] = UserStrategy.LLM,
            user_model: str = "gpt-4o",
            user_provider: Optional[str] = None,
            task_split: str = "test",
            task_index: Optional[int] = None,
        ):
            # Load tasks and convert to Task objects
            raw_tasks = load_env_tasks(self.ENV_NAME, task_split)
            tasks = convert_tasks_to_task_objects(raw_tasks)
            
            # Load policy/wiki
            wiki = load_env_policy(self.ENV_NAME)
            
            # Create data loader function
            def data_loader():
                return load_env_data(self.ENV_NAME)
            
            super().__init__(
                data_load_func=data_loader,
                tools=ALL_TOOLS,
                tasks=tasks,
                wiki=wiki,
                user_strategy=user_strategy,
                user_model=user_model,
                user_provider=user_provider,
                task_index=task_index,
                env_path=self.ENV_PATH,
            )
            self.terminate_tools = ["transfer_to_human_agents"]
    
    # Set a meaningful class name
    GenericDomainEnv.__name__ = f"Mock{env_name.title()}DomainEnv"
    GenericDomainEnv.__qualname__ = f"Mock{env_name.title()}DomainEnv"
    
    return GenericDomainEnv


def get_env_class(env_name: str):
    """Get the environment class for the given environment name."""
    return create_generic_env_class(env_name)
