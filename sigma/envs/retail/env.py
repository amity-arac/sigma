# Copyright Amity
"""
Retail Environment

Uses shared loading logic from sigma.envs.generic_env.
"""

import os
from sigma.envs.base import Env
from sigma.envs.retail.data import load_data
from sigma.envs.retail.rules import RULES
from sigma.envs.retail.tools import ALL_TOOLS
from sigma.types import Task, Action
from typing import Optional, Union, List
from sigma.envs.user import UserStrategy

from sigma.envs.generic_env import (
    load_env_tasks,
    load_env_policy,
    get_task_instruction,
    get_task_actions,
    get_task_user_id,
)


def _load_tasks(split: str) -> List[Task]:
    """Load tasks from retail env and convert to Task objects."""
    raw_tasks = load_env_tasks("retail", split)
    tasks = []
    
    for raw_task in raw_tasks:
        # Extract instruction from new format
        instruction = get_task_instruction(raw_task)
        
        # Extract and convert actions
        raw_actions = get_task_actions(raw_task)
        actions = [
            Action(
                name=action.get("name", ""),
                kwargs=action.get("arguments", action.get("kwargs", {})),
            )
            for action in raw_actions
        ]
        
        # Get communicate_info as outputs
        evaluation = raw_task.get("evaluation_criteria", {})
        outputs = evaluation.get("communicate_info", raw_task.get("outputs", []))
        
        # Get user_id
        user_id = get_task_user_id(raw_task)
        
        task = Task(
            user_id=user_id,
            instruction=instruction,
            actions=actions,
            outputs=outputs,
        )
        tasks.append(task)
    
    return tasks


class MockRetailDomainEnv(Env):
    # Path to this environment's folder (for loading user_guidelines.md etc.)
    ENV_PATH = os.path.dirname(__file__)
    
    def __init__(
        self,
        user_strategy: Union[str, UserStrategy] = UserStrategy.LLM,
        user_model: str = "gpt-4o",
        user_provider: Optional[str] = None,
        task_split: str = "test",
        task_index: Optional[int] = None,
    ):
        # Load tasks from retail env
        tasks = _load_tasks(task_split)
        
        # Load policy directly from retail folder
        wiki = load_env_policy("retail")
        
        super().__init__(
            data_load_func=load_data,
            tools=ALL_TOOLS,
            tasks=tasks,
            wiki=wiki,
            rules=RULES,
            user_strategy=user_strategy,
            user_model=user_model,
            user_provider=user_provider,
            task_index=task_index,
            env_path=self.ENV_PATH,
        )
        self.terminate_tools = ["transfer_to_human_agents"]

