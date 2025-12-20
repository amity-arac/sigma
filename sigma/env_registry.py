# Copyright Amity
"""
Environment Registry for Sigma

This module provides a registry system for environments that AUTO-DETECTS
available environments from the sigma/envs/ folder.

An environment is detected if it has:
- db.json: Combined database
- tasks.json: Tasks in the new format
- policy.md: Agent policy
- split_tasks.json: Train/test/dev splits (optional)

To add a new environment:
1. Create a new folder under sigma/envs/<env_name>/
2. Add the required files (db.json, tasks.json, policy.md)
3. The environment will be automatically detected and registered
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type
import os
import json


# Path to environments folder
ENVS_PATH = os.path.join(os.path.dirname(__file__), "envs")


@dataclass
class EnvironmentConfig:
    """Configuration for an environment."""
    name: str
    display_name: str
    description: str
    
    # Data schema examples (for persona generation)
    user_schema: str
    order_schema: str  # Generic name - could be orders, reservations, etc.
    order_key: str  # Key name in data dict: "orders", "reservations", etc.
    additional_context: str  # Products info, airports, etc.
    
    # Environment class loader
    env_class_loader: Callable[[], Type]
    
    # Data loader for accessing raw data
    data_loader: Callable[[], Dict[str, Any]]
    
    # Task splits available
    task_splits: List[str] = field(default_factory=lambda: ["test"])
    
    # Tasks loader by split
    tasks_loader: Optional[Callable[[str], List]] = None
    
    # Scenario examples for persona creator UI
    scenario_examples: List[str] = field(default_factory=list)
    
    @property
    def data_key(self) -> str:
        """Alias for order_key for clarity."""
        return self.order_key
    
    @property
    def persona_schemas(self) -> str:
        """Get combined schema context for persona generation."""
        return f"""
# User Schema
{self.user_schema}

# {self.order_key.title()} Schema
{self.order_schema}

# Additional Context
{self.additional_context}
"""


# Global registry
_ENVIRONMENT_REGISTRY: Dict[str, EnvironmentConfig] = {}


def register_environment(config: EnvironmentConfig):
    """Register an environment configuration."""
    _ENVIRONMENT_REGISTRY[config.name] = config


def get_environment_config(name: str) -> EnvironmentConfig:
    """Get environment configuration by name."""
    if name not in _ENVIRONMENT_REGISTRY:
        available = ", ".join(_ENVIRONMENT_REGISTRY.keys())
        raise ValueError(f"Unknown environment: {name}. Available: {available}")
    return _ENVIRONMENT_REGISTRY[name]


def list_environments() -> List[str]:
    """List all registered environment names."""
    return list(_ENVIRONMENT_REGISTRY.keys())


def get_all_environment_configs() -> Dict[str, EnvironmentConfig]:
    """Get all registered environment configurations."""
    return _ENVIRONMENT_REGISTRY.copy()


# =============================================================================
# Schema Templates
# =============================================================================

RETAIL_USER_SCHEMA = """
{
    "user_id": "firstname_lastname_1234",
    "name": { "first_name": "John", "last_name": "Doe" },
    "address": {
        "address1": "123 Main Street",
        "address2": "Apt 4B",
        "city": "New York",
        "country": "USA",
        "state": "NY",
        "zip": "10001"
    },
    "email": "john.doe1234@example.com",
    "payment_methods": {
        "credit_card_1234567": {
            "source": "credit_card",
            "brand": "visa",
            "last_four": "1234",
            "id": "credit_card_1234567"
        },
        "gift_card_7654321": {
            "source": "gift_card",
            "balance": 150.00,
            "id": "gift_card_7654321"
        },
        "paypal_9876543": {
            "source": "paypal",
            "id": "paypal_9876543"
        }
    },
    "orders": ["#W1234567", "#W7654321"]
}
"""

RETAIL_ORDER_SCHEMA = """
{
    "order_id": "#W1234567",
    "user_id": "john_doe_1234",
    "address": {
        "address1": "123 Main Street",
        "address2": "Apt 4B",
        "city": "New York",
        "country": "USA",
        "state": "NY",
        "zip": "10001"
    },
    "items": [
        {
            "name": "T-Shirt",
            "product_id": "9523456873",
            "item_id": "1234567890",
            "price": 49.99,
            "options": {
                "color": "blue",
                "size": "M",
                "material": "cotton",
                "style": "crew neck"
            }
        }
    ],
    "fulfillments": [
        {
            "tracking_id": ["123456789012"],
            "item_ids": ["1234567890"]
        }
    ],
    "status": "delivered",
    "payment_history": [
        {
            "transaction_type": "payment",
            "amount": 49.99,
            "payment_method_id": "credit_card_1234567"
        }
    ]
}

Status can be: pending, processed, delivered, cancelled
"""

RETAIL_PRODUCTS_INFO = """
Available product types:
- T-Shirt (options: color, size, material, style)
- Jeans (options: color, size, fit, material)  
- Sneakers (options: color, size, material)
- Backpack (options: color, size, material, compartments)
- Water Bottle (options: capacity, material, color)
- Office Chair (options: material, color, armrest, backrest height)
- Laptop (options: brand, screen size, processor, RAM, storage)
- Headphones (options: type, connectivity, color)
- Smart Watch (options: brand, color, band material, screen size)
- Electric Kettle (options: capacity, material, color)
- Mechanical Keyboard (options: switch type, backlight, size)
- Smart Thermostat (options: compatibility, color)
- Action Camera (options: resolution, waterproof, color)
- Bookshelf (options: material, color, height)
- Desk Lamp (options: color, brightness, power source)
"""

AIRLINE_USER_SCHEMA = """
{
    "user_id": "firstname_lastname_1234",
    "name": { "first_name": "John", "last_name": "Doe" },
    "address": {
        "address1": "123 Main Street",
        "address2": "Apt 4B",
        "city": "New York",
        "country": "USA",
        "state": "NY",
        "zip": "10001"
    },
    "email": "john.doe1234@example.com",
    "dob": "1985-06-15",
    "payment_methods": {
        "credit_card_1234567": {
            "source": "credit_card",
            "brand": "visa",
            "last_four": "1234",
            "id": "credit_card_1234567"
        },
        "gift_card_7654321": {
            "source": "gift_card",
            "amount": 500,
            "id": "gift_card_7654321"
        },
        "certificate_9876543": {
            "source": "certificate",
            "amount": 250,
            "id": "certificate_9876543"
        }
    },
    "saved_passengers": [
        { "first_name": "Jane", "last_name": "Doe", "dob": "1987-03-20" }
    ],
    "membership": "gold",
    "reservations": ["ABC123", "XYZ789"]
}

Membership can be: regular, silver, gold
"""

AIRLINE_RESERVATION_SCHEMA = """
{
    "reservation_id": "ABC123",
    "user_id": "john_doe_1234",
    "origin": "JFK",
    "destination": "LAX",
    "flight_type": "round_trip",
    "cabin": "economy",
    "flights": [
        {
            "origin": "JFK",
            "destination": "LAX",
            "flight_number": "HAT100",
            "date": "2024-05-20",
            "price": 350
        },
        {
            "origin": "LAX",
            "destination": "JFK",
            "flight_number": "HAT101",
            "date": "2024-05-25",
            "price": 380
        }
    ],
    "passengers": [
        { "first_name": "John", "last_name": "Doe", "dob": "1985-06-15" }
    ],
    "payment_history": [
        { "payment_id": "credit_card_1234567", "amount": 730 }
    ],
    "created_at": "2024-05-01T10:30:00",
    "total_baggages": 2,
    "nonfree_baggages": 1,
    "insurance": "yes"
}

flight_type can be: one_way, round_trip
cabin can be: basic_economy, economy, business
insurance can be: yes, no
"""

AIRLINE_AIRPORTS = """
Available airports: JFK, LAX, ORD, DFW, DEN, SFO, SEA, ATL, BOS, MIA, 
PHX, IAH, LAS, MCO, EWR, MSP, DTW, PHL, LGA, CLT

The current simulation date is 2024-05-15.
"""


# =============================================================================
# Auto-Detection of Environments
# =============================================================================

def _is_valid_environment(env_path: str) -> bool:
    """Check if a folder contains a valid environment (has required files)."""
    required_files = ["db.json", "tasks.json", "policy.md"]
    return all(
        os.path.exists(os.path.join(env_path, f))
        for f in required_files
    )


def _detect_environments() -> List[str]:
    """Auto-detect available environments from sigma/envs/ folder."""
    environments = []
    
    if not os.path.exists(ENVS_PATH):
        return environments
    
    for item in os.listdir(ENVS_PATH):
        item_path = os.path.join(ENVS_PATH, item)
        if os.path.isdir(item_path) and not item.startswith("_") and not item.startswith("."):
            if _is_valid_environment(item_path):
                environments.append(item)
    
    return environments


def _create_env_data_loader(env_name: str):
    """Create a data loader function for an environment."""
    def loader():
        env_path = os.path.join(ENVS_PATH, env_name)
        db_path = os.path.join(env_path, "db.json")
        with open(db_path, "r") as f:
            return json.load(f)
    return loader


def _create_env_tasks_loader(env_name: str):
    """Create a tasks loader function for an environment."""
    def loader(split: str):
        env_path = os.path.join(ENVS_PATH, env_name)
        tasks_path = os.path.join(env_path, "tasks.json")
        split_path = os.path.join(env_path, "split_tasks.json")
        
        with open(tasks_path, "r") as f:
            all_tasks = json.load(f)
        
        # If split_tasks.json exists, filter by split
        if os.path.exists(split_path):
            with open(split_path, "r") as f:
                splits = json.load(f)
            if split in splits:
                split_ids = set(splits[split])
                return [task for task in all_tasks if str(task.get("id")) in split_ids]
        
        # Return all tasks if no split filtering
        return all_tasks
    return loader


def _create_env_class_loader(env_name: str):
    """Create an environment class loader using the generic env."""
    def loader():
        from sigma.envs.generic_env import get_env_class
        return get_env_class(env_name)
    return loader


def _get_task_splits(env_name: str) -> List[str]:
    """Get available task splits for an environment."""
    env_path = os.path.join(ENVS_PATH, env_name)
    split_path = os.path.join(env_path, "split_tasks.json")
    
    if os.path.exists(split_path):
        with open(split_path, "r") as f:
            splits = json.load(f)
        return list(splits.keys())
    
    return ["test"]


def _get_env_description(env_name: str) -> str:
    """Get description from policy.md or generate default."""
    env_path = os.path.join(ENVS_PATH, env_name)
    policy_path = os.path.join(env_path, "policy.md")
    
    if os.path.exists(policy_path):
        with open(policy_path, "r") as f:
            content = f.read()
        # Extract first paragraph after title
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if line.startswith("As a ") or line.startswith("You are "):
                # Return first sentence
                return line.split(".")[0] + "."
    
    return f"{env_name.title()} customer service environment"


def _auto_register_environments():
    """Auto-detect and register all valid environments."""
    detected = _detect_environments()
    
    for env_name in detected:
        # Use retail schemas as default (can be customized per env later)
        user_schema = RETAIL_USER_SCHEMA
        order_schema = RETAIL_ORDER_SCHEMA
        order_key = "orders"
        additional_context = RETAIL_PRODUCTS_INFO
        
        config = EnvironmentConfig(
            name=env_name,
            display_name=env_name.title(),
            description=_get_env_description(env_name),
            user_schema=user_schema,
            order_schema=order_schema,
            order_key=order_key,
            additional_context=additional_context,
            env_class_loader=_create_env_class_loader(env_name),
            data_loader=_create_env_data_loader(env_name),
            task_splits=_get_task_splits(env_name),
            tasks_loader=_create_env_tasks_loader(env_name),
            scenario_examples=[],
        )
        register_environment(config)
    
    return detected


# Auto-register environments on module load
_detected_envs = _auto_register_environments()


# =============================================================================
# Helper for adding new environments (sigma format)
# =============================================================================

def create_sigma_environment(env_name: str, output_dir: str = None):
    """
    Create a new sigma environment with data files only.
    
    Sigma environments contain only data/content files:
    - db.json: Database
    - tasks.json: Tasks
    - split_tasks.json: Train/test/dev splits
    - policy.html or policy.md: Agent policy
    - user_guidelines.md: (optional) User simulation guidelines
    
    No Python files - all logic is in sigma.envs.generic_env module.
    
    Args:
        env_name: Name for the new environment
        output_dir: Directory to create env in (default: sigma/envs/)
    """
    import os
    
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "envs")
    
    env_dir = os.path.join(output_dir, env_name)
    os.makedirs(env_dir, exist_ok=True)
    
    # Create data file templates (no __init__.py - pure data folder)
    templates = {
        "db.json": '{\n    "users": {},\n    "products": {},\n    "orders": {}\n}\n',
        "tasks.json": '[\n    {\n        "id": "1",\n        "user_scenario": {\n            "instructions": {\n                "known_info": "You are user_123",\n                "reason_for_call": "You want to check your order status",\n                "unknown_info": "",\n                "task_instructions": "Be polite and provide information when asked"\n            }\n        },\n        "evaluation_criteria": {\n            "actions": [],\n            "communicate_info": []\n        }\n    }\n]\n',
        "split_tasks.json": '{\n    "train": ["1"],\n    "test": ["1"],\n    "dev": ["1"]\n}\n',
        "policy.md": f'# {env_name.title()} Agent Policy\n\n## Overview\n\nDescribe the agent policy here.\n\n## Guidelines\n\n- Guideline 1\n- Guideline 2\n',
        "user_guidelines.md": f'# {env_name.title()} User Simulation Guidelines\n\n## Behavior\n\nDescribe how the simulated user should behave.\n',
    }
    
    for filename, content in templates.items():
        filepath = os.path.join(env_dir, filename)
        with open(filepath, "w") as f:
            f.write(content)
    
    return env_dir


# =============================================================================
# Legacy: environment template (for reference)
# =============================================================================

def create_environment_template(env_name: str, output_dir: str):
    """
    [LEGACY] Create a template for an environment with Python files.
    
    NOTE: For sigma environments, use create_sigma_environment() instead.
    Sigma environments should only contain data files, no Python logic.
    
    This generates the folder structure for environments which
    have Python files for logic (env.py, rules.py, etc.).
    """
    import os
    
    env_dir = os.path.join(output_dir, env_name)
    os.makedirs(env_dir, exist_ok=True)
    os.makedirs(os.path.join(env_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(env_dir, "tools"), exist_ok=True)
    
    # Create __init__.py
    init_content = f'''# Copyright Amity

from sigma.envs.{env_name}.env import Mock{env_name.title()}DomainEnv

__all__ = ["Mock{env_name.title()}DomainEnv"]
'''
    with open(os.path.join(env_dir, "__init__.py"), "w") as f:
        f.write(init_content)
    
    # Create env.py template
    env_content = f'''# Copyright Amity

from sigma.envs.base import Env
from sigma.envs.{env_name}.data import load_data
from sigma.envs.{env_name}.rules import RULES
from sigma.envs.{env_name}.tools import ALL_TOOLS
from sigma.envs.{env_name}.wiki import WIKI
from typing import Optional, Union
from sigma.envs.user import UserStrategy


class Mock{env_name.title()}DomainEnv(Env):
    def __init__(
        self,
        user_strategy: Union[str, UserStrategy] = UserStrategy.LLM,
        user_model: str = "gpt-4o",
        user_provider: Optional[str] = None,
        task_split: str = "test",
        task_index: Optional[int] = None,
    ):
        match task_split:
            case "test":
                from sigma.envs.{env_name}.tasks_test import TASKS
            case _:
                raise ValueError(f"Unknown task split: {{task_split}}")
        super().__init__(
            data_load_func=load_data,
            tools=ALL_TOOLS,
            tasks=TASKS,
            wiki=WIKI,
            rules=RULES,
            user_strategy=user_strategy,
            user_model=user_model,
            user_provider=user_provider,
            task_index=task_index,
        )
        self.terminate_tools = ["transfer_to_human_agents"]
'''
    with open(os.path.join(env_dir, "env.py"), "w") as f:
        f.write(env_content)
    
    # Create other templates...
    templates = {
        "rules.py": '# Copyright Amity\n\nRULES = [\n    "Rule 1",\n    "Rule 2",\n]\n',
        "policy.md": f'# {env_name.title()} Agent Policy\n\nDescribe the agent policy here.\n',
        "tasks.py": '# Copyright Amity\n\ntasks = [\n    # Add tasks here\n]\n',
        "tasks_test.py": '# Copyright Amity\n\nTASKS = [\n    # Add test tasks here\n]\n',
        "data/__init__.py": '# Copyright Amity\n\nimport json\nimport os\nfrom typing import Any\n\nFOLDER_PATH = os.path.dirname(__file__)\n\n\ndef load_data() -> dict[str, Any]:\n    # Load your data files here\n    return {}\n',
        "tools/__init__.py": '# Copyright Amity\n\n# from .your_tool import YourTool\n\nALL_TOOLS = [\n    # Add tool classes here\n]\n',
    }
    
    for filename, content in templates.items():
        filepath = os.path.join(env_dir, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            f.write(content)
    
    return env_dir
