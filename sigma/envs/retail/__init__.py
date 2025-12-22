# Copyright Amity
"""
Retail Environment

This module contains the tools.py file for the retail environment.
Data files (db.json, tasks.json, policy.md, etc.) are stored in data/envs/retail/.

Files in data/envs/retail/:
- db.json: Database with products, users, orders
- tasks.json: Tasks with user_scenario and evaluation_criteria
- policy.md: Agent policy (includes behavioral rules)
- user_guidelines.md: User simulation guidelines
- agent_guidelines.md: Agent guidelines

Source files in sigma/envs/retail/:
- tools.py: Custom tool implementations (only Python file)
"""

from sigma.envs.generic_env import create_generic_env_class

# Create the environment class using the generic loader
MockRetailDomainEnv = create_generic_env_class("retail")
