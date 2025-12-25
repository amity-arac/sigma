# Copyright Amity
"""
Retail Environment

This module creates the retail environment class.
All data and configuration files are stored in data/envs/retail/:
- db.json: Database with products, users, orders
- tasks.json: Tasks with user_scenario and evaluation_criteria
- policy.md: Agent policy (includes guidelines and behavioral rules)
- user_guidelines.md: User simulation guidelines
- tools.py: Custom tool implementations
"""

from sigma.envs.generic_env import create_generic_env_class

# Create the environment class using the generic loader
MockRetailDomainEnv = create_generic_env_class("retail")
