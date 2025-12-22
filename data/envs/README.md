# Sigma Environment Data

This directory contains customizable data files for Sigma environments, separated from the core source code.

## Directory Structure

```
data/envs/
└── <env_name>/          # One folder per environment (e.g., retail)
    ├── db.json          # Database with users, products, orders, etc.
    ├── tasks.json       # Tasks with user scenarios and evaluation criteria
    ├── policy.md        # Agent policy, guidelines, and behavioral rules
    ├── user_guidelines.md   # User simulation guidelines
    ├── agent_guidelines.md  # Agent-specific guidelines
    └── tools.py         # Python tool implementations
```

## File Descriptions

### db.json
The main database containing all environment data. Structure varies by environment but typically includes:
- `users`: User profiles with addresses, payment methods, order history
- `products`: Product catalog with variants and pricing
- `orders`: Order records with status, items, payments

### tasks.json
Array of task definitions, each containing:
- `id`: Unique task identifier
- `user_scenario`: User scenario with instructions
  - `instructions.known_info`: Information the user knows
  - `instructions.reason_for_call`: Why the user is contacting support
  - `instructions.unknown_info`: Information the user doesn't know
  - `instructions.task_instructions`: Behavioral instructions for the user
- `evaluation_criteria`: Expected actions and outcomes

### policy.md
Markdown document describing the agent's policy, including:
- What the agent can help with
- Authentication requirements
- Action guidelines
- Behavioral rules and constraints
- Domain-specific procedures

### user_guidelines.md
Guidelines for simulating realistic user behavior:
- How to initiate conversations
- When to provide information
- How to end conversations

### agent_guidelines.md
Additional agent-specific guidelines and context.

### tools.py
Python module containing tool implementations for the environment:
- Tool classes that extend the base `Tool` class
- Each tool has `invoke()` and `get_info()` methods
- `ALL_TOOLS` list exporting all available tools

## Adding a New Environment

1. Create a new folder: `data/envs/<env_name>/`
2. Add the required files:
   - `db.json` (required)
   - `tasks.json` (required)
   - `policy.md` (required)
   - `tools.py` (required) - Python tool implementations
3. Add optional files as needed
4. The environment will be auto-detected on server restart

## Editing Files

Files can be edited:
1. **Via Web UI**: Navigate to `/environments` in the Sigma web interface
2. **Directly**: Edit files in this directory with any text editor

Changes to data files take effect on the next simulation session (no server restart required for most changes).
