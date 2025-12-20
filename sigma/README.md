# Sigma - Scenario Simulator for tau-bench

Sigma is an interactive simulation tool where a human acts as the **agent** while an LLM simulates the **user**. This allows you to practice and test agent responses in realistic customer service scenarios.

## Features

- 🎮 **Multiple Interfaces** - CLI and Web UI available
- 🤖 **LLM User Simulation** - User is simulated by an LLM based on a persona/instruction
- 🎭 **Persona Creator** - Generate custom scenarios with realistic data
- 🔧 **Tool Calling** - Call environment tools with guided parameter input
- 📋 **Wiki/Policy Reference** - View agent policies during interaction
- 📊 **Reward Evaluation** - Get feedback on your performance at the end
- ✨ **LLM-Assisted Responses** - Describe what to say and let LLM generate responses
- 🌐 **REST API** - Programmatic access via FastAPI for custom integrations

## Architecture

Sigma has a modular architecture:

```
sigma/
├── simulator_core.py   # Core simulation logic (UI-agnostic)
├── cli_simulator.py    # CLI interface using Rich
├── api_server.py       # FastAPI web server
├── static/index.html   # Web UI
├── env_registry.py     # Environment registry
└── persona_creator.py  # Persona generation tool
```

## Installation

Make sure you have the required dependencies:

```bash
# Core dependencies (CLI)
pip install litellm rich

# Web dependencies (optional)
pip install fastapi uvicorn websockets

# Or install everything with extras
pip install -e ".[web]"
```

## Quick Start

### Option 1: CLI Interface

```bash
# Use a random task from retail environment
python -m sigma --env retail --user-model gpt-4o --user-provider openai
```

### Option 2: Web Interface

```bash
# Start the web server
python -m sigma.api_server --port 8000

# Open http://localhost:8000 in your browser
```

### Option 3: Create a Custom Persona First (Recommended)

```bash
# Create a persona with generated data that matches your scenario
python -m sigma.persona_creator --env retail --model gpt-4o --provider openai

# Then run with the persona
python -m sigma --env retail --user-model gpt-4o --user-provider openai \
    --persona-file sigma/personas/my_persona_retail.json
```

## CLI Usage

### Simulator Commands

```bash
# Use a random task from existing test data
python -m sigma --env retail --user-model gpt-4o --user-provider openai

# Use a specific task
python -m sigma --env retail --user-model gpt-4o --user-provider openai --task-index 5

# Use a text persona (quick but no data backing)
python -m sigma --env retail --user-model gpt-4o --user-provider openai \
    --persona "You are John. You want to cancel order #W1234567."

# Use a persona file (recommended - has realistic data)
python -m sigma --env retail --user-model gpt-4o --user-provider openai \
    --persona-file sigma/personas/my_persona_retail.json

# List available tasks
python -m sigma --list-tasks --env retail
```

### Persona Creator Commands

```bash
# Create a persona for retail
python -m sigma.persona_creator --env retail --model gpt-4o --provider openai

# Create a persona for airline
python -m sigma.persona_creator --env airline --model gpt-4o --provider openai

# List saved personas
python -m sigma.persona_creator --list
```

### Command Line Options (Simulator)

```
--env {retail,airline}     Environment to use (default: retail)
--user-model MODEL         LLM model for user simulation (default: gpt-4o)
--user-provider PROVIDER   LLM provider for user simulation (default: openai)
--agent-model MODEL        LLM model for response generation (default: same as user-model)
--agent-provider PROVIDER  LLM provider for response generation (default: same as user-provider)
--persona PERSONA          Custom text persona for the simulated user
--persona-file FILE        Path to a persona JSON file (from persona_creator)
--task-index INDEX         Specific task index to use (default: random)
--task-split {test,train,dev}  Task split to use (default: test)
--list-tasks               List available tasks and exit
```

## Web API Usage

### Start the Server

```bash
python -m sigma.api_server --host 0.0.0.0 --port 8000

# Or with uvicorn directly (with auto-reload)
uvicorn sigma.api_server:app --reload --port 8000
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web UI |
| GET | `/health` | Health check |
| GET | `/environments` | List available environments |
| POST | `/sessions` | Create a new session |
| GET | `/sessions` | List all sessions |
| POST | `/sessions/{id}/start` | Start simulation |
| POST | `/sessions/{id}/respond` | Send text response |
| POST | `/sessions/{id}/tool` | Call a tool |
| POST | `/sessions/{id}/generate-response` | Generate AI response |
| GET | `/sessions/{id}/history` | Get conversation history |
| GET | `/sessions/{id}/result` | Get final result |
| POST | `/sessions/{id}/save-trajectory` | Save trajectory to CosmosDB |
| GET | `/trajectory/status` | Check CosmosDB configuration status |
| GET | `/trajectories` | List saved trajectories |
| GET | `/trajectories/{id}` | Get a specific trajectory |
| WS | `/ws/{id}` | WebSocket for real-time updates |

### Trajectory Storage (CosmosDB)

Sigma supports saving session trajectories to Azure CosmosDB for later analysis and training data creation. Trajectories include:

- Task instructions and environment info
- All conversation messages (user, agent, tool calls, tool results)
- Reasoning content for each agent action
- Rejected suggestions (when user cancels an LLM-suggested action)
- Final reward and result information

#### Setup

1. Install the Azure Cosmos SDK:
   ```bash
   pip install azure-cosmos
   ```

2. Set environment variables:
   ```bash
   export COSMOSDB_ENDPOINT="https://your-account.documents.azure.com:443/"
   export COSMOSDB_KEY="your-cosmos-db-key"
   # Optional:
   export COSMOSDB_DATABASE="sigma"       # default: sigma
   export COSMOSDB_CONTAINER="trajectories"  # default: trajectories
   ```

3. The "Save" button will appear in the Chat Header when CosmosDB is configured.

#### Trajectory Data Structure

```json
{
  "id": "uuid",
  "session_id": "session-uuid",
  "created_at": "2025-12-16T10:30:00Z",
  "env_name": "retail",
  "task_index": 5,
  "task_instruction": "User wants to cancel order...",
  "user_model": "gpt-4o",
  "agent_model": "gpt-4o",
  "persona": "You are John...",
  "wiki": "Policy document...",
  "messages": [
    {"id": "1", "role": "user", "content": "Hi, I want to cancel my order", "timestamp": "..."},
    {"id": "2", "role": "agent", "content": "I can help with that...", "reasoning": "User wants to cancel..."},
    {"id": "3", "role": "tool", "content": "Calling get_order_details...", "reasoning": "Need order info"},
    {"id": "4", "role": "tool-result", "content": "{\"order_id\": ...}"}
  ],
  "rejected_suggestions": [
    {"action_type": "respond", "content": "Original suggestion...", "reasoning": "...", "rejected_at": "..."}
  ],
  "is_done": true,
  "reward": 1.0,
  "reward_info": {...}
}
```

### Example API Usage

```python
import requests

# Create session
resp = requests.post("http://localhost:8000/sessions", json={
    "env_name": "retail",
    "user_model": "gpt-4o",
    "user_provider": "openai"
})
session_id = resp.json()["session_id"]

# Start simulation
resp = requests.post(f"http://localhost:8000/sessions/{session_id}/start")
initial_message = resp.json()["initial_message"]

# Send a response
resp = requests.post(f"http://localhost:8000/sessions/{session_id}/respond", json={
    "message": "Hello! How can I help you today?"
})
user_reply = resp.json()["observation"]
```

## Programmatic Usage

Use `SimulatorCore` directly for custom integrations:

```python
from sigma import SimulatorCore

# Create simulator
sim = SimulatorCore(
    env_name="retail",
    user_model="gpt-4o",
    user_provider="openai",
)

# Start and get initial message
initial_msg = sim.start()
print(f"User: {initial_msg}")

# Respond to user
result = sim.respond_to_user("Hello! How can I help?")
print(f"User: {result.observation}")

# Call a tool
result = sim.call_tool("get_user_details", {"user_id": "john_doe_123"})
print(f"Tool result: {result.observation}")

# Check if done
if sim.is_done:
    final = sim.get_final_result()
    print(f"Reward: {final['reward']}")
```

## How It Works

### Persona Creation Flow
1. Describe the scenario you want to simulate
2. LLM generates realistic user data (profile, orders/reservations)
3. Data is saved to a JSON file
4. Use the file in simulation - data is injected into the environment

### Simulation Flow
1. **Start**: The simulation begins with the user persona displayed (for reference)
2. **User Message**: The LLM simulates the user sending a message based on the persona
3. **Agent Action**: You choose to:
   - **Respond** - Type exact text OR describe what to say (LLM generates)
   - **Call Tool** - Select and execute a tool with parameters
   - **View Tools** - See available tools
   - **View Wiki** - Review agent policy
4. **Loop**: Continue until the user sends `###STOP###` (task completed)
5. **Evaluation**: Get reward feedback comparing your actions to ground truth

## Interaction Menu

During simulation, you have these options:

| Key | Action |
|-----|--------|
| 1 | Respond to user |
| 2 | Call a tool |
| 3 | View available tools |
| 4 | View tool details |
| 5 | View wiki/policy |
| 6 | View conversation history |
| b | Go back (in submenus) |
| q | Quit simulation |

### Response Options

When responding to the user, you can:
1. **Type exact response** - Write exactly what you want to say
2. **Describe what to say** - Tell the LLM what you want (e.g., "greet the user and ask for email") and it generates a professional response

## Environment Details

### Retail Environment
Simulates an online retail customer service agent:
- Cancel/modify pending orders
- Return/exchange delivered orders
- Look up user information, orders, products
- Modify user addresses

### Airline Environment
Simulates an airline customer service agent:
- Book/cancel/modify flight reservations
- Search for flights
- Update baggage, passengers, cabin class
- Handle refunds and certificates

## Tips for Good Performance

1. **Always authenticate first** - Find user ID by email or name+zip
2. **Confirm before changes** - Ask for explicit "yes" before database modifications
3. **Use tools efficiently** - Gather information before making changes
4. **Follow the wiki policy** - Read the policy document for domain rules
5. **Don't make up information** - Only use what the user tells you or what tools return

## API Keys

Set your API keys as environment variables:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-..."
```

Or use a `.env` file with `litellm`.

## Adding New Environments

Sigma uses a **registry pattern** that makes it easy to add new industries/domains without modifying core code.

### Using the Environment Registry

```python
from sigma import (
    EnvironmentConfig,
    register_environment,
    list_environments,
    get_environment_config,
)

# List available environments
print(list_environments())  # ['retail', 'airline']

# Get environment configuration
config = get_environment_config('retail')
print(config.display_name)  # 'Retail'
print(config.data_key)      # 'orders'
```

### Registering a New Environment

To add a new environment (e.g., `banking`):

1. **Create the environment module** under `sigma/envs/banking/`:
   - `env.py` - Environment class
   - `data.py` - Data loader
   - `tools.py` - Tool implementations
   - `policy.md` - Agent policy
   - `rules.py` - Business rules
   - `db.json` - Database
   - `tasks.json` - Tasks

2. **Register in `sigma/env_registry.py`**:

```python
# Schema templates for persona generation
BANKING_USER_SCHEMA = """{ ... }"""
BANKING_ACCOUNT_SCHEMA = """{ ... }"""
BANKING_ADDITIONAL_CONTEXT = """Account types: checking, savings, ..."""

# Loaders
def _load_banking_env_class():
    from sigma.envs.banking import MockBankingDomainEnv
    return MockBankingDomainEnv

def _load_banking_data():
    from sigma.envs.banking.data import load_data
    return load_data()

def _load_banking_tasks(split: str):
    from sigma.envs.generic_env import load_env_tasks
    return load_env_tasks("banking", split)

# Register
register_environment(EnvironmentConfig(
    name="banking",
    display_name="Banking",
    description="Bank customer service - accounts, transfers, payments",
    user_schema=BANKING_USER_SCHEMA,
    order_schema=BANKING_ACCOUNT_SCHEMA,  # Generic - could be accounts, transactions
    order_key="accounts",  # Key in data dict
    additional_context=BANKING_ADDITIONAL_CONTEXT,
    env_class_loader=_load_banking_env_class,
    data_loader=_load_banking_data,
    task_splits=["test"],
    tasks_loader=_load_banking_tasks,
    scenario_examples=[
        "Customer wants to check account balance",
        "Customer wants to transfer money between accounts",
        "Customer reports unauthorized transaction",
    ],
))
```

3. **Generate template files** (optional helper):

```python
from sigma import create_environment_template
create_environment_template("banking", "sigma/envs")
```

The new environment will automatically appear in:
- `python -m sigma --help` (--env choices)
- `python -m sigma.persona_creator --help` (--env choices)
- `python -m sigma --list-tasks --env banking`

