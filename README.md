# Sigma

**Interactive Scenario Simulator for Agent Training & Evaluation**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Sigma is an interactive simulation tool for training and evaluating AI agents. A human acts as the **agent** while an LLM simulates realistic **users** based on configurable personas and scenarios.

Built for [tau-bench](https://github.com/sierra-research/tau-bench) environments, Sigma helps you:
- **Practice** agent responses in realistic customer service scenarios
- **Collect** high-quality training data with human demonstrations
- **Evaluate** agent performance with automatic reward scoring

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎮 **Multiple Interfaces** | CLI for quick testing, Web UI for rich interactions |
| 🤖 **LLM User Simulation** | Realistic user behavior powered by GPT-4, Claude, or any LiteLLM-supported model |
| 🎭 **Scenario Generation** | Auto-generate personas with realistic data that matches your environment |
| 🔧 **Tool Calling** | Execute environment tools with guided parameter input |
| 📋 **Policy Reference** | View agent guidelines and policies during simulation |
| 📊 **Reward Evaluation** | Get immediate feedback comparing actions to ground truth |
| 💾 **Trajectory Storage** | Save sessions for training data creation (SFT, DPO, GRPO) |
| 🌐 **REST API** | Programmatic access for custom integrations |

## 📦 Installation

### From PyPI (recommended)

```bash
pip install sigma-simulator
```

### From Source

```bash
git clone https://github.com/AmityCo/sigma.git
cd sigma
pip install -e ".[all]"
```

### Dependencies

```bash
# Core only (CLI)
pip install -e .

# With web UI
pip install -e ".[web]"

# With storage backends
pip install -e ".[storage]"

# Everything including dev tools
pip install -e ".[all]"
```

## 🚀 Quick Start

### 1. Set up your API key

```bash
export OPENAI_API_KEY="sk-..."
```

Or copy `.env.example` to `.env` and fill in your keys.

### 2. Run the simulator

**CLI Interface:**
```bash
python -m sigma --env retail
```

**Web Interface:**
```bash
python -m sigma.api_server --port 8000
# Open http://localhost:8000
```

### 3. Interact with the simulated user

The LLM will simulate a customer based on the selected task. You respond as the agent, calling tools and sending messages until the task is complete.

## 📖 Documentation

### Available Environments

| Environment | Description | Tools |
|-------------|-------------|-------|
| `retail` | Online retail customer service | Orders, returns, exchanges, user management |
| `airline` | Airline customer service | Reservations, flights, baggage, refunds |

### CLI Options

```bash
python -m sigma --help

Options:
  --env {retail,airline}     Environment to use (default: retail)
  --user-model MODEL         LLM model for user simulation (default: gpt-4o)
  --user-provider PROVIDER   LLM provider (default: openai)
  --task-index INDEX         Specific task to run (default: random)
  --persona TEXT             Custom persona description
  --persona-file PATH        JSON file with persona and data
  --list-tasks               List available tasks and exit
```

### Web API

Start the server:
```bash
python -m sigma.api_server --host 0.0.0.0 --port 8000
```

Key endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/environments` | List available environments |
| `POST` | `/sessions` | Create a new session |
| `POST` | `/sessions/{id}/start` | Start simulation |
| `POST` | `/sessions/{id}/respond` | Send agent response |
| `POST` | `/sessions/{id}/tool` | Call a tool |
| `GET` | `/sessions/{id}/result` | Get final reward |

### Programmatic Usage

```python
from sigma import SimulatorCore

sim = SimulatorCore(
    env_name="retail",
    user_model="gpt-4o",
    user_provider="openai",
)

# Start simulation
initial_message = sim.start()
print(f"User: {initial_message}")

# Respond as agent
result = sim.respond_to_user("Hello! How can I help you today?")
print(f"User: {result.observation}")

# Call a tool
result = sim.call_tool("get_user_details", {"user_id": "john_doe_123"})
print(f"Tool result: {result.observation}")

# Check completion
if sim.is_done:
    final = sim.get_final_result()
    print(f"Reward: {final['reward']}")
```

### Creating Custom Personas

Generate a persona with realistic data:

```bash
python -m sigma.persona_creator --env retail --model gpt-4o
```

Then use it in simulation:

```bash
python -m sigma --env retail --persona-file sigma/personas/my_persona.json
```

### Adding New Environments

Create a new environment directory under `data/envs/`:

```
data/envs/your_env/
├── db.json           # Database with sample data
├── tasks.json        # Task definitions
├── policy.md         # Agent policy document
├── user_guidelines.md # User simulation guidelines
└── tools.py          # Tool implementations
```

See [data/envs/README.md](data/envs/README.md) for detailed documentation.

## 🗂️ Project Structure

```
sigma/
├── sigma/                  # Main package
│   ├── simulator_core.py   # Core simulation engine
│   ├── cli_simulator.py    # CLI interface
│   ├── api_server.py       # FastAPI web server
│   ├── env_registry.py     # Environment registry
│   ├── persona_creator.py  # Persona generation
│   ├── trajectory.py       # Trajectory management
│   ├── envs/               # Environment implementations
│   ├── exports/            # Training data exporters
│   └── static/             # Web UI assets
├── data/
│   └── envs/               # Environment data files
├── tests/                  # Test suite
└── scripts/                # Utility scripts
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `ANTHROPIC_API_KEY` | Anthropic API key | Optional |
| `AZURE_STORAGE_CONNECTION_STRING` | Azure Blob Storage connection string | Optional (falls back to local) |
| `DEBUG_SIMULATOR` | Enable debug logging | `false` |

See [.env.example](.env.example) for all options.

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
git clone https://github.com/AmityCo/sigma.git
cd sigma
pip install -e ".[all]"
pytest
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built for [tau-bench](https://github.com/sierra-research/tau-bench) benchmark environments
- Powered by [LiteLLM](https://github.com/BerriAI/litellm) for multi-provider LLM support
- Web UI built with [React](https://react.dev/) and [Vite](https://vitejs.dev/)

