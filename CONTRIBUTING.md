# Contributing to Sigma

Thank you for your interest in contributing to Sigma! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Submitting Changes](#submitting-changes)
- [Style Guidelines](#style-guidelines)
- [Adding New Environments](#adding-new-environments)

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone. Please be kind and constructive in all interactions.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/sigma.git
   cd sigma
   ```
3. **Add the upstream remote**:
   ```bash
   git remote add upstream https://github.com/AmityCo/sigma.git
   ```

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Node.js 18+ (for web UI development)
- An OpenAI API key (or other LLM provider)

### Installation

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install in development mode with all dependencies**:
   ```bash
   pip install -e ".[all]"
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **For web UI development**:
   ```bash
   cd sigma/static/react-app
   npm install
   npm run build
   ```

### Running Tests

```bash
pytest
```

With coverage:
```bash
pytest --cov=sigma --cov-report=html
```

## Making Changes

### Branch Naming

- `feature/` - New features (e.g., `feature/add-banking-env`)
- `fix/` - Bug fixes (e.g., `fix/tool-parameter-validation`)
- `docs/` - Documentation updates (e.g., `docs/improve-api-docs`)
- `refactor/` - Code refactoring (e.g., `refactor/simplify-env-registry`)

### Workflow

1. **Create a branch** from `main`:
   ```bash
   git checkout main
   git pull upstream main
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the style guidelines

3. **Test your changes**:
   ```bash
   pytest
   ```

4. **Commit with clear messages**:
   ```bash
   git commit -m "feat: add support for custom reward functions"
   ```

## Submitting Changes

### Pull Request Process

1. **Update your fork**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Push your branch**:
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create a Pull Request** on GitHub with:
   - Clear title describing the change
   - Description of what was changed and why
   - Any relevant issue numbers (e.g., "Fixes #123")
   - Screenshots for UI changes

### PR Checklist

- [ ] Code follows the project style guidelines
- [ ] Tests added/updated for new functionality
- [ ] Documentation updated if needed
- [ ] All tests pass
- [ ] No merge conflicts

## Style Guidelines

### Python

- Follow [PEP 8](https://pep8.org/) style guide
- Use type hints for function signatures
- Maximum line length: 100 characters
- Use docstrings for public functions and classes

```python
def process_action(
    action: Action,
    context: Dict[str, Any],
) -> ActionResult:
    """
    Process an agent action and return the result.
    
    Args:
        action: The action to process
        context: Additional context for processing
        
    Returns:
        ActionResult containing success status and observation
    """
    ...
```

### Formatting

We use `black` for code formatting and `ruff` for linting:

```bash
black sigma/
ruff check sigma/ --fix
```

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

## Adding New Environments

To add a new environment (e.g., `banking`):

1. **Create the data directory**:
   ```
   data/envs/banking/
   ├── db.json           # Database with sample data
   ├── tasks.json        # Task definitions
   ├── policy.md         # Agent policy document
   ├── user_guidelines.md # User simulation guidelines
   └── tools.py          # Tool implementations
   ```

2. **Implement tools** in `tools.py`:
   ```python
   from sigma.envs.tool import Tool
   
   class GetAccountBalance(Tool):
       @staticmethod
       def invoke(data: Dict[str, Any], account_id: str) -> str:
           # Implementation
           ...
       
       @staticmethod
       def get_info() -> Dict[str, Any]:
           return {
               "type": "function",
               "function": {
                   "name": "get_account_balance",
                   "description": "Get the balance of a bank account",
                   "parameters": {...}
               }
           }
   ```

3. **Register the environment** in `sigma/env_registry.py`

4. **Add tests** for your environment

5. **Update documentation** in README.md

## Questions?

If you have questions or need help:

1. Check existing [issues](https://github.com/AmityCo/sigma/issues)
2. Open a new issue with the `question` label
3. Join discussions in the repository

Thank you for contributing! 🎉
