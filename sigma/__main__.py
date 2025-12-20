# Copyright Amity
"""
Entry point for running sigma as a module.

Usage:
    python -m sigma --env retail --user-model gpt-4o --user-provider openai
"""

from sigma.cli_simulator import main

if __name__ == "__main__":
    main()
