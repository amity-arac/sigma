#!/usr/bin/env python3
# Copyright Amity
"""
Standalone entry point for sigma CLI simulator.
Can be run from any directory.

Usage:
    ./run_simulator.py --env retail --user-model gpt-4o --user-provider openai
"""

import sys
import os

# Add the parent directory to the path so imports work
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from sigma.cli_simulator import main

if __name__ == "__main__":
    main()
