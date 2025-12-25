# Copyright Amity
"""
Tests for SimulatorCore.
"""

import pytest
from sigma.simulator_core import SimulatorCore, ActionType


class TestSimulatorCore:
    """Test suite for SimulatorCore class."""

    def test_action_type_enum(self):
        """Test ActionType enum values."""
        assert ActionType.RESPOND.value == "respond"
        assert ActionType.TOOL_CALL.value == "tool_call"

    # TODO: Add more tests for SimulatorCore
    # - test_init
    # - test_start
    # - test_respond_to_user
    # - test_call_tool
    # - test_generate_response
