# Copyright Amity
"""
Tests for environment registry.
"""

import pytest
from sigma.env_registry import list_environments, get_environment_config


class TestEnvRegistry:
    """Test suite for environment registry."""

    def test_list_environments(self):
        """Test that environments are listed."""
        envs = list_environments()
        assert isinstance(envs, list)
        assert len(envs) > 0
        assert "retail" in envs

    def test_get_environment_config(self):
        """Test getting environment configuration."""
        config = get_environment_config("retail")
        assert config is not None
        assert config.name == "retail"
        assert config.display_name is not None
