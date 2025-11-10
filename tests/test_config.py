"""
Tests for Chaukas configuration management.
"""

import os
from unittest.mock import patch

import pytest

from chaukas.sdk.core.config import ChaukasConfig, get_config, reset_config, set_config


def test_config_from_dict():
    """Test creating config from dictionary."""
    config_dict = {
        "tenant_id": "test-tenant",
        "project_id": "test-project",
        "endpoint": "https://api.chaukas.ai",
        "api_key": "test-key",
        "batch_size": 50,
        "flush_interval": 10.0,
        "timeout": 60.0,
        "branch": "main",
        "tags": ["tag1", "tag2"],
    }

    config = ChaukasConfig.from_dict(config_dict)

    assert config.tenant_id == "test-tenant"
    assert config.project_id == "test-project"
    assert config.endpoint == "https://api.chaukas.ai"
    assert config.api_key == "test-key"
    assert config.batch_size == 50
    assert config.flush_interval == 10.0
    assert config.timeout == 60.0
    assert config.branch == "main"
    assert config.tags == ["tag1", "tag2"]


def test_config_from_env():
    """Test loading config from environment variables."""
    env_vars = {
        "CHAUKAS_TENANT_ID": "env-tenant",
        "CHAUKAS_PROJECT_ID": "env-project",
        "CHAUKAS_ENDPOINT": "https://env.chaukas.ai",
        "CHAUKAS_API_KEY": "env-key",
        "CHAUKAS_BATCH_SIZE": "200",
        "CHAUKAS_FLUSH_INTERVAL": "15.5",
        "CHAUKAS_TIMEOUT": "45.0",
        "CHAUKAS_BRANCH": "develop",
        "CHAUKAS_TAGS": "env,test,v1",
    }

    with patch.dict(os.environ, env_vars, clear=False):
        config = ChaukasConfig.from_env()

    assert config.tenant_id == "env-tenant"
    assert config.project_id == "env-project"
    assert config.endpoint == "https://env.chaukas.ai"
    assert config.api_key == "env-key"
    assert config.batch_size == 200
    assert config.flush_interval == 15.5
    assert config.timeout == 45.0
    assert config.branch == "develop"
    assert config.tags == ["env", "test", "v1"]


def test_config_from_env_missing_required():
    """Test that missing required env vars raise errors."""
    # Clear all CHAUKAS env vars
    env_vars = {k: v for k, v in os.environ.items() if not k.startswith("CHAUKAS_")}

    with patch.dict(os.environ, env_vars, clear=True):
        with pytest.raises(ValueError, match="CHAUKAS_TENANT_ID"):
            ChaukasConfig.from_env()

    # Add tenant but missing project
    env_vars["CHAUKAS_TENANT_ID"] = "test"
    with patch.dict(os.environ, env_vars, clear=True):
        with pytest.raises(ValueError, match="CHAUKAS_PROJECT_ID"):
            ChaukasConfig.from_env()

    # Add project but missing endpoint
    env_vars["CHAUKAS_PROJECT_ID"] = "test"
    with patch.dict(os.environ, env_vars, clear=True):
        with pytest.raises(ValueError, match="CHAUKAS_ENDPOINT"):
            ChaukasConfig.from_env()

    # Add endpoint but missing API key
    env_vars["CHAUKAS_ENDPOINT"] = "https://test.com"
    with patch.dict(os.environ, env_vars, clear=True):
        with pytest.raises(ValueError, match="CHAUKAS_API_KEY"):
            ChaukasConfig.from_env()


def test_config_from_env_defaults():
    """Test that optional env vars use defaults."""
    env_vars = {
        "CHAUKAS_TENANT_ID": "test",
        "CHAUKAS_PROJECT_ID": "test",
        "CHAUKAS_ENDPOINT": "https://test.com",
        "CHAUKAS_API_KEY": "key",
    }

    with patch.dict(os.environ, env_vars, clear=False):
        config = ChaukasConfig.from_env()

    # Check defaults
    assert config.batch_size == 20  # Default changed to 20 to prevent high memory usage
    assert config.flush_interval == 5.0
    assert config.timeout == 30.0
    assert config.branch is None
    assert config.tags is None


def test_config_to_dict():
    """Test converting config to dictionary."""
    config = ChaukasConfig(
        tenant_id="test",
        project_id="project",
        endpoint="https://api.com",
        api_key="key",
        branch="main",
        tags=["tag1"],
    )

    config_dict = config.to_dict()

    assert config_dict["tenant_id"] == "test"
    assert config_dict["project_id"] == "project"
    assert config_dict["endpoint"] == "https://api.com"
    assert config_dict["api_key"] == "key"
    assert config_dict["batch_size"] == 20  # Default is 20
    assert config_dict["flush_interval"] == 5.0
    assert config_dict["timeout"] == 30.0
    assert config_dict["branch"] == "main"
    assert config_dict["tags"] == ["tag1"]


def test_global_config_management():
    """Test global config get/set/reset functions."""
    # Reset to ensure clean state
    reset_config()

    # First call should load from env
    env_vars = {
        "CHAUKAS_TENANT_ID": "global-tenant",
        "CHAUKAS_PROJECT_ID": "global-project",
        "CHAUKAS_ENDPOINT": "https://global.com",
        "CHAUKAS_API_KEY": "global-key",
    }

    with patch.dict(os.environ, env_vars, clear=False):
        config1 = get_config()
        assert config1.tenant_id == "global-tenant"

        # Second call should return same instance
        config2 = get_config()
        assert config1 is config2

    # Set new config
    new_config = ChaukasConfig(
        tenant_id="new", project_id="new", endpoint="https://new.com", api_key="new-key"
    )
    set_config(new_config)

    config3 = get_config()
    assert config3.tenant_id == "new"
    assert config3 is new_config

    # Reset config
    reset_config()

    # After reset, should load from env again
    with patch.dict(os.environ, env_vars, clear=False):
        config4 = get_config()
        assert config4.tenant_id == "global-tenant"
        assert config4 is not config1  # New instance


def test_config_from_dict_missing_required():
    """Test that missing required fields in dict raise errors."""
    with pytest.raises(ValueError, match="tenant_id"):
        ChaukasConfig.from_dict({})

    with pytest.raises(ValueError, match="project_id"):
        ChaukasConfig.from_dict({"tenant_id": "test"})

    with pytest.raises(ValueError, match="endpoint"):
        ChaukasConfig.from_dict({"tenant_id": "test", "project_id": "test"})

    with pytest.raises(ValueError, match="api_key"):
        ChaukasConfig.from_dict(
            {"tenant_id": "test", "project_id": "test", "endpoint": "https://test.com"}
        )
