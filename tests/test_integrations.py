"""
Tests for SDK integrations.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from chaukas.utils.monkey_patch import MonkeyPatcher
from chaukas.core.tracer import ChaukasTracer
from chaukas.core.client import ChaukasClient


@pytest.fixture
def mock_client():
    return MagicMock(spec=ChaukasClient)


@pytest.fixture
def mock_tracer(mock_client):
    return MagicMock(spec=ChaukasTracer)


@pytest.fixture
def patcher(mock_tracer):
    return MonkeyPatcher(tracer=mock_tracer, config={})


def test_monkey_patcher_initialization(patcher, mock_tracer):
    """Test MonkeyPatcher initialization."""
    assert patcher.tracer == mock_tracer
    assert patcher.config == {}
    assert patcher.patches == []
    assert patcher._auto_detect is True


def test_patch_detection():
    """Test SDK detection logic."""
    with patch("importlib.import_module") as mock_import:
        # Mock successful import of OpenAI agents
        mock_import.return_value = MagicMock()
        
        tracer = MagicMock(spec=ChaukasTracer)
        patcher = MonkeyPatcher(tracer=tracer, config={})
        
        with patch.object(patcher, "_patch_openai_agents") as mock_patch:
            patcher._detect_and_patch_sdks()
            mock_patch.assert_called_once()


def test_patch_detection_import_error():
    """Test SDK detection with import errors."""
    with patch("importlib.import_module") as mock_import:
        # Mock ImportError for missing SDK
        mock_import.side_effect = ImportError("Module not found")
        
        tracer = MagicMock(spec=ChaukasTracer)
        patcher = MonkeyPatcher(tracer=tracer, config={})
        
        with patch.object(patcher, "_patch_openai_agents") as mock_patch:
            patcher._detect_and_patch_sdks()
            mock_patch.assert_not_called()


@pytest.mark.asyncio
async def test_openai_agents_wrapper():
    """Test OpenAI Agents wrapper functionality."""
    from chaukas.integrations.openai_agents import OpenAIAgentsWrapper
    
    mock_tracer = MagicMock(spec=ChaukasTracer)
    mock_span = MagicMock()
    mock_tracer.start_span.return_value.__enter__ = MagicMock(return_value=mock_span)
    mock_tracer.start_span.return_value.__exit__ = MagicMock(return_value=None)
    mock_tracer.send_event = AsyncMock()
    
    wrapper = OpenAIAgentsWrapper(mock_tracer)
    
    # Mock agent instance
    mock_agent = MagicMock()
    mock_agent.name = "test-agent"
    mock_agent.id = "agent-123"
    mock_agent.instructions = "Test instructions"
    mock_agent.model = "gpt-4"
    mock_agent.tools = []
    
    # Mock original method
    async def mock_agent_run(*args, **kwargs):
        return MagicMock(messages=[])
    
    # Wrap the method
    wrapped_method = wrapper.wrap_agent_run(mock_agent_run, mock_agent, [], {})
    
    # Execute wrapped method
    result = await wrapped_method()
    
    # Verify tracer interactions
    mock_tracer.start_span.assert_called_once_with("openai_agent.run")
    assert mock_tracer.send_event.call_count >= 2  # Start and end events


@pytest.mark.asyncio
async def test_crewai_wrapper():
    """Test CrewAI wrapper functionality."""
    from chaukas.integrations.crewai import CrewAIWrapper
    
    mock_tracer = MagicMock(spec=ChaukasTracer)
    mock_span = MagicMock()
    mock_tracer.start_span.return_value.__enter__ = MagicMock(return_value=mock_span)
    mock_tracer.start_span.return_value.__exit__ = MagicMock(return_value=None)
    mock_tracer.send_event = AsyncMock()
    
    wrapper = CrewAIWrapper(mock_tracer)
    
    # Mock crew instance
    mock_crew = MagicMock()
    mock_crew.agents = []
    mock_crew.tasks = []
    mock_crew.process = "sequential"
    
    # Mock original method
    async def mock_crew_kickoff(*args, **kwargs):
        return "Task completed"
    
    # Wrap the method
    wrapped_method = wrapper.wrap_crew_kickoff(mock_crew_kickoff, mock_crew, [], {})
    
    # Execute wrapped method
    result = await wrapped_method()
    
    # Verify tracer interactions
    mock_tracer.start_span.assert_called_once_with("crewai.crew.kickoff")
    assert mock_tracer.send_event.call_count >= 2  # Start and end events
    assert result == "Task completed"


def test_enabled_integrations_config():
    """Test configuration for enabled integrations."""
    tracer = MagicMock(spec=ChaukasTracer)
    config = {
        "auto_detect": False,
        "enabled_integrations": ["openai_agents", "crewai"]
    }
    
    patcher = MonkeyPatcher(tracer=tracer, config=config)
    
    assert patcher._auto_detect is False
    assert patcher._enabled_integrations == ["openai_agents", "crewai"]