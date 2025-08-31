"""
Tests for EventBuilder with all 20 proto event types.
"""

import pytest
import os
from datetime import datetime

from chaukas.core.event_builder import EventBuilder
from chaukas.core.config import ChaukasConfig, set_config


@pytest.fixture(autouse=True)
def setup_config():
    """Set up test configuration."""
    config = ChaukasConfig(
        tenant_id="test-tenant",
        project_id="test-project",
        endpoint="https://test.chaukas.com",
        api_key="test-key"
    )
    set_config(config)
    yield
    # Reset config after test
    from chaukas.core.config import reset_config
    reset_config()


@pytest.fixture
def builder():
    """Create EventBuilder instance."""
    return EventBuilder()


def test_create_session_start(builder):
    """Test SESSION_START event creation."""
    event = builder.create_session_start(
        session_id="test-session",
        metadata={"key": "value"}
    )
    
    assert event.tenant_id == "test-tenant"
    assert event.project_id == "test-project"
    assert event.session_id == "test-session"
    assert event.event_id is not None
    assert event.trace_id is not None
    assert event.span_id is not None


def test_create_session_end(builder):
    """Test SESSION_END event creation."""
    event = builder.create_session_end(
        session_id="test-session",
        metadata={"result": "success"}
    )
    
    assert event.session_id == "test-session"
    assert event.type == 2  # EVENT_TYPE_SESSION_END


def test_create_agent_start(builder):
    """Test AGENT_START event creation."""
    event = builder.create_agent_start(
        agent_id="agent-123",
        agent_name="Test Agent",
        role="assistant",
        instructions="Help users",
        tools=["tool1", "tool2"]
    )
    
    assert event.agent_id == "agent-123"
    assert event.agent_name == "Test Agent"
    assert event.type == 3  # EVENT_TYPE_AGENT_START


def test_create_agent_end(builder):
    """Test AGENT_END event creation."""
    event = builder.create_agent_end(
        agent_id="agent-123",
        agent_name="Test Agent"
    )
    
    assert event.agent_id == "agent-123"
    assert event.type == 4  # EVENT_TYPE_AGENT_END
    assert event.status == 3  # EVENT_STATUS_COMPLETED


def test_create_model_invocation_start(builder):
    """Test MODEL_INVOCATION_START event creation."""
    event = builder.create_model_invocation_start(
        provider="openai",
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0.7,
        max_tokens=100
    )
    
    assert event.llm_invocation.provider == "openai"
    assert event.llm_invocation.model == "gpt-4"
    assert event.llm_invocation.temperature == 0.7
    assert event.llm_invocation.max_tokens == 100
    assert event.type == 5  # EVENT_TYPE_MODEL_INVOCATION_START


def test_create_model_invocation_end(builder):
    """Test MODEL_INVOCATION_END event creation."""
    event = builder.create_model_invocation_end(
        provider="openai",
        model="gpt-4",
        response_content="Hello back",
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15
    )
    
    assert event.llm_invocation.provider == "openai"
    assert event.llm_invocation.prompt_tokens == 10
    assert event.llm_invocation.completion_tokens == 5
    assert event.llm_invocation.total_tokens == 15
    assert event.type == 6  # EVENT_TYPE_MODEL_INVOCATION_END


def test_create_tool_call_start(builder):
    """Test TOOL_CALL_START event creation."""
    event = builder.create_tool_call_start(
        tool_name="calculator",
        arguments={"operation": "add", "a": 1, "b": 2},
        call_id="call-123"
    )
    
    assert event.tool_call.name == "calculator"
    assert event.tool_call.id == "call-123"
    assert event.type == 7  # EVENT_TYPE_TOOL_CALL_START


def test_create_tool_call_end(builder):
    """Test TOOL_CALL_END event creation."""
    event = builder.create_tool_call_end(
        tool_name="calculator",
        call_id="call-123",
        output={"result": 3},
        execution_time_ms=50.5
    )
    
    assert event.tool_response.tool_name == "calculator"
    assert event.tool_response.tool_call_id == "call-123"
    assert event.tool_response.execution_time_ms == 50.5
    assert event.type == 8  # EVENT_TYPE_TOOL_CALL_END


def test_create_input_received(builder):
    """Test INPUT_RECEIVED event creation."""
    event = builder.create_input_received(
        content="User input text",
        source="chat"
    )
    
    assert event.message.role == "user"
    assert event.message.text == "User input text"
    assert event.type == 9  # EVENT_TYPE_INPUT_RECEIVED


def test_create_output_emitted(builder):
    """Test OUTPUT_EMITTED event creation."""
    event = builder.create_output_emitted(
        content="Agent response",
        target="user"
    )
    
    assert event.message.role == "assistant"
    assert event.message.text == "Agent response"
    assert event.type == 10  # EVENT_TYPE_OUTPUT_EMITTED


def test_create_error(builder):
    """Test ERROR event creation."""
    event = builder.create_error(
        error_message="Something went wrong",
        error_code="ERR_001",
        stack_trace="Traceback...",
        recoverable=True,
        recovery_action="Retry"
    )
    
    assert event.error.error_message == "Something went wrong"
    assert event.error.error_code == "ERR_001"
    assert event.error.recoverable is True
    assert event.error.recovery_action == "Retry"
    assert event.type == 11  # EVENT_TYPE_ERROR
    assert event.severity == 4  # SEVERITY_ERROR


def test_create_agent_handoff(builder):
    """Test AGENT_HANDOFF event creation."""
    event = builder.create_agent_handoff(
        from_agent_id="agent-1",
        from_agent_name="Agent 1",
        to_agent_id="agent-2",
        to_agent_name="Agent 2",
        reason="Task delegation"
    )
    
    assert event.agent_handoff.from_agent_id == "agent-1"
    assert event.agent_handoff.to_agent_id == "agent-2"
    assert event.agent_handoff.reason == "Task delegation"
    assert event.type == 14  # EVENT_TYPE_AGENT_HANDOFF


def test_create_state_update(builder):
    """Test STATE_UPDATE event creation."""
    event = builder.create_state_update(
        state_data={"step": 1, "status": "processing"}
    )
    
    # Check that state_update struct was populated
    assert "step" in event.state_update
    assert "status" in event.state_update
    assert event.type == 19  # EVENT_TYPE_STATE_UPDATE


def test_distributed_tracing_hierarchy(builder):
    """Test that distributed tracing hierarchy is maintained."""
    # Set context
    from chaukas.core.tracer import _session_id, _trace_id, _span_id, _parent_span_id
    
    _session_id.set("session-123")
    _trace_id.set("trace-456")
    _span_id.set("span-789")
    _parent_span_id.set("parent-000")
    
    event = builder.create_agent_start("agent", "Agent")
    
    assert event.session_id == "session-123"
    assert event.trace_id == "trace-456"
    assert event.span_id == "span-789"
    assert event.parent_span_id == "parent-000"


def test_uuid7_event_id_generation(builder):
    """Test that event IDs are UUID7 format."""
    event1 = builder.create_session_start()
    event2 = builder.create_session_start()
    
    # UUID7 characteristics
    assert len(event1.event_id) == 36  # Standard UUID string length
    assert event1.event_id != event2.event_id  # Unique
    assert "-" in event1.event_id  # UUID format