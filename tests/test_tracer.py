"""
Tests for Chaukas tracer functionality.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from chaukas.sdk.core.tracer import ChaukasTracer, Span
from chaukas.sdk.core.client import ChaukasClient


@pytest.fixture
def mock_client():
    client = MagicMock(spec=ChaukasClient)
    client.create_event = MagicMock()
    client.send_event = AsyncMock()
    return client


@pytest.fixture
def tracer(mock_client):
    return ChaukasTracer(client=mock_client, session_id="test-session")


def test_tracer_initialization(tracer, mock_client):
    """Test tracer initialization."""
    assert tracer.client == mock_client
    assert tracer.default_session_id == "test-session"


def test_start_span(tracer):
    """Test span creation."""
    # Clear any existing context from other tests
    from chaukas.sdk.core.tracer import _session_id
    _session_id.set(None)
    
    span = tracer.start_span("test-span")
    
    assert span.name == "test-span"
    assert span.tracer == tracer
    assert span.session_id == "test-session"
    assert span.span_id is not None
    assert span.trace_id is not None
    assert span.start_time is not None
    assert span.end_time is None
    assert span.status == "active"


def test_span_attributes():
    """Test span attribute management."""
    from chaukas.sdk.core.client import ChaukasClient
    
    client = MagicMock(spec=ChaukasClient)
    tracer = ChaukasTracer(client=client)
    span = tracer.start_span("test")
    
    span.set_attribute("key", "value")
    assert span.attributes["key"] == "value"
    
    span.add_event("test_event", {"data": "test"})
    assert len(span.events) == 1
    assert span.events[0]["event_type"] == "test_event"
    assert span.events[0]["data"] == {"data": "test"}


def test_span_status():
    """Test span status management."""
    from chaukas.sdk.core.client import ChaukasClient
    
    client = MagicMock(spec=ChaukasClient)
    tracer = ChaukasTracer(client=client)
    span = tracer.start_span("test")
    
    span.set_status("ok", "All good")
    assert span.status == "ok"
    assert span.attributes["status_description"] == "All good"


@pytest.mark.asyncio
async def test_span_context_manager(tracer, mock_client):
    """Test span as context manager."""
    mock_event = MagicMock()
    mock_client.create_event.return_value = mock_event
    
    with tracer.start_span("test-span") as span:
        assert span.start_time is not None
        assert span.end_time is None
        span.set_attribute("test", "value")
    
    # Span should be finished after context exit
    assert span.end_time is not None
    assert span.status == "ok"
    
    # Event should be created and sent
    mock_client.create_event.assert_called_once()
    mock_client.send_event.assert_called_once()


@pytest.mark.asyncio
async def test_span_context_manager_with_exception(tracer, mock_client):
    """Test span context manager with exception."""
    mock_event = MagicMock()
    mock_client.create_event.return_value = mock_event
    
    try:
        with tracer.start_span("test-span") as span:
            raise ValueError("Test error")
    except ValueError:
        pass
    
    # Span should be finished with error status
    assert span.end_time is not None
    assert span.status == "error"
    assert span.attributes["error"] == "Test error"
    assert span.attributes["error_type"] == "ValueError"


@pytest.mark.asyncio
async def test_send_event(tracer, mock_client):
    """Test sending standalone events."""
    mock_event = MagicMock()
    mock_client.create_event.return_value = mock_event
    
    await tracer.send_event(
        event_type="test.event",
        source="test",
        data={"key": "value"},
        metadata={"meta": "data"},
    )
    
    mock_client.create_event.assert_called_once()
    mock_client.send_event.assert_called_once_with(mock_event)


def test_get_current_span_context(tracer):
    """Test getting current span context."""
    context = tracer.get_current_span_context()
    
    assert "session_id" in context
    assert "trace_id" in context
    assert "span_id" in context
    assert "parent_span_id" in context