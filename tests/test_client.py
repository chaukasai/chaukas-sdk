"""
Tests for Chaukas client functionality.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime, timezone

from chaukas.core.client import ChaukasClient, ChaukasEvent


@pytest.fixture
def client():
    return ChaukasClient(
        endpoint="https://test.chaukas.com",
        api_key="test-key",
        batch_size=2,
        flush_interval=1.0,
    )


@pytest.fixture
def sample_event():
    return ChaukasEvent(
        event_id="test-event-id",
        session_id="test-session",
        trace_id="test-trace",
        span_id="test-span",
        parent_span_id="test-parent-span",
        timestamp=datetime.now(timezone.utc),
        event_type="test.event",
        source="test",
        data={"key": "value"},
        metadata={"meta": "data"},
    )


@pytest.mark.asyncio
async def test_client_initialization(client):
    """Test client initialization."""
    assert client.endpoint == "https://test.chaukas.com"
    assert client.api_key == "test-key"
    assert client.batch_size == 2
    assert client._events_queue == []
    assert not client._closed


@pytest.mark.asyncio
async def test_create_event(client):
    """Test event creation."""
    event = client.create_event(
        event_type="test.event",
        source="test",
        data={"key": "value"},
        session_id="session-123",
        trace_id="trace-123",
        span_id="span-123",
        parent_span_id="parent-123",
        metadata={"meta": "data"},
    )
    
    assert event.event_type == "test.event"
    assert event.source == "test"
    assert event.data == {"key": "value"}
    assert event.session_id == "session-123"
    assert event.trace_id == "trace-123"
    assert event.span_id == "span-123"
    assert event.parent_span_id == "parent-123"
    assert event.metadata == {"meta": "data"}
    assert event.event_id is not None
    assert event.timestamp is not None


@pytest.mark.asyncio
async def test_send_event_queuing(client, sample_event):
    """Test event queuing behavior."""
    with patch.object(client, "_flush_events", new_callable=AsyncMock) as mock_flush:
        # Send first event (should not trigger flush)
        await client.send_event(sample_event)
        assert len(client._events_queue) == 1
        mock_flush.assert_not_called()
        
        # Send second event (should trigger flush due to batch_size=2)
        await client.send_event(sample_event)
        mock_flush.assert_called_once()


@pytest.mark.asyncio
async def test_send_events_batch(client, sample_event):
    """Test sending multiple events."""
    events = [sample_event, sample_event, sample_event]
    
    with patch.object(client, "_flush_events", new_callable=AsyncMock) as mock_flush:
        await client.send_events(events)
        # Should trigger flush due to batch_size=2
        mock_flush.assert_called_once()
        assert len(client._events_queue) == 1  # One event remaining after flush


@pytest.mark.asyncio
async def test_flush_events_success(client, sample_event):
    """Test successful event flushing."""
    client._events_queue = [sample_event, sample_event]
    
    with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        await client._flush_events()
        
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == "https://test.chaukas.com/events"
        assert "events" in call_args[1]["json"]
        assert len(call_args[1]["json"]["events"]) == 2
        assert client._events_queue == []


@pytest.mark.asyncio
async def test_flush_events_failure(client, sample_event):
    """Test event flushing failure and re-queuing."""
    client._events_queue = [sample_event]
    
    with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = Exception("Network error")
        
        await client._flush_events()
        
        # Events should be re-queued on failure
        assert len(client._events_queue) == 1


@pytest.mark.asyncio
async def test_client_close(client, sample_event):
    """Test client close behavior."""
    client._events_queue = [sample_event]
    
    with patch.object(client, "_flush_events", new_callable=AsyncMock) as mock_flush:
        with patch.object(client._client, "aclose", new_callable=AsyncMock) as mock_aclose:
            await client.close()
            
            mock_flush.assert_called_once()
            mock_aclose.assert_called_once()
            assert client._closed


@pytest.mark.asyncio
async def test_client_context_manager(sample_event):
    """Test client as async context manager."""
    async with ChaukasClient("https://test.com", "key") as client:
        assert not client._closed
        await client.send_event(sample_event)
    
    assert client._closed