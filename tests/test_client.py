"""
Tests for Chaukas client functionality with proto compliance.
"""

import asyncio
import os
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chaukas.sdk import ChaukasClient, ChaukasConfig, EventBuilder


@pytest.fixture
def config():
    """Create test configuration."""
    return ChaukasConfig(
        tenant_id="test-tenant",
        project_id="test-project",
        endpoint="https://test.chaukas.ai",
        api_key="test-key",
        batch_size=2,
        flush_interval=1.0,
    )


@pytest.fixture
def client(config):
    """Create test client with config."""
    return ChaukasClient(config=config)


@pytest.fixture
def event_builder():
    """Create event builder with test config."""
    # Set environment variables for config
    os.environ["CHAUKAS_TENANT_ID"] = "test-tenant"
    os.environ["CHAUKAS_PROJECT_ID"] = "test-project"
    os.environ["CHAUKAS_ENDPOINT"] = "https://test.chaukas.ai"
    os.environ["CHAUKAS_API_KEY"] = "test-key"
    return EventBuilder()


@pytest.fixture
def sample_event(event_builder):
    """Create a sample proto event."""
    return event_builder.create_agent_start(
        agent_id="test-agent",
        agent_name="Test Agent",
        role="assistant",
        instructions="Test instructions",
    )


@pytest.mark.asyncio
async def test_client_initialization(client):
    """Test client initialization with config."""
    assert client.endpoint == "https://test.chaukas.ai"
    assert client.api_key == "test-key"
    assert client.batch_size == 2
    assert client.flush_interval == 1.0
    assert len(client._events_queue) == 0
    assert not client._closed


@pytest.mark.asyncio
async def test_create_event_builder(client):
    """Test event builder creation from client."""
    with patch.dict(
        "os.environ",
        {
            "CHAUKAS_TENANT_ID": "test-tenant",
            "CHAUKAS_PROJECT_ID": "test-project",
            "CHAUKAS_ENDPOINT": "https://api.example.com",
            "CHAUKAS_API_KEY": "test-key",
        },
    ):
        builder = client.create_event_builder()
        assert builder is not None

    # Test creating an event with the builder
    event = builder.create_session_start()
    assert event.tenant_id == "test-tenant"
    assert event.project_id == "test-project"
    assert event.event_id is not None
    assert event.session_id is not None


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
        # Should trigger flush due to batch_size=2 being exceeded
        mock_flush.assert_called_once()


@pytest.mark.asyncio
async def test_flush_single_event(client, sample_event):
    """Test flushing a single event."""
    client._events_queue = [sample_event]

    with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        await client._flush_events()

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "/events" in call_args[0][0]
        assert client._events_queue == []


@pytest.mark.asyncio
async def test_flush_batch_events(client, event_builder):
    """Test flushing multiple events as batch."""
    events = [
        event_builder.create_agent_start("agent1", "Agent 1"),
        event_builder.create_agent_end("agent1", "Agent 1"),
        event_builder.create_session_end(),
    ]
    client._events_queue = events

    with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        await client._flush_events()

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "/events/batch" in call_args[0][0]
        assert client._events_queue == []


@pytest.mark.asyncio
async def test_flush_events_failure(client, sample_event):
    """Test event flushing failure and re-queuing."""
    original_events = [sample_event]
    client._events_queue = original_events.copy()

    with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = Exception("Network error")

        await client._flush_events()

        # Events should be re-queued on failure
        assert len(client._events_queue) == 1


@pytest.mark.asyncio
async def test_event_wrapper_support(client):
    """Test that client accepts EventWrapper."""
    wrapper = client.create_event_wrapper()
    wrapper.with_message("user", "Hello")

    with patch.object(client, "_flush_events", new_callable=AsyncMock):
        await client.send_event(wrapper)
        assert len(client._events_queue) == 1
        # Should have converted wrapper to proto
        assert hasattr(client._events_queue[0], "SerializeToString")


@pytest.mark.asyncio
async def test_client_close(client, sample_event):
    """Test client close behavior."""
    client._events_queue = [sample_event]

    with patch.object(client, "_flush_events", new_callable=AsyncMock) as mock_flush:
        with patch.object(
            client._client, "aclose", new_callable=AsyncMock
        ) as mock_aclose:
            await client.close()

            mock_flush.assert_called_once()
            mock_aclose.assert_called_once()
            assert client._closed


@pytest.mark.asyncio
async def test_client_context_manager():
    """Test client as async context manager."""
    config = ChaukasConfig(
        tenant_id="test", project_id="test", endpoint="https://test.com", api_key="key"
    )

    async with ChaukasClient(config=config) as client:
        assert not client._closed
        builder = client.create_event_builder()
        event = builder.create_session_start()
        await client.send_event(event)

    assert client._closed


@pytest.mark.asyncio
async def test_send_to_closed_client(client, sample_event):
    """Test sending events to closed client."""
    await client.close()

    # Should not raise but should warn
    await client.send_event(sample_event)
    assert len(client._events_queue) == 0  # Event not queued
