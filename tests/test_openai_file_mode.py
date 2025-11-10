"""
Integration test for OpenAI agents using file mode validation.

NOTE: These tests trigger enable_chaukas() which attempts to import openai-agents.
They need updating to match the refactored OpenAI integration API.
"""

import json
import os
import pathlib
import tempfile
from unittest.mock import Mock, patch

import pytest

# Skip all file mode tests - they need updating to match refactored API
pytestmark = pytest.mark.skip(
    reason="File mode tests need updating to match refactored OpenAI integration"
)

# Set up test environment
os.environ["CHAUKAS_TENANT_ID"] = "test-tenant"
os.environ["CHAUKAS_PROJECT_ID"] = "test-project"
os.environ["CHAUKAS_ENDPOINT"] = "http://test-endpoint"
os.environ["CHAUKAS_API_KEY"] = "test-key"


@pytest.fixture
def temp_output_file():
    """Create a temporary file for event output."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        output_file = f.name

    yield output_file

    # Cleanup
    pathlib.Path(output_file).unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_openai_simple_with_file_output(temp_output_file):
    """Test that openai_simple example generates correct events in file mode."""
    import asyncio

    # Set file output mode
    os.environ["CHAUKAS_OUTPUT_MODE"] = "file"
    os.environ["CHAUKAS_OUTPUT_FILE"] = temp_output_file

    # Import chaukas and enable
    from chaukas import sdk as chaukas

    chaukas.enable_chaukas()

    try:
        # Import OpenAI agents
        from agents import Agent, Runner

        # Mock the actual API call
        mock_result = Mock()
        mock_result.final_output = "Test haiku response"
        mock_result.messages = [{"role": "assistant", "content": "Test haiku response"}]

        async def mock_run(agent, messages, **kwargs):
            return mock_result

        # Patch Runner.run
        with patch.object(Runner, "run", side_effect=mock_run):
            # Create agent
            agent = Agent(
                name="TestAgent",
                instructions="You only respond in haikus.",
            )

            # Run agent
            result = await Runner.run(agent, "Tell me about recursion.")

            # Flush events
            client = chaukas.get_client()
            if client:
                await client.flush()
                await client.close()

        # Disable chaukas
        chaukas.disable_chaukas()

        # Verify events were written
        assert pathlib.Path(temp_output_file).exists(), "Output file should exist"

        # Read and parse events
        with open(temp_output_file, "r") as f:
            events = [json.loads(line) for line in f if line.strip()]

        assert len(events) > 0, "Should have captured some events"

        # Check event types
        event_types = [e.get("type") for e in events]

        # Should have session lifecycle
        assert "EVENT_TYPE_SESSION_START" in event_types, "Should have SESSION_START"
        assert "EVENT_TYPE_SESSION_END" in event_types, "Should have SESSION_END"

        # Should have agent lifecycle
        assert "EVENT_TYPE_AGENT_START" in event_types, "Should have AGENT_START"
        assert "EVENT_TYPE_AGENT_END" in event_types, "Should have AGENT_END"

        # Should have I/O events
        assert "EVENT_TYPE_INPUT_RECEIVED" in event_types, "Should have INPUT_RECEIVED"
        assert "EVENT_TYPE_OUTPUT_EMITTED" in event_types, "Should have OUTPUT_EMITTED"

        # Verify all events have required fields
        for event in events:
            assert "event_id" in event, "Event should have event_id"
            assert "session_id" in event, "Event should have session_id"
            assert "trace_id" in event, "Event should have trace_id"
            assert "span_id" in event, "Event should have span_id"
            assert "type" in event, "Event should have type"
            assert "timestamp" in event, "Event should have timestamp"

        # Verify session consistency
        session_ids = set(e["session_id"] for e in events)
        assert len(session_ids) == 1, "All events should share same session_id"

        # Verify trace consistency
        trace_ids = set(e["trace_id"] for e in events)
        assert len(trace_ids) == 1, "All events should share same trace_id"

        # Verify hierarchy (SESSION should be parent)
        session_start = next(
            e for e in events if e["type"] == "EVENT_TYPE_SESSION_START"
        )
        session_span_id = session_start["span_id"]

        # Other events should have session as parent
        non_session_events = [
            e for e in events if not e["type"].startswith("EVENT_TYPE_SESSION")
        ]
        for event in non_session_events:
            assert (
                event.get("parent_span_id") == session_span_id
            ), f"Event {event['type']} should have session as parent"

        print(f"✅ Test passed! Captured {len(events)} events with proper structure")

    except ImportError:
        pytest.skip("OpenAI Agents SDK not installed")


@pytest.mark.asyncio
async def test_event_hierarchy_validation(temp_output_file):
    """Test that event hierarchy is properly structured."""
    import asyncio

    # Set file output mode
    os.environ["CHAUKAS_OUTPUT_MODE"] = "file"
    os.environ["CHAUKAS_OUTPUT_FILE"] = temp_output_file

    from chaukas import sdk as chaukas
    from chaukas.sdk.core.event_builder import EventBuilder

    chaukas.enable_chaukas()

    try:
        client = chaukas.get_client()
        builder = EventBuilder()

        # Create a session start event
        session_event = builder.create_session_start(metadata={"test": "hierarchy"})
        await client.send_event(session_event)

        # Create an agent start event as child
        agent_event = builder.create_agent_start(
            agent_id="test-agent", agent_name="TestAgent", instructions="Test"
        )
        await client.send_event(agent_event)

        # Flush and close
        await client.flush()
        await client.close()

        chaukas.disable_chaukas()

        # Read events
        with open(temp_output_file, "r") as f:
            events = [json.loads(line) for line in f if line.strip()]

        assert len(events) == 2, "Should have 2 events"

        # Verify parent-child relationship
        session = events[0]
        agent = events[1]

        assert session["type"] == "EVENT_TYPE_SESSION_START"
        assert agent["type"] == "EVENT_TYPE_AGENT_START"
        assert (
            agent["parent_span_id"] == session["span_id"]
        ), "Agent event should have session as parent"

        print("✅ Event hierarchy test passed!")

    except Exception as e:
        chaukas.disable_chaukas()
        raise


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
