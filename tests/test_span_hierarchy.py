"""
Tests for validating span hierarchy and START/END event matching.

This test suite ensures:
1. START/END events share the same span_id
2. Parent-child hierarchy is correct
3. No orphaned parent references
4. Proper distributed tracing context
"""

import json
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import pytest
from chaukas.spec.common.v1.events_pb2 import EventStatus, EventType

from chaukas.sdk.core.config import ChaukasConfig, set_config
from chaukas.sdk.core.event_builder import EventBuilder
from chaukas.sdk.core.tracer import ChaukasTracer, _parent_span_id, _span_id


@pytest.fixture(autouse=True)
def setup_config():
    """Set up test configuration."""
    config = ChaukasConfig(
        tenant_id="test-tenant",
        project_id="test-project",
        endpoint="https://test.chaukas.ai",
        api_key="test-key",
    )
    set_config(config)
    yield
    # Reset config after test
    from chaukas.sdk.core.config import reset_config

    reset_config()


@pytest.fixture
def builder():
    """Create EventBuilder instance."""
    return EventBuilder()


class SpanHierarchyValidator:
    """Validator for span hierarchy and event relationships."""

    def __init__(self):
        self.events = []
        self.span_events = defaultdict(list)  # span_id -> [events]
        self.parent_child_map = defaultdict(set)  # parent_span_id -> {child_span_ids}
        self.start_end_pairs = defaultdict(
            dict
        )  # event_type -> {identifier -> {"start": event, "end": event}}

    def add_event(self, event):
        """Add an event to the validator."""
        self.events.append(event)

        # Track by span_id
        self.span_events[event.span_id].append(event)

        # Track parent-child relationships
        if event.parent_span_id:
            self.parent_child_map[event.parent_span_id].add(event.span_id)

        # Track START/END pairs
        self._track_start_end_pair(event)

    def _track_start_end_pair(self, event):
        """Track START/END event pairs."""
        event_type = event.type

        # Map of START/END event type pairs
        start_end_map = {
            EventType.EVENT_TYPE_SESSION_START: EventType.EVENT_TYPE_SESSION_END,
            EventType.EVENT_TYPE_AGENT_START: EventType.EVENT_TYPE_AGENT_END,
            EventType.EVENT_TYPE_MODEL_INVOCATION_START: EventType.EVENT_TYPE_MODEL_INVOCATION_END,
            EventType.EVENT_TYPE_TOOL_CALL_START: EventType.EVENT_TYPE_TOOL_CALL_END,
            EventType.EVENT_TYPE_MCP_CALL_START: EventType.EVENT_TYPE_MCP_CALL_END,
        }

        # Determine if this is a START or END event
        if event_type in start_end_map:
            # This is a START event
            pair_key = self._get_pair_key(event)
            self.start_end_pairs[event_type][pair_key] = {"start": event, "end": None}
        elif event_type in start_end_map.values():
            # This is an END event - find its START
            for start_type, end_type in start_end_map.items():
                if event_type == end_type:
                    pair_key = self._get_pair_key(event)
                    if pair_key in self.start_end_pairs[start_type]:
                        self.start_end_pairs[start_type][pair_key]["end"] = event
                    break

    def _get_pair_key(self, event) -> str:
        """Get a unique key for matching START/END pairs."""
        # Use session_id for SESSION events
        if event.type in [
            EventType.EVENT_TYPE_SESSION_START,
            EventType.EVENT_TYPE_SESSION_END,
        ]:
            return event.session_id

        # Use agent_id for AGENT events
        if event.type in [
            EventType.EVENT_TYPE_AGENT_START,
            EventType.EVENT_TYPE_AGENT_END,
        ]:
            return event.agent_id

        # Use a combination for MODEL_INVOCATION events
        if event.type in [
            EventType.EVENT_TYPE_MODEL_INVOCATION_START,
            EventType.EVENT_TYPE_MODEL_INVOCATION_END,
        ]:
            return f"{event.agent_id}_{event.llm_invocation.model}"

        # Use tool call_id for TOOL_CALL events
        if event.type == EventType.EVENT_TYPE_TOOL_CALL_START:
            return event.tool_call.id if event.tool_call.id else event.tool_call.name
        if event.type == EventType.EVENT_TYPE_TOOL_CALL_END:
            return (
                event.tool_response.tool_call_id
                if event.tool_response.tool_call_id
                else "unknown"
            )

        # Use server name for MCP_CALL events
        if event.type in [
            EventType.EVENT_TYPE_MCP_CALL_START,
            EventType.EVENT_TYPE_MCP_CALL_END,
        ]:
            return event.mcp_call.server_name

        return "unknown"

    def validate_span_matching(self) -> List[str]:
        """Validate that START/END events share the same span_id."""
        errors = []

        for event_type, pairs in self.start_end_pairs.items():
            for pair_key, pair_events in pairs.items():
                start_event = pair_events.get("start")
                end_event = pair_events.get("end")

                if start_event and end_event:
                    if start_event.span_id != end_event.span_id:
                        errors.append(
                            f"Span mismatch for {EventType.Name(event_type)} (key: {pair_key}): "
                            f"START span={start_event.span_id[-8:]}, END span={end_event.span_id[-8:]}"
                        )
                elif start_event and not end_event:
                    errors.append(
                        f"Missing END event for {EventType.Name(event_type)} (key: {pair_key})"
                    )

        return errors

    def validate_parent_hierarchy(self) -> List[str]:
        """Validate that all parent_span_ids reference existing spans."""
        errors = []
        all_span_ids = set(self.span_events.keys())

        for event in self.events:
            if event.parent_span_id and event.parent_span_id not in all_span_ids:
                errors.append(
                    f"Orphaned parent reference: Event {EventType.Name(event.type)} "
                    f"(span={event.span_id[-8:]}) references non-existent parent {event.parent_span_id[-8:]}"
                )

        return errors

    def validate_hierarchy_depth(self) -> List[str]:
        """Validate that the hierarchy follows expected patterns."""
        errors = []

        # SESSION events should not have parents
        for event in self.events:
            if event.type in [
                EventType.EVENT_TYPE_SESSION_START,
                EventType.EVENT_TYPE_SESSION_END,
            ]:
                if event.parent_span_id:
                    errors.append(
                        f"SESSION event should not have parent: {EventType.Name(event.type)} "
                        f"has parent_span_id={event.parent_span_id[-8:]}"
                    )

        return errors

    def get_summary(self) -> Dict:
        """Get a summary of the validation results."""
        return {
            "total_events": len(self.events),
            "unique_spans": len(self.span_events),
            "parent_child_relationships": len(self.parent_child_map),
            "start_end_pairs": sum(
                len(pairs) for pairs in self.start_end_pairs.values()
            ),
        }


def test_session_span_matching(builder):
    """Test that SESSION_START and SESSION_END share the same span_id."""
    validator = SpanHierarchyValidator()

    # Create SESSION_START
    session_start = builder.create_session_start(
        session_id="test-session-123", metadata={"test": "value"}
    )
    validator.add_event(session_start)

    # Create SESSION_END with same span_id
    session_end = builder.create_session_end(
        session_id="test-session-123",
        span_id=session_start.span_id,
        metadata={"result": "success"},
    )
    validator.add_event(session_end)

    # Validate
    errors = validator.validate_span_matching()
    assert len(errors) == 0, f"Span matching errors: {errors}"

    # Verify they share the same span_id
    assert session_start.span_id == session_end.span_id


def test_agent_span_matching(builder):
    """Test that AGENT_START and AGENT_END share the same span_id."""
    validator = SpanHierarchyValidator()

    # Create AGENT_START
    agent_start = builder.create_agent_start(
        agent_id="agent-456", agent_name="Test Agent", role="assistant"
    )
    validator.add_event(agent_start)

    # Create AGENT_END with same span_id
    agent_end = builder.create_agent_end(
        agent_id="agent-456",
        agent_name="Test Agent",
        span_id=agent_start.span_id,
        status=EventStatus.EVENT_STATUS_COMPLETED,
    )
    validator.add_event(agent_end)

    # Validate
    errors = validator.validate_span_matching()
    assert len(errors) == 0, f"Span matching errors: {errors}"

    # Verify they share the same span_id
    assert agent_start.span_id == agent_end.span_id


def test_model_invocation_span_matching(builder):
    """Test that MODEL_INVOCATION_START and MODEL_INVOCATION_END share the same span_id."""
    validator = SpanHierarchyValidator()

    # Create MODEL_INVOCATION_START
    model_start = builder.create_model_invocation_start(
        provider="openai",
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
        agent_id="agent-789",
        agent_name="GPT Agent",
    )
    validator.add_event(model_start)

    # Create MODEL_INVOCATION_END with same span_id
    model_end = builder.create_model_invocation_end(
        provider="openai",
        model="gpt-4",
        response_content="Hello! How can I help you?",
        span_id=model_start.span_id,
        agent_id="agent-789",
        agent_name="GPT Agent",
    )
    validator.add_event(model_end)

    # Validate
    errors = validator.validate_span_matching()
    assert len(errors) == 0, f"Span matching errors: {errors}"

    # Verify they share the same span_id
    assert model_start.span_id == model_end.span_id


def test_tool_call_span_matching(builder):
    """Test that TOOL_CALL_START and TOOL_CALL_END share the same span_id."""
    validator = SpanHierarchyValidator()

    # Create TOOL_CALL_START
    tool_start = builder.create_tool_call_start(
        tool_name="calculator",
        arguments={"operation": "add", "a": 1, "b": 2},
        call_id="tool-call-123",
    )
    validator.add_event(tool_start)

    # Create TOOL_CALL_END with same span_id
    tool_end = builder.create_tool_call_end(
        tool_name="calculator",
        call_id="tool-call-123",
        output="3",
        span_id=tool_start.span_id,
    )
    validator.add_event(tool_end)

    # Validate
    errors = validator.validate_span_matching()
    assert len(errors) == 0, f"Span matching errors: {errors}"

    # Verify they share the same span_id
    assert tool_start.span_id == tool_end.span_id


def test_mcp_call_span_matching(builder):
    """Test that MCP_CALL_START and MCP_CALL_END share the same span_id."""
    validator = SpanHierarchyValidator()

    # Create MCP_CALL_START
    mcp_start = builder.create_mcp_call_start(
        server_name="test-server",
        server_url="mcp://localhost:8080",
        operation="execute",
        method="run",
        request={"command": "test"},
    )
    validator.add_event(mcp_start)

    # Create MCP_CALL_END with same span_id
    mcp_end = builder.create_mcp_call_end(
        server_name="test-server",
        server_url="mcp://localhost:8080",
        operation="execute",
        method="run",
        response={"result": "success"},
        span_id=mcp_start.span_id,
    )
    validator.add_event(mcp_end)

    # Validate
    errors = validator.validate_span_matching()
    assert len(errors) == 0, f"Span matching errors: {errors}"

    # Verify they share the same span_id
    assert mcp_start.span_id == mcp_end.span_id


def test_parent_hierarchy_validation(builder):
    """Test that parent-child hierarchy is properly maintained."""
    validator = SpanHierarchyValidator()

    # Create a hierarchy: SESSION -> AGENT -> MODEL_INVOCATION

    # SESSION_START (root)
    session_start = builder.create_session_start(session_id="session-abc")
    validator.add_event(session_start)

    # Set session span as parent context
    token = _parent_span_id.set(session_start.span_id)

    try:
        # AGENT_START (child of SESSION)
        agent_start = builder.create_agent_start(
            agent_id="agent-def", agent_name="Child Agent"
        )
        validator.add_event(agent_start)

        # Verify parent relationship
        assert agent_start.parent_span_id == session_start.span_id

        # Set agent span as parent context
        token2 = _parent_span_id.set(agent_start.span_id)

        try:
            # MODEL_INVOCATION_START (child of AGENT)
            model_start = builder.create_model_invocation_start(
                provider="openai",
                model="gpt-4",
                messages=[{"role": "user", "content": "test"}],
                agent_id="agent-def",
                agent_name="Child Agent",
            )
            validator.add_event(model_start)

            # Verify parent relationship
            assert model_start.parent_span_id == agent_start.span_id

        finally:
            _parent_span_id.reset(token2)
    finally:
        _parent_span_id.reset(token)

    # Validate hierarchy
    parent_errors = validator.validate_parent_hierarchy()
    assert len(parent_errors) == 0, f"Parent hierarchy errors: {parent_errors}"

    hierarchy_errors = validator.validate_hierarchy_depth()
    assert len(hierarchy_errors) == 0, f"Hierarchy depth errors: {hierarchy_errors}"

    # Verify summary
    summary = validator.get_summary()
    assert summary["total_events"] == 3
    assert summary["parent_child_relationships"] == 2  # session->agent, agent->model


def test_session_events_have_no_parent(builder):
    """Test that SESSION events never have a parent_span_id."""
    # Even with parent context set, SESSION events should have no parent
    token = _parent_span_id.set("some-parent-span")

    try:
        session_start = builder.create_session_start(session_id="test-session")
        # Check that parent_span_id is empty (protobuf default for string is empty string)
        assert (
            session_start.parent_span_id == ""
        ), "SESSION_START should not have parent_span_id"

        session_end = builder.create_session_end(session_id="test-session")
        assert (
            session_end.parent_span_id == ""
        ), "SESSION_END should not have parent_span_id"
    finally:
        _parent_span_id.reset(token)


def test_span_registry_cleanup(builder):
    """Test that span registry properly cleans up after use."""
    # Create SESSION_START
    session_start = builder.create_session_start(session_id="cleanup-test")

    # Registry should have the entry
    assert ("session", "cleanup-test") in builder._span_registry

    # Create SESSION_END using registry
    session_end = builder.create_session_end(session_id="cleanup-test")

    # Registry should be cleaned up
    assert ("session", "cleanup-test") not in builder._span_registry

    # Spans should match
    assert session_start.span_id == session_end.span_id


def test_missing_end_event_detection(builder):
    """Test detection of missing END events."""
    validator = SpanHierarchyValidator()

    # Add only START events
    session_start = builder.create_session_start(session_id="incomplete-session")
    validator.add_event(session_start)

    agent_start = builder.create_agent_start(
        agent_id="incomplete-agent", agent_name="Test"
    )
    validator.add_event(agent_start)

    # Validate - should detect missing END events
    errors = validator.validate_span_matching()
    assert len(errors) == 2, "Should detect 2 missing END events"
    assert any("Missing END event" in error for error in errors)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
