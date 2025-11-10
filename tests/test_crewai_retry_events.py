"""Test retry event capture in CrewAI integration."""

import asyncio
import os
from unittest.mock import MagicMock, Mock, patch

import pytest
from chaukas.spec.common.v1.events_pb2 import EventType

from chaukas.sdk.core.event_builder import EventBuilder
from chaukas.sdk.core.tracer import ChaukasTracer
from chaukas.sdk.integrations.crewai import CrewAIEventBusListener, CrewAIWrapper

# Set required environment variables for testing
os.environ["CHAUKAS_TENANT_ID"] = "test-tenant"
os.environ["CHAUKAS_PROJECT_ID"] = "test-project"
os.environ["CHAUKAS_ENDPOINT"] = "http://test-endpoint"
os.environ["CHAUKAS_API_KEY"] = "test-key"


class TestCrewAIRetryEvents:
    """Test retry event capture in CrewAI integration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tracer = Mock(spec=ChaukasTracer)
        self.tracer.client = Mock()
        self.tracer.client.send_event = Mock()
        self.wrapper = CrewAIWrapper(self.tracer)
        self.listener = CrewAIEventBusListener(self.wrapper)

    def test_llm_retry_event_on_rate_limit_error(self):
        """Test that RETRY event is emitted for rate limit errors in LLM calls."""
        # Create a mock LLM failed event
        event = Mock()
        event.provider = "openai"
        event.model = "gpt-4"
        event.error = "Rate limit exceeded. Please retry after 2 seconds."
        event.agent_id = "test-agent"
        event.agent_role = "researcher"

        # Track events sent
        sent_events = []

        def capture_event(event):
            sent_events.append(event)

        self.wrapper._send_event_sync = Mock(side_effect=capture_event)

        # Handle the failed event
        self.listener._handle_llm_failed(event)

        # Verify RETRY event was sent
        assert len(sent_events) == 2  # RETRY and MODEL_INVOCATION_END
        retry_event = sent_events[0]
        assert retry_event.type == EventType.EVENT_TYPE_RETRY
        assert retry_event.retry.attempt == 1
        assert retry_event.retry.strategy == "exponential"
        assert retry_event.retry.backoff_ms == 1000
        # Check metadata contains retry reason
        assert "retry_reason" in retry_event.metadata.fields
        assert "Rate limit" in retry_event.metadata.fields["retry_reason"].string_value

        # Verify MODEL_INVOCATION_END was also sent
        end_event = sent_events[1]
        assert end_event.type == EventType.EVENT_TYPE_MODEL_INVOCATION_END

    def test_tool_retry_event_on_timeout_error(self):
        """Test that RETRY event is emitted for timeout errors in tool calls."""
        # Create a mock tool error event
        event = Mock()
        event.tool_name = "search_tool"
        event.error = "Connection timeout while fetching results"
        event.agent_id = "test-agent"
        event.agent_role = "researcher"

        # Track events sent
        sent_events = []

        def capture_event(event):
            sent_events.append(event)

        self.wrapper._send_event_sync = Mock(side_effect=capture_event)

        # Handle the error event
        self.listener._handle_tool_error(event)

        # Verify RETRY event was sent
        assert len(sent_events) == 2  # RETRY and TOOL_CALL_END
        retry_event = sent_events[0]
        assert retry_event.type == EventType.EVENT_TYPE_RETRY
        assert retry_event.retry.attempt == 1
        assert retry_event.retry.strategy == "linear"
        assert retry_event.retry.backoff_ms == 500
        # Check metadata contains retry reason
        assert "retry_reason" in retry_event.metadata.fields
        assert (
            "timeout"
            in retry_event.metadata.fields["retry_reason"].string_value.lower()
        )

        # Verify TOOL_CALL_END was also sent
        end_event = sent_events[1]
        assert end_event.type == EventType.EVENT_TYPE_TOOL_CALL_END

    def test_task_retry_event_on_network_error(self):
        """Test that RETRY event is emitted for network errors in task execution."""
        # Create a mock task failed event
        event = Mock()
        event.task_id = "task-123"
        event.error = "Network unavailable. Please check connection."
        event.agent_id = "test-agent"
        event.agent_role = "researcher"

        # Track events sent
        sent_events = []

        def capture_event(event):
            sent_events.append(event)

        self.wrapper._send_event_sync = Mock(side_effect=capture_event)

        # Handle the failed event
        self.listener._handle_task_failed(event)

        # Verify RETRY event was sent
        assert len(sent_events) == 2  # RETRY and ERROR
        retry_event = sent_events[0]
        assert retry_event.type == EventType.EVENT_TYPE_RETRY
        assert retry_event.retry.attempt == 1
        assert retry_event.retry.strategy == "exponential"
        assert retry_event.retry.backoff_ms == 2000
        # Check metadata contains retry reason
        assert "retry_reason" in retry_event.metadata.fields
        assert "Network" in retry_event.metadata.fields["retry_reason"].string_value

        # Verify ERROR event was also sent
        error_event = sent_events[1]
        assert error_event.type == EventType.EVENT_TYPE_ERROR

    def test_no_retry_on_non_retryable_error(self):
        """Test that no RETRY event is emitted for non-retryable errors."""
        # Create a mock LLM failed event with non-retryable error
        event = Mock()
        event.provider = "openai"
        event.model = "gpt-4"
        event.error = "Invalid API key"
        event.agent_id = "test-agent"
        event.agent_role = "researcher"

        # Track events sent
        sent_events = []

        def capture_event(event):
            sent_events.append(event)

        self.wrapper._send_event_sync = Mock(side_effect=capture_event)

        # Handle the failed event
        self.listener._handle_llm_failed(event)

        # Verify only MODEL_INVOCATION_END was sent (no RETRY)
        assert len(sent_events) == 1
        end_event = sent_events[0]
        assert end_event.type == EventType.EVENT_TYPE_MODEL_INVOCATION_END

    def test_retry_counter_increases_on_multiple_failures(self):
        """Test that retry attempt counter increases correctly."""
        # Create a mock LLM failed event
        event = Mock()
        event.provider = "openai"
        event.model = "gpt-4"
        event.error = "503 Service Unavailable"
        event.agent_id = "test-agent"
        event.agent_role = "researcher"

        # Track events sent
        sent_events = []

        def capture_event(event):
            sent_events.append(event)

        self.wrapper._send_event_sync = Mock(side_effect=capture_event)

        # First failure
        self.listener._handle_llm_failed(event)
        assert sent_events[0].retry.attempt == 1
        assert sent_events[0].retry.backoff_ms == 1000  # 1s for first retry

        # Second failure
        sent_events.clear()
        self.listener._handle_llm_failed(event)
        assert sent_events[0].retry.attempt == 2
        assert sent_events[0].retry.backoff_ms == 2000  # 2s for second retry

        # Third failure
        sent_events.clear()
        self.listener._handle_llm_failed(event)
        assert sent_events[0].retry.attempt == 3
        assert sent_events[0].retry.backoff_ms == 4000  # 4s for third retry

        # Fourth failure (exceeds max retries)
        sent_events.clear()
        self.listener._handle_llm_failed(event)
        # No RETRY event should be sent, only MODEL_INVOCATION_END
        assert len(sent_events) == 1
        assert sent_events[0].type == EventType.EVENT_TYPE_MODEL_INVOCATION_END

    def test_retry_counter_resets_on_success(self):
        """Test that retry counter resets after successful completion."""
        # Create mock events
        failed_event = Mock()
        failed_event.provider = "openai"
        failed_event.model = "gpt-4"
        failed_event.error = "429 Too Many Requests"
        failed_event.agent_id = "test-agent"
        failed_event.agent_role = "researcher"

        success_event = Mock()
        success_event.provider = "openai"
        success_event.model = "gpt-4"
        success_event.agent_id = "test-agent"
        success_event.agent_role = "researcher"

        # Track events sent
        sent_events = []

        def capture_event(event):
            sent_events.append(event)

        self.wrapper._send_event_sync = Mock(side_effect=capture_event)

        # First failure
        self.listener._handle_llm_failed(failed_event)
        assert sent_events[0].retry.attempt == 1

        # Success - should reset counter
        self.listener._handle_llm_completed(success_event)

        # Another failure should start from attempt 1 again
        sent_events.clear()
        self.listener._handle_llm_failed(failed_event)
        assert sent_events[0].retry.attempt == 1

    def test_is_retryable_error(self):
        """Test the _is_retryable_error helper method."""
        assert self.listener._is_retryable_error("Rate limit exceeded")
        assert self.listener._is_retryable_error("Connection timeout")
        assert self.listener._is_retryable_error("503 Service Unavailable")
        assert self.listener._is_retryable_error("429 Too Many Requests")
        assert self.listener._is_retryable_error("Network error occurred")
        assert self.listener._is_retryable_error("Service temporarily unavailable")

        assert not self.listener._is_retryable_error("Invalid API key")
        assert not self.listener._is_retryable_error("Syntax error in query")
        assert not self.listener._is_retryable_error("Permission denied")
