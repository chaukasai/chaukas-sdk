"""
Tests for LangChain integration.
Tests the callback handler implementation for comprehensive event capture.
"""

import asyncio
import json
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import pytest
from chaukas.spec.common.v1.events_pb2 import EventStatus, EventType

from chaukas.sdk.core.client import ChaukasClient
from chaukas.sdk.core.tracer import ChaukasTracer
from chaukas.sdk.integrations.langchain import ChaukasCallbackHandler, LangChainWrapper


@pytest.fixture
def mock_client():
    """Create a mock Chaukas client with async send_event for testing."""
    client = Mock(spec=ChaukasClient)
    # Store events for verification
    client.sent_events = []

    # Create an async mock that stores events
    async def async_send_event(event):
        client.sent_events.append(event)
        return None

    client.send_event = async_send_event
    return client


@pytest.fixture
def tracer(mock_client):
    """Create a tracer with mock client."""
    return ChaukasTracer(client=mock_client)


@pytest.fixture
def langchain_wrapper(tracer):
    """Create LangChain wrapper."""
    return LangChainWrapper(tracer)


@pytest.fixture
def callback_handler(langchain_wrapper, mock_client):
    """Create callback handler."""
    handler = langchain_wrapper.get_callback_handler()
    # Attach mock_client to handler for test verification
    handler._test_mock_client = mock_client
    return handler


class TestLangChainWrapper:
    """Tests for LangChain wrapper initialization."""

    def test_wrapper_initialization(self, tracer):
        """Test wrapper initializes correctly."""
        wrapper = LangChainWrapper(tracer)
        assert wrapper.tracer is tracer
        assert wrapper.event_builder is not None
        assert wrapper.callback_handler is None

    def test_get_callback_handler(self, langchain_wrapper):
        """Test callback handler creation."""
        handler = langchain_wrapper.get_callback_handler()
        assert handler is not None
        assert langchain_wrapper.callback_handler is handler

        # Should return same instance on subsequent calls
        handler2 = langchain_wrapper.get_callback_handler()
        assert handler is handler2


class TestChainCallbacks:
    """Tests for chain lifecycle callbacks."""

    def test_on_chain_start_root(self, callback_handler, mock_client):
        """Test on_chain_start for root chain creates SESSION_START."""
        # Setup
        serialized = {"name": "test_chain"}
        inputs = {"query": "test query"}
        run_id = "chain_123"

        # Execute
        callback_handler.on_chain_start(
            serialized=serialized, inputs=inputs, run_id=run_id, parent_run_id=None
        )

        # Verify SESSION_START was emitted
        assert len(mock_client.sent_events) >= 1
        session_event = None
        for event in mock_client.sent_events:
            if event.type == EventType.EVENT_TYPE_SESSION_START:
                session_event = event
                break

        assert session_event is not None
        assert callback_handler._session_started is True
        assert callback_handler._root_chain_id == str(run_id)

    def test_on_chain_start_agent(self, callback_handler, mock_client):
        """Test on_chain_start for agent chain creates AGENT_START."""
        # First start root chain
        callback_handler.on_chain_start(
            serialized={"name": "root"}, inputs={}, run_id="root_1", parent_run_id=None
        )

        mock_client.sent_events.clear()

        # Execute agent chain
        serialized = {"name": "agent_executor"}
        inputs = {"input": "test"}
        run_id = "agent_123"

        callback_handler.on_chain_start(
            serialized=serialized, inputs=inputs, run_id=run_id, parent_run_id="root_1"
        )

        # Verify AGENT_START was emitted
        agent_event = None
        for event in mock_client.sent_events:
            # event already unpacked
            if event.type == EventType.EVENT_TYPE_AGENT_START:
                agent_event = event
                break

        assert agent_event is not None
        assert agent_event.agent_id == str(run_id)

    def test_on_chain_end_root(self, callback_handler, mock_client):
        """Test on_chain_end for root chain creates SESSION_END."""
        # Start session first
        run_id = "chain_123"
        callback_handler.on_chain_start(
            serialized={"name": "test"}, inputs={}, run_id=run_id, parent_run_id=None
        )

        mock_client.sent_events.clear()

        # End chain
        callback_handler.on_chain_end(
            outputs={"result": "success"}, run_id=run_id, parent_run_id=None
        )

        # Verify SESSION_END was emitted
        session_end = None
        for event in mock_client.sent_events:
            # event already unpacked
            if event.type == EventType.EVENT_TYPE_SESSION_END:
                session_end = event
                break

        assert session_end is not None
        assert callback_handler._session_started is False

    def test_on_chain_error(self, callback_handler, mock_client):
        """Test on_chain_error creates ERROR event."""
        # Start chain
        run_id = "chain_123"
        callback_handler.on_chain_start(
            serialized={"name": "test"}, inputs={}, run_id=run_id, parent_run_id=None
        )

        mock_client.sent_events.clear()

        # Trigger error
        error = Exception("Test error")
        callback_handler.on_chain_error(error=error, run_id=run_id, parent_run_id=None)

        # Verify ERROR event was emitted
        error_event = None
        for event in mock_client.sent_events:
            # event already unpacked
            if event.type == EventType.EVENT_TYPE_ERROR:
                error_event = event
                break

        assert error_event is not None
        assert "Test error" in error_event.error.error_message


class TestLLMCallbacks:
    """Tests for LLM/model callbacks."""

    def test_on_llm_start(self, callback_handler, mock_client):
        """Test on_llm_start creates MODEL_INVOCATION_START."""
        serialized = {"name": "gpt-4"}
        prompts = ["Test prompt"]
        run_id = "llm_123"

        callback_handler.on_llm_start(
            serialized=serialized, prompts=prompts, run_id=run_id, parent_run_id=None
        )

        # Verify MODEL_INVOCATION_START was emitted
        llm_event = None
        for event in mock_client.sent_events:
            # event already unpacked
            if event.type == EventType.EVENT_TYPE_MODEL_INVOCATION_START:
                llm_event = event
                break

        assert llm_event is not None
        assert llm_event.llm_invocation.model == "gpt-4"
        assert str(run_id) in callback_handler._llm_spans

    def test_on_chat_model_start(self, callback_handler, mock_client):
        """Test on_chat_model_start creates MODEL_INVOCATION_START."""
        serialized = {"name": "gpt-4"}

        # Mock message objects
        class MockMessage:
            def __init__(self, type_val, content):
                self.type = type_val
                self.content = content

        messages = [[MockMessage("human", "Hello"), MockMessage("ai", "Hi")]]
        run_id = "chat_123"

        callback_handler.on_chat_model_start(
            serialized=serialized, messages=messages, run_id=run_id, parent_run_id=None
        )

        # Verify MODEL_INVOCATION_START was emitted
        llm_event = None
        for event in mock_client.sent_events:
            # event already unpacked
            if event.type == EventType.EVENT_TYPE_MODEL_INVOCATION_START:
                llm_event = event
                break

        assert llm_event is not None

    def test_on_llm_end(self, callback_handler, mock_client):
        """Test on_llm_end creates MODEL_INVOCATION_END."""
        # Start LLM first
        run_id = "llm_123"
        callback_handler.on_llm_start(
            serialized={"name": "gpt-4"},
            prompts=["test"],
            run_id=run_id,
            parent_run_id=None,
        )

        mock_client.sent_events.clear()

        # Mock response
        class MockGeneration:
            def __init__(self):
                self.text = "Test response"

        class MockResponse:
            def __init__(self):
                self.generations = [[MockGeneration()]]
                self.llm_output = {
                    "token_usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 20,
                        "total_tokens": 30,
                    }
                }

        response = MockResponse()

        callback_handler.on_llm_end(
            response=response, run_id=run_id, parent_run_id=None, name="gpt-4"
        )

        # Verify MODEL_INVOCATION_END was emitted
        llm_end = None
        for event in mock_client.sent_events:
            # event already unpacked
            if event.type == EventType.EVENT_TYPE_MODEL_INVOCATION_END:
                llm_end = event
                break

        assert llm_end is not None

    def test_on_llm_error(self, callback_handler, mock_client):
        """Test on_llm_error creates ERROR and MODEL_INVOCATION_END."""
        # Start LLM first
        run_id = "llm_123"
        callback_handler.on_llm_start(
            serialized={"name": "gpt-4"},
            prompts=["test"],
            run_id=run_id,
            parent_run_id=None,
        )

        mock_client.sent_events.clear()

        # Trigger error
        error = Exception("Rate limit exceeded")
        callback_handler.on_llm_error(
            error=error, run_id=run_id, parent_run_id=None, name="gpt-4"
        )

        # Verify ERROR and MODEL_INVOCATION_END were emitted
        error_event = None
        llm_end = None
        for event in mock_client.sent_events:
            # event already unpacked
            if event.type == EventType.EVENT_TYPE_ERROR:
                error_event = event
            elif event.type == EventType.EVENT_TYPE_MODEL_INVOCATION_END:
                llm_end = event

        assert error_event is not None
        assert llm_end is not None
        assert "Rate limit exceeded" in error_event.error.error_message


class TestToolCallbacks:
    """Tests for tool execution callbacks."""

    def test_on_tool_start(self, callback_handler, mock_client):
        """Test on_tool_start creates TOOL_CALL_START."""
        serialized = {"name": "calculator"}
        input_str = '{"operation": "add", "numbers": [1, 2]}'
        run_id = "tool_123"

        callback_handler.on_tool_start(
            serialized=serialized,
            input_str=input_str,
            run_id=run_id,
            parent_run_id=None,
        )

        # Verify TOOL_CALL_START was emitted
        tool_event = None
        for event in mock_client.sent_events:
            # event already unpacked
            if event.type == EventType.EVENT_TYPE_TOOL_CALL_START:
                tool_event = event
                break

        assert tool_event is not None
        assert tool_event.tool_call.name == "calculator"

    def test_on_tool_end(self, callback_handler, mock_client):
        """Test on_tool_end creates TOOL_CALL_END."""
        # Start tool first
        run_id = "tool_123"
        callback_handler.on_tool_start(
            serialized={"name": "calculator"},
            input_str="test",
            run_id=run_id,
            parent_run_id=None,
        )

        mock_client.sent_events.clear()

        # End tool
        callback_handler.on_tool_end(
            output="3", run_id=run_id, parent_run_id=None, name="calculator"
        )

        # Verify TOOL_CALL_END was emitted
        tool_end = None
        for event in mock_client.sent_events:
            # event already unpacked
            if event.type == EventType.EVENT_TYPE_TOOL_CALL_END:
                tool_end = event
                break

        assert tool_end is not None
        assert tool_end.tool_response.output.fields["output"].string_value == "3"

    def test_on_tool_error(self, callback_handler, mock_client):
        """Test on_tool_error creates ERROR and TOOL_CALL_END."""
        # Start tool first
        run_id = "tool_123"
        callback_handler.on_tool_start(
            serialized={"name": "calculator"},
            input_str="test",
            run_id=run_id,
            parent_run_id=None,
        )

        mock_client.sent_events.clear()

        # Trigger error
        error = Exception("Tool execution failed")
        callback_handler.on_tool_error(
            error=error, run_id=run_id, parent_run_id=None, name="calculator"
        )

        # Verify ERROR and TOOL_CALL_END were emitted
        error_event = None
        tool_end = None
        for event in mock_client.sent_events:
            # event already unpacked
            if event.type == EventType.EVENT_TYPE_ERROR:
                error_event = event
            elif event.type == EventType.EVENT_TYPE_TOOL_CALL_END:
                tool_end = event

        assert error_event is not None
        assert tool_end is not None


class TestRetrieverCallbacks:
    """Tests for retriever/data access callbacks."""

    def test_on_retriever_start(self, callback_handler, mock_client):
        """Test on_retriever_start creates DATA_ACCESS."""
        serialized = {"name": "vector_store"}
        query = "test query"
        run_id = "retriever_123"

        callback_handler.on_retriever_start(
            serialized=serialized, query=query, run_id=run_id, parent_run_id=None
        )

        # Verify DATA_ACCESS was emitted
        data_event = None
        for event in mock_client.sent_events:
            # event already unpacked
            if event.type == EventType.EVENT_TYPE_DATA_ACCESS:
                data_event = event
                break

        assert data_event is not None
        assert data_event.data_access.datasource == "vector_store"

    def test_on_retriever_end(self, callback_handler, mock_client):
        """Test on_retriever_end creates DATA_ACCESS with results."""
        # Start retriever first
        run_id = "retriever_123"
        callback_handler.on_retriever_start(
            serialized={"name": "vector_store"},
            query="test",
            run_id=run_id,
            parent_run_id=None,
        )

        mock_client.sent_events.clear()

        # Mock documents
        class MockDocument:
            def __init__(self, doc_id):
                self.metadata = {"id": doc_id}

        documents = [MockDocument("doc1"), MockDocument("doc2")]

        callback_handler.on_retriever_end(
            documents=documents, run_id=run_id, parent_run_id=None
        )

        # Verify DATA_ACCESS was emitted
        data_event = None
        for event in mock_client.sent_events:
            # event already unpacked
            if event.type == EventType.EVENT_TYPE_DATA_ACCESS:
                data_event = event
                break

        assert data_event is not None

    def test_on_retriever_error(self, callback_handler, mock_client):
        """Test on_retriever_error creates ERROR."""
        # Start retriever first
        run_id = "retriever_123"
        callback_handler.on_retriever_start(
            serialized={"name": "vector_store"},
            query="test",
            run_id=run_id,
            parent_run_id=None,
        )

        mock_client.sent_events.clear()

        # Trigger error
        error = Exception("Retrieval failed")
        callback_handler.on_retriever_error(
            error=error, run_id=run_id, parent_run_id=None
        )

        # Verify ERROR was emitted
        error_event = None
        for event in mock_client.sent_events:
            # event already unpacked
            if event.type == EventType.EVENT_TYPE_ERROR:
                error_event = event
                break

        assert error_event is not None
        assert "Retrieval failed" in error_event.error.error_message


class TestAgentCallbacks:
    """Tests for agent action callbacks."""

    def test_on_agent_finish(self, callback_handler, mock_client):
        """Test on_agent_finish creates OUTPUT_EMITTED."""

        # Mock finish object
        class MockAgentFinish:
            def __init__(self):
                self.return_values = {"output": "Final answer"}

        finish = MockAgentFinish()

        callback_handler.on_agent_finish(
            finish=finish, run_id="agent_123", parent_run_id=None
        )

        # Verify OUTPUT_EMITTED was emitted
        output_event = None
        for event in mock_client.sent_events:
            # event already unpacked
            if event.type == EventType.EVENT_TYPE_OUTPUT_EMITTED:
                output_event = event
                break

        assert output_event is not None


class TestRetryCallbacks:
    """Tests for retry detection and events."""

    def test_on_retry(self, callback_handler, mock_client):
        """Test on_retry creates RETRY event."""

        # Mock retry state
        class MockRetryState:
            def __init__(self):
                self.attempt_number = 2

        retry_state = MockRetryState()

        callback_handler.on_retry(
            retry_state=retry_state, run_id="retry_123", parent_run_id=None
        )

        # Verify RETRY event was emitted
        retry_event = None
        for event in mock_client.sent_events:
            # event already unpacked
            if event.type == EventType.EVENT_TYPE_RETRY:
                retry_event = event
                break

        assert retry_event is not None
        assert retry_event.retry.attempt == 2

    def test_retryable_error_detection(self, callback_handler):
        """Test retryable error detection."""
        assert callback_handler._is_retryable_error("rate limit exceeded") is True
        assert callback_handler._is_retryable_error("timeout") is True
        assert callback_handler._is_retryable_error("503 service unavailable") is True
        assert callback_handler._is_retryable_error("invalid api key") is False


class TestHelperMethods:
    """Tests for helper methods."""

    def test_extract_provider(self, callback_handler):
        """Test provider extraction from model name."""
        assert callback_handler._extract_provider("gpt-4") == "openai"
        assert callback_handler._extract_provider("claude-3") == "anthropic"
        assert callback_handler._extract_provider("gemini-pro") == "google"
        assert callback_handler._extract_provider("unknown-model") == "unknown"

    def test_agent_context_tracking(self, callback_handler):
        """Test agent context tracking."""
        # No agent context initially
        agent_id, agent_name = callback_handler._get_current_agent_context()
        assert agent_id is None
        assert agent_name is None

        # Set agent context
        callback_handler._last_active_agent = ("agent_1", "test_agent")
        agent_id, agent_name = callback_handler._get_current_agent_context()
        assert agent_id == "agent_1"
        assert agent_name == "test_agent"


class TestAgentHandoff:
    """Tests for agent handoff detection."""

    def test_agent_handoff_detection(self, callback_handler, mock_client):
        """Test agent handoff event creation."""
        # Start root chain
        callback_handler.on_chain_start(
            serialized={"name": "root"}, inputs={}, run_id="root_1", parent_run_id=None
        )

        # Start first agent
        callback_handler.on_chain_start(
            serialized={"name": "agent_1"},
            inputs={},
            run_id="agent_1",
            parent_run_id="root_1",
        )

        mock_client.sent_events.clear()

        # Start second agent - should trigger handoff
        callback_handler.on_chain_start(
            serialized={"name": "agent_2"},
            inputs={},
            run_id="agent_2",
            parent_run_id="root_1",
        )

        # Verify AGENT_HANDOFF was emitted
        handoff_event = None
        for event in mock_client.sent_events:
            # event already unpacked
            if event.type == EventType.EVENT_TYPE_AGENT_HANDOFF:
                handoff_event = event
                break

        assert handoff_event is not None
        assert handoff_event.agent_handoff.from_agent_id == "agent_1"
        assert handoff_event.agent_handoff.to_agent_id == "agent_2"
