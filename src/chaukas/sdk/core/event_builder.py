"""
Event builder for creating proto-compliant events.

Provides factory methods for all 20 proto event types with proper
context management and field population.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from google.protobuf import timestamp_pb2, struct_pb2

from chaukas.spec.common.v1.events_pb2 import (
    Event, EventType, EventStatus, Severity, Author,
    MessageContent, ToolCall, ToolResponse, LLMInvocation,
    PolicyDecision, DataAccess, ErrorInfo, RetryInfo,
    AgentHandoff, MCPCall, PerformanceMetrics, CostDetails
)

from chaukas.sdk.core.config import get_config
from chaukas.sdk.core.tracer import _session_id, _trace_id, _span_id, _parent_span_id
from chaukas.sdk.utils.uuid7 import generate_uuid7


class EventBuilder:
    """
    Builder for creating proto-compliant events.
    
    All events follow the distributed tracing hierarchy:
    session -> trace -> span -> parent_span
    """
    
    def __init__(self):
        """Initialize event builder with configuration."""
        self.config = get_config()
    
    def _create_base_event(
        self,
        event_type: EventType,
        severity: Severity = Severity.SEVERITY_INFO,
        status: EventStatus = EventStatus.EVENT_STATUS_STARTED,
        author: Author = Author.AUTHOR_SYSTEM,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
    ) -> Event:
        """
        Create base event with common fields populated.
        
        Args:
            event_type: Proto event type enum
            severity: Event severity (default INFO)
            status: Event status (default STARTED)
            author: Event author (default SYSTEM)
            agent_id: Optional agent ID
            agent_name: Optional agent name
            
        Returns:
            Proto Event with base fields populated
        """
        event = Event()
        
        # Required fields
        event.event_id = generate_uuid7()
        event.tenant_id = self.config.tenant_id
        event.project_id = self.config.project_id
        
        # Distributed tracing context
        event.session_id = _session_id.get() or generate_uuid7()
        event.trace_id = _trace_id.get() or generate_uuid7()
        event.span_id = _span_id.get() or generate_uuid7()
        
        parent_span = _parent_span_id.get()
        if parent_span:
            event.parent_span_id = parent_span
        
        # Event metadata
        event.type = event_type
        event.severity = severity
        event.status = status
        event.author = author
        
        # Timestamp
        event.timestamp.CopyFrom(timestamp_pb2.Timestamp())
        event.timestamp.GetCurrentTime()
        
        # Agent info if provided
        if agent_id:
            event.agent_id = agent_id
        if agent_name:
            event.agent_name = agent_name
        
        # Optional fields from config
        if self.config.branch:
            event.branch = self.config.branch
        if self.config.tags:
            event.tags.extend(self.config.tags)
        
        return event
    
    # Session Events
    
    def create_session_start(
        self,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Event:
        """Create SESSION_START event."""
        event = self._create_base_event(
            event_type=EventType.EVENT_TYPE_SESSION_START,
            status=EventStatus.EVENT_STATUS_STARTED,
            author=Author.AUTHOR_SYSTEM,
        )
        
        if session_id:
            event.session_id = session_id
        
        if metadata:
            self._set_metadata(event, metadata)
        
        return event
    
    def create_session_end(
        self,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Event:
        """Create SESSION_END event."""
        event = self._create_base_event(
            event_type=EventType.EVENT_TYPE_SESSION_END,
            status=EventStatus.EVENT_STATUS_COMPLETED,
            author=Author.AUTHOR_SYSTEM,
        )
        
        if session_id:
            event.session_id = session_id
        
        if metadata:
            self._set_metadata(event, metadata)
        
        return event
    
    # Agent Events
    
    def create_agent_start(
        self,
        agent_id: str,
        agent_name: str,
        role: Optional[str] = None,
        instructions: Optional[str] = None,
        tools: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Event:
        """Create AGENT_START event."""
        event = self._create_base_event(
            event_type=EventType.EVENT_TYPE_AGENT_START,
            status=EventStatus.EVENT_STATUS_STARTED,
            author=Author.AUTHOR_AGENT,
            agent_id=agent_id,
            agent_name=agent_name,
        )
        
        # Add agent-specific data to metadata
        agent_data = {}
        if role:
            agent_data["role"] = role
        if instructions:
            agent_data["instructions"] = instructions
        if tools:
            agent_data["tools"] = tools
        
        if agent_data:
            if not metadata:
                metadata = {}
            metadata.update(agent_data)
        
        if metadata:
            self._set_metadata(event, metadata)
        
        return event
    
    def create_agent_end(
        self,
        agent_id: str,
        agent_name: str,
        status: EventStatus = EventStatus.EVENT_STATUS_COMPLETED,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Event:
        """Create AGENT_END event."""
        event = self._create_base_event(
            event_type=EventType.EVENT_TYPE_AGENT_END,
            status=status,
            author=Author.AUTHOR_AGENT,
            agent_id=agent_id,
            agent_name=agent_name,
        )
        
        if metadata:
            self._set_metadata(event, metadata)
        
        return event
    
    def create_agent_handoff(
        self,
        from_agent_id: str,
        from_agent_name: str,
        to_agent_id: str,
        to_agent_name: str,
        reason: Optional[str] = None,
        handoff_type: Optional[str] = None,
        handoff_data: Optional[Dict[str, Any]] = None
    ) -> Event:
        """Create AGENT_HANDOFF event."""
        event = self._create_base_event(
            event_type=EventType.EVENT_TYPE_AGENT_HANDOFF,
            status=EventStatus.EVENT_STATUS_COMPLETED,
            author=Author.AUTHOR_AGENT,
            agent_id=from_agent_id,
            agent_name=from_agent_name,
        )
        
        # Create AgentHandoff message
        handoff = AgentHandoff()
        handoff.from_agent_id = from_agent_id
        handoff.from_agent_name = from_agent_name
        handoff.to_agent_id = to_agent_id
        handoff.to_agent_name = to_agent_name
        
        if reason:
            handoff.reason = reason
        if handoff_type:
            handoff.handoff_type = handoff_type
        if handoff_data:
            self._dict_to_struct(handoff_data, handoff.handoff_data)
        
        event.agent_handoff.CopyFrom(handoff)
        
        return event
    
    # Model Events
    
    def create_model_invocation_start(
        self,
        provider: str,
        model: str,
        messages: List[Dict[str, str]],
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Event:
        """Create MODEL_INVOCATION_START event."""
        event = self._create_base_event(
            event_type=EventType.EVENT_TYPE_MODEL_INVOCATION_START,
            status=EventStatus.EVENT_STATUS_STARTED,
            author=Author.AUTHOR_LLM,
            agent_id=agent_id,
            agent_name=agent_name,
        )
        
        # Create LLMInvocation message
        llm = LLMInvocation()
        llm.provider = provider
        llm.model = model
        
        # Set request data
        request_data = {
            "messages": messages,
        }
        if temperature is not None:
            llm.temperature = temperature
            request_data["temperature"] = temperature
        if max_tokens is not None:
            llm.max_tokens = max_tokens
            request_data["max_tokens"] = max_tokens
        if tools:
            request_data["tools"] = tools
        
        self._dict_to_struct(request_data, llm.request)
        
        # Set start time
        llm.start_time.CopyFrom(event.timestamp)
        
        event.llm_invocation.CopyFrom(llm)
        
        return event
    
    def create_model_invocation_end(
        self,
        provider: str,
        model: str,
        response_content: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        finish_reason: Optional[str] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        duration_ms: Optional[float] = None,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        error: Optional[str] = None
    ) -> Event:
        """Create MODEL_INVOCATION_END event."""
        status = EventStatus.EVENT_STATUS_FAILED if error else EventStatus.EVENT_STATUS_COMPLETED
        
        event = self._create_base_event(
            event_type=EventType.EVENT_TYPE_MODEL_INVOCATION_END,
            status=status,
            author=Author.AUTHOR_LLM,
            agent_id=agent_id,
            agent_name=agent_name,
        )
        
        # Create LLMInvocation message
        llm = LLMInvocation()
        llm.provider = provider
        llm.model = model
        
        # Set response data
        response_data = {}
        if response_content:
            response_data["content"] = response_content
        if tool_calls:
            response_data["tool_calls"] = tool_calls
        if finish_reason:
            llm.finish_reason = finish_reason
            response_data["finish_reason"] = finish_reason
        if error:
            response_data["error"] = error
        
        self._dict_to_struct(response_data, llm.response)
        
        # Set token counts
        if prompt_tokens is not None:
            llm.prompt_tokens = prompt_tokens
        if completion_tokens is not None:
            llm.completion_tokens = completion_tokens
        if total_tokens is not None:
            llm.total_tokens = total_tokens
        
        # Set timing
        llm.end_time.CopyFrom(event.timestamp)
        if duration_ms is not None:
            llm.duration_ms = duration_ms
        
        event.llm_invocation.CopyFrom(llm)
        
        # Add error info if present
        if error:
            error_info = ErrorInfo()
            error_info.error_message = error
            error_info.recoverable = True
            event.error.CopyFrom(error_info)
        
        return event
    
    # Tool Events
    
    def create_tool_call_start(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        call_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None
    ) -> Event:
        """Create TOOL_CALL_START event."""
        event = self._create_base_event(
            event_type=EventType.EVENT_TYPE_TOOL_CALL_START,
            status=EventStatus.EVENT_STATUS_STARTED,
            author=Author.AUTHOR_TOOL,
            agent_id=agent_id,
            agent_name=agent_name,
        )
        
        # Create ToolCall message
        tool_call = ToolCall()
        tool_call.name = tool_name
        self._dict_to_struct(arguments, tool_call.arguments)
        
        if call_id:
            tool_call.id = call_id
        else:
            tool_call.id = generate_uuid7()
        
        event.tool_call.CopyFrom(tool_call)
        
        return event
    
    def create_tool_call_end(
        self,
        tool_name: str,
        call_id: Optional[str] = None,
        output: Optional[Any] = None,
        error: Optional[str] = None,
        execution_time_ms: Optional[float] = None,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None
    ) -> Event:
        """Create TOOL_CALL_END event."""
        status = EventStatus.EVENT_STATUS_FAILED if error else EventStatus.EVENT_STATUS_COMPLETED
        
        event = self._create_base_event(
            event_type=EventType.EVENT_TYPE_TOOL_CALL_END,
            status=status,
            author=Author.AUTHOR_TOOL,
            agent_id=agent_id,
            agent_name=agent_name,
        )
        
        # Create ToolResponse message
        tool_response = ToolResponse()
        
        if call_id:
            tool_response.tool_call_id = call_id
        
        if output is not None:
            self._dict_to_struct({"output": output}, tool_response.output)
        
        if error:
            tool_response.error_message = error
        
        if execution_time_ms is not None:
            tool_response.execution_time_ms = execution_time_ms
        
        event.tool_response.CopyFrom(tool_response)
        
        # Add error info if present
        if error:
            error_info = ErrorInfo()
            error_info.error_message = error
            error_info.recoverable = True
            event.error.CopyFrom(error_info)
        
        return event
    
    # MCP Events
    
    def create_mcp_call_start(
        self,
        server_name: str,
        server_url: str,
        operation: str,
        method: str,
        request: Dict[str, Any],
        protocol_version: Optional[str] = None,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None
    ) -> Event:
        """Create MCP_CALL_START event."""
        event = self._create_base_event(
            event_type=EventType.EVENT_TYPE_MCP_CALL_START,
            status=EventStatus.EVENT_STATUS_STARTED,
            author=Author.AUTHOR_SYSTEM,
            agent_id=agent_id,
            agent_name=agent_name,
        )
        
        # Create MCPCall message
        mcp_call = MCPCall()
        mcp_call.server_name = server_name
        mcp_call.server_url = server_url
        mcp_call.operation = operation
        mcp_call.method = method
        self._dict_to_struct(request, mcp_call.request)
        
        if protocol_version:
            mcp_call.protocol_version = protocol_version
        
        event.mcp_call.CopyFrom(mcp_call)
        
        return event
    
    def create_mcp_call_end(
        self,
        server_name: str,
        server_url: str,
        operation: str,
        method: str,
        response: Dict[str, Any],
        execution_time_ms: Optional[float] = None,
        error: Optional[str] = None,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None
    ) -> Event:
        """Create MCP_CALL_END event."""
        status = EventStatus.EVENT_STATUS_FAILED if error else EventStatus.EVENT_STATUS_COMPLETED
        
        event = self._create_base_event(
            event_type=EventType.EVENT_TYPE_MCP_CALL_END,
            status=status,
            author=Author.AUTHOR_SYSTEM,
            agent_id=agent_id,
            agent_name=agent_name,
        )
        
        # Create MCPCall message
        mcp_call = MCPCall()
        mcp_call.server_name = server_name
        mcp_call.server_url = server_url
        mcp_call.operation = operation
        mcp_call.method = method
        self._dict_to_struct(response, mcp_call.response)
        
        if execution_time_ms is not None:
            mcp_call.execution_time_ms = execution_time_ms
        
        event.mcp_call.CopyFrom(mcp_call)
        
        # Add error info if present
        if error:
            error_info = ErrorInfo()
            error_info.error_message = error
            error_info.recoverable = True
            event.error.CopyFrom(error_info)
        
        return event
    
    # I/O Events
    
    def create_input_received(
        self,
        content: str,
        source: Optional[str] = None,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Event:
        """Create INPUT_RECEIVED event."""
        event = self._create_base_event(
            event_type=EventType.EVENT_TYPE_INPUT_RECEIVED,
            status=EventStatus.EVENT_STATUS_COMPLETED,
            author=Author.AUTHOR_USER,
            agent_id=agent_id,
            agent_name=agent_name,
        )
        
        # Create MessageContent
        message = MessageContent()
        message.role = "user"
        message.text = content
        
        if metadata:
            self._dict_to_struct(metadata, message.metadata)
        
        event.message.CopyFrom(message)
        
        return event
    
    def create_output_emitted(
        self,
        content: str,
        target: Optional[str] = None,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Event:
        """Create OUTPUT_EMITTED event."""
        event = self._create_base_event(
            event_type=EventType.EVENT_TYPE_OUTPUT_EMITTED,
            status=EventStatus.EVENT_STATUS_COMPLETED,
            author=Author.AUTHOR_AGENT,
            agent_id=agent_id,
            agent_name=agent_name,
        )
        
        # Create MessageContent
        message = MessageContent()
        message.role = "assistant"
        message.text = content
        
        if metadata:
            self._dict_to_struct(metadata, message.metadata)
        
        event.message.CopyFrom(message)
        
        return event
    
    # Error/Retry Events
    
    def create_error(
        self,
        error_message: str,
        error_code: Optional[str] = None,
        stack_trace: Optional[str] = None,
        recoverable: bool = False,
        recovery_action: Optional[str] = None,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None
    ) -> Event:
        """Create ERROR event."""
        event = self._create_base_event(
            event_type=EventType.EVENT_TYPE_ERROR,
            severity=Severity.SEVERITY_ERROR,
            status=EventStatus.EVENT_STATUS_FAILED,
            author=Author.AUTHOR_SYSTEM,
            agent_id=agent_id,
            agent_name=agent_name,
        )
        
        # Create ErrorInfo message
        error_info = ErrorInfo()
        error_info.error_message = error_message
        
        if error_code:
            error_info.error_code = error_code
        if stack_trace:
            error_info.stack_trace = stack_trace
        
        error_info.recoverable = recoverable
        
        if recovery_action:
            error_info.recovery_action = recovery_action
        
        event.error.CopyFrom(error_info)
        
        return event
    
    def create_retry(
        self,
        attempt: int,
        strategy: str,
        backoff_ms: int,
        reason: Optional[str] = None,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None
    ) -> Event:
        """Create RETRY event."""
        event = self._create_base_event(
            event_type=EventType.EVENT_TYPE_RETRY,
            severity=Severity.SEVERITY_WARN,
            status=EventStatus.EVENT_STATUS_IN_PROGRESS,
            author=Author.AUTHOR_SYSTEM,
            agent_id=agent_id,
            agent_name=agent_name,
        )
        
        # Create RetryInfo message
        retry_info = RetryInfo()
        retry_info.attempt = attempt
        retry_info.strategy = strategy
        retry_info.backoff_ms = backoff_ms
        
        event.retry.CopyFrom(retry_info)
        
        if reason:
            metadata = {"retry_reason": reason}
            self._set_metadata(event, metadata)
        
        return event
    
    # Policy/Data Events
    
    def create_policy_decision(
        self,
        policy_id: str,
        outcome: str,
        rule_ids: List[str],
        rationale: Optional[str] = None,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None
    ) -> Event:
        """Create POLICY_DECISION event."""
        event = self._create_base_event(
            event_type=EventType.EVENT_TYPE_POLICY_DECISION,
            status=EventStatus.EVENT_STATUS_COMPLETED,
            author=Author.AUTHOR_SYSTEM,
            agent_id=agent_id,
            agent_name=agent_name,
        )
        
        # Create PolicyDecision message
        policy = PolicyDecision()
        policy.policy_id = policy_id
        policy.outcome = outcome
        policy.rule_ids.extend(rule_ids)
        
        if rationale:
            policy.rationale = rationale
        
        event.policy.CopyFrom(policy)
        
        return event
    
    def create_data_access(
        self,
        datasource: str,
        document_ids: Optional[List[str]] = None,
        chunk_ids: Optional[List[str]] = None,
        pii_categories: Optional[List[str]] = None,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None
    ) -> Event:
        """Create DATA_ACCESS event."""
        event = self._create_base_event(
            event_type=EventType.EVENT_TYPE_DATA_ACCESS,
            status=EventStatus.EVENT_STATUS_COMPLETED,
            author=Author.AUTHOR_SYSTEM,
            agent_id=agent_id,
            agent_name=agent_name,
        )
        
        # Create DataAccess message
        data_access = DataAccess()
        data_access.datasource = datasource
        
        if document_ids:
            data_access.document_ids.extend(document_ids)
        if chunk_ids:
            data_access.chunk_ids.extend(chunk_ids)
        if pii_categories:
            data_access.pii_categories.extend(pii_categories)
            # Also set on event level
            event.pii_categories.extend(pii_categories)
        
        event.data_access.CopyFrom(data_access)
        
        return event
    
    def create_state_update(
        self,
        state_data: Dict[str, Any],
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None
    ) -> Event:
        """Create STATE_UPDATE event."""
        event = self._create_base_event(
            event_type=EventType.EVENT_TYPE_STATE_UPDATE,
            status=EventStatus.EVENT_STATUS_COMPLETED,
            author=Author.AUTHOR_SYSTEM,
            agent_id=agent_id,
            agent_name=agent_name,
        )
        
        # Set state update as struct
        self._dict_to_struct(state_data, event.state_update)
        
        return event
    
    # System Events
    
    def create_system_event(
        self,
        message: str,
        severity: Severity = Severity.SEVERITY_INFO,
        metadata: Optional[Dict[str, Any]] = None,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None
    ) -> Event:
        """Create SYSTEM event."""
        event = self._create_base_event(
            event_type=EventType.EVENT_TYPE_SYSTEM,
            severity=severity,
            status=EventStatus.EVENT_STATUS_COMPLETED,
            author=Author.AUTHOR_SYSTEM,
            agent_id=agent_id,
            agent_name=agent_name,
        )
        
        # Add system message to metadata
        if not metadata:
            metadata = {}
        metadata["message"] = message
        
        self._set_metadata(event, metadata)
        
        return event
    
    # Helper methods
    
    def _dict_to_struct(self, data: Dict[str, Any], struct: struct_pb2.Struct) -> None:
        """Convert Python dict to protobuf Struct."""
        for key, value in data.items():
            if value is None:
                struct[key] = None
            elif isinstance(value, bool):
                struct[key] = value
            elif isinstance(value, (int, float)):
                struct[key] = value
            elif isinstance(value, str):
                struct[key] = value
            elif isinstance(value, dict):
                struct[key] = {}
                self._dict_to_struct(value, struct[key])
            elif isinstance(value, list):
                struct[key] = value
            else:
                # Convert to string for unsupported types
                struct[key] = str(value)
    
    def _set_metadata(self, event: Event, metadata: Dict[str, Any]) -> None:
        """Set metadata on event."""
        self._dict_to_struct(metadata, event.metadata)