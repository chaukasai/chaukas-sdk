"""
Proto event wrapper for better developer experience.

Provides a thin wrapper around proto Event messages with convenient
methods for setting content and converting to wire format.
"""

from typing import Optional, Dict, Any, Union
from google.protobuf import struct_pb2
from google.protobuf.json_format import MessageToDict, MessageToJson

from chaukas.spec.common.v1.events_pb2 import (
    Event, EventType, MessageContent, LLMInvocation, ToolCall, ToolResponse,
    AgentHandoff, ErrorInfo, MCPCall, PolicyDecision, DataAccess,
    PerformanceMetrics, CostDetails
)
from chaukas.spec.client.v1.client_pb2 import IngestEventRequest

from .event_builder import EventBuilder


class EventWrapper:
    """
    Wrapper around proto Event for convenient usage.
    
    Provides fluent API for building events and easy conversion
    to proto messages for sending.
    """
    
    def __init__(self, event: Optional[Event] = None):
        """
        Initialize wrapper with optional proto event.
        
        Args:
            event: Optional proto Event, creates new if not provided
        """
        self._event = event or Event()
        self._builder = EventBuilder()
    
    @classmethod
    def from_builder(cls, builder_method: str, **kwargs) -> "EventWrapper":
        """
        Create wrapper using EventBuilder method.
        
        Args:
            builder_method: Name of EventBuilder method (e.g., "create_agent_start")
            **kwargs: Arguments for the builder method
            
        Returns:
            EventWrapper instance
        """
        builder = EventBuilder()
        method = getattr(builder, builder_method)
        event = method(**kwargs)
        return cls(event)
    
    # Content setters for fluent API
    
    def with_message(self, role: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> "EventWrapper":
        """Set message content."""
        message = MessageContent()
        message.role = role
        message.text = text
        
        if metadata:
            self._dict_to_struct(metadata, message.metadata)
        
        self._event.message.CopyFrom(message)
        return self
    
    def with_llm_invocation(
        self,
        provider: str,
        model: str,
        request: Optional[Dict[str, Any]] = None,
        response: Optional[Dict[str, Any]] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        duration_ms: Optional[float] = None
    ) -> "EventWrapper":
        """Set LLM invocation content."""
        llm = LLMInvocation()
        llm.provider = provider
        llm.model = model
        
        if request:
            self._dict_to_struct(request, llm.request)
        if response:
            self._dict_to_struct(response, llm.response)
        
        if prompt_tokens is not None:
            llm.prompt_tokens = prompt_tokens
        if completion_tokens is not None:
            llm.completion_tokens = completion_tokens
        if total_tokens is not None:
            llm.total_tokens = total_tokens
        if duration_ms is not None:
            llm.duration_ms = duration_ms
        
        self._event.llm_invocation.CopyFrom(llm)
        return self
    
    def with_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        call_id: Optional[str] = None,
        auth_required: bool = False,
        function_name: Optional[str] = None
    ) -> "EventWrapper":
        """Set tool call content."""
        tool_call = ToolCall()
        tool_call.name = tool_name
        self._dict_to_struct(arguments, tool_call.arguments)
        
        if call_id:
            tool_call.id = call_id
        
        tool_call.auth_required = auth_required
        
        if function_name:
            tool_call.function_name = function_name
        
        self._event.tool_call.CopyFrom(tool_call)
        return self
    
    def with_tool_response(
        self,
        tool_name: str,
        output: Optional[Any] = None,
        error_message: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        execution_time_ms: Optional[float] = None,
        http_status: Optional[int] = None
    ) -> "EventWrapper":
        """Set tool response content."""
        tool_response = ToolResponse()
        tool_response.tool_name = tool_name
        
        if output is not None:
            self._dict_to_struct({"output": output}, tool_response.output)
        
        if error_message:
            tool_response.error_message = error_message
        
        if tool_call_id:
            tool_response.tool_call_id = tool_call_id
        
        if execution_time_ms is not None:
            tool_response.execution_time_ms = execution_time_ms
        
        if http_status is not None:
            tool_response.http_status = http_status
        
        self._event.tool_response.CopyFrom(tool_response)
        return self
    
    def with_agent_handoff(
        self,
        from_agent_id: str,
        from_agent_name: str,
        to_agent_id: str,
        to_agent_name: str,
        handoff_data: Optional[Dict[str, Any]] = None,
        reason: Optional[str] = None,
        handoff_type: Optional[str] = None
    ) -> "EventWrapper":
        """Set agent handoff content."""
        handoff = AgentHandoff()
        handoff.from_agent_id = from_agent_id
        handoff.from_agent_name = from_agent_name
        handoff.to_agent_id = to_agent_id
        handoff.to_agent_name = to_agent_name
        
        if handoff_data:
            self._dict_to_struct(handoff_data, handoff.handoff_data)
        
        if reason:
            handoff.reason = reason
        
        if handoff_type:
            handoff.handoff_type = handoff_type
        
        self._event.agent_handoff.CopyFrom(handoff)
        return self
    
    def with_error(
        self,
        error_message: str,
        error_code: Optional[str] = None,
        stack_trace: Optional[str] = None,
        recoverable: bool = False,
        recovery_action: Optional[str] = None
    ) -> "EventWrapper":
        """Set error content."""
        error = ErrorInfo()
        error.error_message = error_message
        
        if error_code:
            error.error_code = error_code
        
        if stack_trace:
            error.stack_trace = stack_trace
        
        error.recoverable = recoverable
        
        if recovery_action:
            error.recovery_action = recovery_action
        
        self._event.error.CopyFrom(error)
        return self
    
    def with_mcp_call(
        self,
        server_name: str,
        server_url: str,
        operation: str,
        method: str,
        request: Optional[Dict[str, Any]] = None,
        response: Optional[Dict[str, Any]] = None,
        protocol_version: Optional[str] = None,
        execution_time_ms: Optional[float] = None
    ) -> "EventWrapper":
        """Set MCP call content."""
        mcp = MCPCall()
        mcp.server_name = server_name
        mcp.server_url = server_url
        mcp.operation = operation
        mcp.method = method
        
        if request:
            self._dict_to_struct(request, mcp.request)
        
        if response:
            self._dict_to_struct(response, mcp.response)
        
        if protocol_version:
            mcp.protocol_version = protocol_version
        
        if execution_time_ms is not None:
            mcp.execution_time_ms = execution_time_ms
        
        self._event.mcp_call.CopyFrom(mcp)
        return self
    
    def with_policy(
        self,
        policy_id: str,
        outcome: str,
        rule_ids: list,
        rationale: Optional[str] = None
    ) -> "EventWrapper":
        """Set policy decision content."""
        policy = PolicyDecision()
        policy.policy_id = policy_id
        policy.outcome = outcome
        policy.rule_ids.extend(rule_ids)
        
        if rationale:
            policy.rationale = rationale
        
        self._event.policy.CopyFrom(policy)
        return self
    
    def with_data_access(
        self,
        datasource: str,
        document_ids: Optional[list] = None,
        chunk_ids: Optional[list] = None,
        pii_categories: Optional[list] = None
    ) -> "EventWrapper":
        """Set data access content."""
        data_access = DataAccess()
        data_access.datasource = datasource
        
        if document_ids:
            data_access.document_ids.extend(document_ids)
        if chunk_ids:
            data_access.chunk_ids.extend(chunk_ids)
        if pii_categories:
            data_access.pii_categories.extend(pii_categories)
        
        self._event.data_access.CopyFrom(data_access)
        return self
    
    def with_state_update(self, state_data: Dict[str, Any]) -> "EventWrapper":
        """Set state update content."""
        self._dict_to_struct(state_data, self._event.state_update)
        return self
    
    def with_metadata(self, metadata: Dict[str, Any]) -> "EventWrapper":
        """Set event metadata."""
        self._dict_to_struct(metadata, self._event.metadata)
        return self
    
    def with_performance_metrics(
        self,
        cpu_percent: Optional[float] = None,
        memory_mb: Optional[float] = None,
        duration_ms: Optional[float] = None,
        throughput: Optional[float] = None,
        latency_ms: Optional[float] = None
    ) -> "EventWrapper":
        """Set performance metrics."""
        metrics = PerformanceMetrics()
        
        if cpu_percent is not None:
            metrics.cpu_percent = cpu_percent
        if memory_mb is not None:
            metrics.memory_mb = memory_mb
        if duration_ms is not None:
            metrics.duration_ms = duration_ms
        if throughput is not None:
            metrics.throughput = throughput
        if latency_ms is not None:
            metrics.latency_ms = latency_ms
        
        self._event.performance.CopyFrom(metrics)
        return self
    
    def with_cost_details(
        self,
        provider: str,
        service: str,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        input_cost: Optional[float] = None,
        output_cost: Optional[float] = None,
        total_cost: Optional[float] = None,
        currency: str = "USD"
    ) -> "EventWrapper":
        """Set cost details."""
        cost = CostDetails()
        cost.provider = provider
        cost.service = service
        cost.currency = currency
        
        if input_tokens is not None:
            cost.input_tokens = input_tokens
        if output_tokens is not None:
            cost.output_tokens = output_tokens
        if total_tokens is not None:
            cost.total_tokens = total_tokens
        if input_cost is not None:
            cost.input_cost = input_cost
        if output_cost is not None:
            cost.output_cost = output_cost
        if total_cost is not None:
            cost.total_cost = total_cost
        
        self._event.cost.CopyFrom(cost)
        return self
    
    # Conversion methods
    
    def to_proto(self) -> Event:
        """Get the underlying proto Event."""
        return self._event
    
    def to_request(self) -> IngestEventRequest:
        """Convert to IngestEventRequest for sending."""
        request = IngestEventRequest()
        request.event.CopyFrom(self._event)
        return request
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to Python dictionary."""
        return MessageToDict(self._event)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return MessageToJson(self._event)
    
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
                struct_value = struct[key]
                self._dict_to_struct(value, struct_value)
            elif isinstance(value, list):
                list_value = struct[key]
                for item in value:
                    if isinstance(item, dict):
                        item_struct = list_value.add()
                        self._dict_to_struct(item, item_struct)
                    else:
                        list_value.append(item)
            else:
                struct[key] = str(value)
    
    # Properties for direct access
    
    @property
    def event_id(self) -> str:
        """Get event ID."""
        return self._event.event_id
    
    @property
    def trace_id(self) -> str:
        """Get trace ID."""
        return self._event.trace_id
    
    @property
    def span_id(self) -> str:
        """Get span ID."""
        return self._event.span_id
    
    @property
    def session_id(self) -> str:
        """Get session ID."""
        return self._event.session_id
    
    @property
    def event_type(self) -> EventType:
        """Get event type."""
        return self._event.type