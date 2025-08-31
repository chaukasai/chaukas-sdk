"""
Event normalization and unified schema for different agent SDKs.
"""

from typing import Dict, Any, Optional, Union
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel


class EventType(str, Enum):
    """Standard event types for agent instrumentation."""
    
    # LLM Events
    LLM_REQUEST = "llm.request"
    LLM_RESPONSE = "llm.response"
    LLM_STREAM_START = "llm.stream.start"
    LLM_STREAM_CHUNK = "llm.stream.chunk"
    LLM_STREAM_END = "llm.stream.end"
    
    # Tool Events
    TOOL_CALL = "tool.call"
    TOOL_RESPONSE = "tool.response"
    TOOL_ERROR = "tool.error"
    
    # Agent Events
    AGENT_START = "agent.start"
    AGENT_END = "agent.end"
    AGENT_ERROR = "agent.error"
    AGENT_HANDOFF = "agent.handoff"
    
    # Session Events
    SESSION_START = "session.start"
    SESSION_END = "session.end"
    
    # User Interaction Events
    USER_INPUT = "user.input"
    USER_OUTPUT = "user.output"
    
    # MCP Events
    MCP_CALL = "mcp.call"
    MCP_RESPONSE = "mcp.response"
    
    # Guardrail Events
    GUARDRAIL_CHECK = "guardrail.check"
    GUARDRAIL_VIOLATION = "guardrail.violation"
    
    # Artifact Events
    ARTIFACT_CREATE = "artifact.create"
    ARTIFACT_UPDATE = "artifact.update"


class LLMRequestData(BaseModel):
    """Normalized LLM request data."""
    
    provider: str  # openai, google, anthropic, etc.
    model: str
    messages: list
    tools: Optional[list] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = False
    metadata: Optional[Dict[str, Any]] = None


class LLMResponseData(BaseModel):
    """Normalized LLM response data."""
    
    provider: str
    model: str
    content: Optional[str] = None
    tool_calls: Optional[list] = None
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    latency_ms: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class ToolCallData(BaseModel):
    """Normalized tool call data."""
    
    tool_name: str
    arguments: Dict[str, Any]
    call_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ToolResponseData(BaseModel):
    """Normalized tool response data."""
    
    tool_name: str
    call_id: Optional[str] = None
    result: Any = None
    error: Optional[str] = None
    latency_ms: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class AgentData(BaseModel):
    """Normalized agent data."""
    
    agent_id: str
    agent_name: str
    agent_type: str  # openai, google_adk, crewai, custom
    role: Optional[str] = None
    instructions: Optional[str] = None
    tools: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class EventNormalizer:
    """Normalizes events from different agent SDKs into unified schema."""
    
    @staticmethod
    def normalize_openai_agent_event(
        event_type: str,
        raw_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Normalize OpenAI Agents SDK events."""
        if event_type.startswith("llm."):
            return EventNormalizer._normalize_llm_event("openai", raw_data)
        elif event_type.startswith("agent."):
            return EventNormalizer._normalize_agent_event("openai", raw_data)
        elif event_type.startswith("tool."):
            return EventNormalizer._normalize_tool_event("openai", raw_data)
        return raw_data
    
    @staticmethod
    def normalize_google_adk_event(
        event_type: str,
        raw_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Normalize Google ADK events."""
        if event_type.startswith("llm."):
            return EventNormalizer._normalize_llm_event("google", raw_data)
        elif event_type.startswith("agent."):
            return EventNormalizer._normalize_agent_event("google", raw_data)
        elif event_type.startswith("tool."):
            return EventNormalizer._normalize_tool_event("google", raw_data)
        return raw_data
    
    @staticmethod
    def normalize_crewai_event(
        event_type: str,
        raw_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Normalize CrewAI events."""
        if event_type.startswith("llm."):
            return EventNormalizer._normalize_llm_event("crewai", raw_data)
        elif event_type.startswith("agent."):
            return EventNormalizer._normalize_agent_event("crewai", raw_data)
        elif event_type.startswith("tool."):
            return EventNormalizer._normalize_tool_event("crewai", raw_data)
        return raw_data
    
    @staticmethod
    def _normalize_llm_event(provider: str, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize LLM events to unified schema."""
        # This will be implemented based on each SDK's specific structure
        normalized = {
            "provider": provider,
            "model": raw_data.get("model", "unknown"),
            "messages": raw_data.get("messages", []),
            "tools": raw_data.get("tools"),
            "stream": raw_data.get("stream", False),
            "metadata": {
                "original_data": raw_data,
                "sdk_version": raw_data.get("sdk_version"),
            }
        }
        return normalized
    
    @staticmethod
    def _normalize_agent_event(provider: str, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize agent events to unified schema."""
        normalized = {
            "agent_id": raw_data.get("agent_id", str(uuid.uuid4())),
            "agent_name": raw_data.get("name", "unknown"),
            "agent_type": provider,
            "role": raw_data.get("role"),
            "instructions": raw_data.get("instructions"),
            "tools": raw_data.get("tools", []),
            "metadata": {
                "original_data": raw_data,
                "sdk_version": raw_data.get("sdk_version"),
            }
        }
        return normalized
    
    @staticmethod
    def _normalize_tool_event(provider: str, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize tool events to unified schema."""
        normalized = {
            "tool_name": raw_data.get("name", "unknown"),
            "arguments": raw_data.get("arguments", {}),
            "call_id": raw_data.get("call_id"),
            "result": raw_data.get("result"),
            "error": raw_data.get("error"),
            "metadata": {
                "original_data": raw_data,
                "provider": provider,
            }
        }
        return normalized


class ChaukasTracer:
    """Main tracer for creating and managing spans."""
    
    def __init__(self, client: ChaukasClient, session_id: Optional[str] = None):
        self.client = client
        self.default_session_id = session_id or str(uuid.uuid4())
        self.normalizer = EventNormalizer()
    
    def start_span(
        self,
        name: str,
        trace_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Span:
        """Start a new span."""
        current_trace_id = trace_id or _trace_id.get() or str(uuid.uuid4())
        current_parent_span_id = parent_span_id or _span_id.get()
        current_session_id = session_id or _session_id.get() or self.default_session_id
        
        span_id = str(uuid.uuid4())
        
        return Span(
            tracer=self,
            name=name,
            span_id=span_id,
            trace_id=current_trace_id,
            parent_span_id=current_parent_span_id,
            session_id=current_session_id,
        )
    
    def get_current_span_context(self) -> Dict[str, Optional[str]]:
        """Get the current span context."""
        return {
            "session_id": _session_id.get(),
            "trace_id": _trace_id.get(),
            "span_id": _span_id.get(),
            "parent_span_id": _parent_span_id.get(),
        }
    
    async def send_event(
        self,
        event_type: str,
        source: str,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        normalize_for: Optional[str] = None,
    ) -> None:
        """Send a standalone event with current context."""
        # Normalize data if requested
        if normalize_for:
            if normalize_for == "openai":
                data = self.normalizer.normalize_openai_agent_event(event_type, data)
            elif normalize_for == "google":
                data = self.normalizer.normalize_google_adk_event(event_type, data)
            elif normalize_for == "crewai":
                data = self.normalizer.normalize_crewai_event(event_type, data)
        
        context = self.get_current_span_context()
        
        event = self.client.create_event(
            event_type=event_type,
            source=source,
            data=data,
            session_id=context["session_id"] or self.default_session_id,
            trace_id=context["trace_id"] or str(uuid.uuid4()),
            span_id=context["span_id"] or str(uuid.uuid4()),
            parent_span_id=context["parent_span_id"],
            metadata=metadata,
        )
        
        await self.client.send_event(event)