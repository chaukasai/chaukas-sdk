"""
Distributed tracing infrastructure for Chaukas SDK.

Hierarchy: session_id -> trace_id -> span_id -> parent_span_id
- session_id: Top-level, can encompass multiple traces
- trace_id: One per end-to-end execution across agents/systems
- span_id: Unique for each unit of work
- parent_span_id: Links to parent span
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List, TYPE_CHECKING
from contextvars import ContextVar
from datetime import datetime, timezone

if TYPE_CHECKING:
    from chaukas.sdk.core.client import ChaukasClient

from chaukas.sdk.utils.uuid7 import generate_uuid7

logger = logging.getLogger(__name__)

# Context variables for distributed tracing
_session_id: ContextVar[Optional[str]] = ContextVar("session_id", default=None)
_trace_id: ContextVar[Optional[str]] = ContextVar("trace_id", default=None)
_span_id: ContextVar[Optional[str]] = ContextVar("span_id", default=None)
_parent_span_id: ContextVar[Optional[str]] = ContextVar("parent_span_id", default=None)


class Span:
    """Represents a single span in the distributed trace."""
    
    def __init__(
        self,
        tracer: "ChaukasTracer",
        name: str,
        span_id: str,
        trace_id: str,
        parent_span_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        self.tracer = tracer
        self.name = name
        self.span_id = span_id
        self.trace_id = trace_id
        self.parent_span_id = parent_span_id
        self.session_id = session_id or _session_id.get()
        self.start_time = datetime.now(timezone.utc)
        self.end_time: Optional[datetime] = None
        self.status = "active"
        self.attributes: Dict[str, Any] = {}
        self.events: List[Dict[str, Any]] = []
    
    def set_attribute(self, key: str, value: Any) -> None:
        """Set an attribute on the span."""
        self.attributes[key] = value
    
    def add_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Add an event to the span."""
        self.events.append({
            "timestamp": datetime.now(timezone.utc),
            "event_type": event_type,
            "data": data,
        })
    
    def set_status(self, status: str, description: Optional[str] = None) -> None:
        """Set the span status."""
        self.status = status
        if description:
            self.attributes["status_description"] = description
    
    def finish(self, error: Optional[Exception] = None) -> None:
        """Finish the span and send event."""
        if self.end_time:
            return  # Already finished
        
        self.end_time = datetime.now(timezone.utc)
        
        if error:
            self.status = "error"
            self.attributes["error"] = str(error)
            self.attributes["error_type"] = type(error).__name__
        elif self.status == "active":
            self.status = "ok"
        
        # Create span event
        event_data = {
            "span_name": self.name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_ms": (self.end_time - self.start_time).total_seconds() * 1000,
            "status": self.status,
            "attributes": self.attributes,
            "events": self.events,
        }
        
        event = self.tracer.client.create_event(
            event_type="span",
            source="chaukas-sdk",
            data=event_data,
            session_id=self.session_id or "",
            trace_id=self.trace_id,
            span_id=self.span_id,
            parent_span_id=self.parent_span_id,
        )
        
        # Send event asynchronously
        asyncio.create_task(self.tracer.client.send_event(event))
    
    def __enter__(self):
        # Set context variables
        self._session_token = _session_id.set(self.session_id)
        self._trace_token = _trace_id.set(self.trace_id)
        self._span_token = _span_id.set(self.span_id)
        self._parent_token = _parent_span_id.set(self.parent_span_id)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Reset context variables
        _session_id.reset(self._session_token)
        _trace_id.reset(self._trace_token)
        _span_id.reset(self._span_token)
        _parent_span_id.reset(self._parent_token)
        
        # Finish span with error if exception occurred
        error = exc_val if exc_type else None
        self.finish(error)


class ChaukasTracer:
    """
    Main tracer for creating and managing spans.
    
    Manages the distributed tracing hierarchy:
    session -> trace -> span -> parent_span
    """
    
    def __init__(self, client: "ChaukasClient", session_id: Optional[str] = None):
        self.client = client
        self.default_session_id = session_id or generate_uuid7()
    
    def start_session(self, session_id: Optional[str] = None) -> str:
        """
        Start a new session.
        
        Args:
            session_id: Optional session ID, generates UUID7 if not provided
            
        Returns:
            The session ID
        """
        sid = session_id or generate_uuid7()
        _session_id.set(sid)
        return sid
    
    def start_trace(self, trace_id: Optional[str] = None) -> str:
        """
        Start a new trace within the current session.
        
        Args:
            trace_id: Optional trace ID, generates UUID7 if not provided
            
        Returns:
            The trace ID
        """
        tid = trace_id or generate_uuid7()
        _trace_id.set(tid)
        return tid
    
    def start_span(
        self,
        name: str,
        trace_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Span:
        """
        Start a new span.
        
        Args:
            name: Name of the span (e.g., "agent.execute", "llm.call")
            trace_id: Trace ID (uses context or generates new one)
            parent_span_id: Parent span ID (uses context if available)
            session_id: Session ID (uses context or default)
        """
        # Use context if available, otherwise generate new IDs
        current_trace_id = trace_id or _trace_id.get() or generate_uuid7()
        current_parent_span_id = parent_span_id or _span_id.get()
        current_session_id = session_id or _session_id.get() or self.default_session_id
        
        span_id = generate_uuid7()
        
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
    ) -> None:
        """Send a standalone event with current context."""
        context = self.get_current_span_context()
        
        event = self.client.create_event(
            event_type=event_type,
            source=source,
            data=data,
            session_id=context["session_id"] or self.default_session_id,
            trace_id=context["trace_id"] or generate_uuid7(),
            span_id=context["span_id"] or generate_uuid7(),
            parent_span_id=context["parent_span_id"],
            metadata=metadata,
        )
        
        await self.client.send_event(event)