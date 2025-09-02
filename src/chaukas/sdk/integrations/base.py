"""
Base class for all SDK integration wrappers.
Provides common functionality for event creation, sending, and error handling.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, Callable, Tuple
from functools import wraps
from abc import ABC, abstractmethod

from chaukas.spec.common.v1.events_pb2 import Event, EventStatus

from chaukas.sdk.core.tracer import ChaukasTracer
from chaukas.sdk.core.event_builder import EventBuilder
from chaukas.sdk.core.agent_mapper import AgentMapper

logger = logging.getLogger(__name__)


class BaseIntegrationWrapper(ABC):
    """
    Base class for SDK integration wrappers.
    
    Provides common functionality:
    - Event builder and tracer initialization
    - Sync/async event sending
    - Error handling patterns
    - Agent handoff tracking
    - START/END event pairing
    """
    
    def __init__(self, tracer: ChaukasTracer):
        """
        Initialize base wrapper.
        
        Args:
            tracer: ChaukasTracer instance for distributed tracing
        """
        self.tracer = tracer
        self.event_builder = EventBuilder()
        
        # Track last active agent for handoff detection
        self._last_active_agent: Optional[Tuple[str, str]] = None
        
        # Track span IDs for START/END event pairing
        self._active_spans: Dict[str, str] = {}
        
        # Initialize framework-specific components
        self._initialize_framework()
    
    @abstractmethod
    def _initialize_framework(self) -> None:
        """Initialize framework-specific components. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def get_framework_name(self) -> str:
        """Return the name of the framework being integrated."""
        pass
    
    def _send_event_sync(self, event: Event) -> None:
        """
        Helper to send event from synchronous context.
        
        Args:
            event: Proto Event to send
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If event loop is running, schedule as task
                asyncio.create_task(self.tracer.client.send_event(event))
            else:
                # If no loop running, run until complete
                loop.run_until_complete(self.tracer.client.send_event(event))
        except RuntimeError:
            # No event loop exists, create one
            asyncio.run(self.tracer.client.send_event(event))
        except Exception as e:
            logger.error(f"Failed to send event: {e}")
    
    async def _send_event_async(self, event: Event) -> None:
        """
        Helper to send event from asynchronous context.
        
        Args:
            event: Proto Event to send
        """
        try:
            await self.tracer.client.send_event(event)
        except Exception as e:
            logger.error(f"Failed to send event: {e}")
    
    def create_wrapper(self, wrapped: Callable, wrapper_func: Callable) -> Callable:
        """
        Create a wrapper that handles both sync and async functions.
        
        Args:
            wrapped: Original function to wrap
            wrapper_func: Wrapper implementation (should be async)
            
        Returns:
            Appropriate wrapper based on wrapped function type
        """
        @wraps(wrapped)
        async def async_wrapper(*args, **kwargs):
            return await wrapper_func(wrapped, *args, **kwargs)
        
        @wraps(wrapped)
        def sync_wrapper(*args, **kwargs):
            loop = None
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Return a task for the caller to await
                    return asyncio.create_task(async_wrapper(*args, **kwargs))
                else:
                    # Run in the existing loop
                    return loop.run_until_complete(async_wrapper(*args, **kwargs))
            except RuntimeError:
                # No loop exists, create one
                return asyncio.run(async_wrapper(*args, **kwargs))
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(wrapped):
            return async_wrapper
        else:
            return sync_wrapper
    
    def track_agent_handoff(
        self,
        agent_id: str,
        agent_name: str,
        send_event: bool = True
    ) -> Optional[Event]:
        """
        Track agent changes and emit AGENT_HANDOFF event if needed.
        
        Args:
            agent_id: Current agent ID
            agent_name: Current agent name
            send_event: Whether to send the handoff event
            
        Returns:
            AGENT_HANDOFF event if handoff detected, None otherwise
        """
        handoff_event = None
        
        if self._last_active_agent and self._last_active_agent != (agent_id, agent_name):
            last_id, last_name = self._last_active_agent
            
            # Create handoff event
            handoff_event = self.event_builder.create_agent_handoff(
                from_agent_id=last_id,
                from_agent_name=last_name,
                to_agent_id=agent_id,
                to_agent_name=agent_name,
                reason="Task delegation",
                handoff_type="sequential",
                handoff_data={
                    "framework": self.get_framework_name()
                }
            )
            
            if send_event:
                self._send_event_sync(handoff_event)
        
        # Update last active agent
        self._last_active_agent = (agent_id, agent_name)
        
        return handoff_event
    
    def store_span_id(self, key: str, span_id: str) -> None:
        """
        Store span ID for START/END event pairing.
        
        Args:
            key: Unique key for this span (e.g., "agent_123")
            span_id: Span ID to store
        """
        self._active_spans[key] = span_id
    
    def get_span_id(self, key: str) -> Optional[str]:
        """
        Retrieve stored span ID for END event.
        
        Args:
            key: Unique key for this span
            
        Returns:
            Stored span ID if exists, None otherwise
        """
        return self._active_spans.get(key)
    
    def clear_span_id(self, key: str) -> None:
        """
        Clear stored span ID after END event.
        
        Args:
            key: Unique key for this span
        """
        self._active_spans.pop(key, None)
    
    async def handle_agent_execution(
        self,
        wrapped: Callable,
        instance: Any,
        args: tuple,
        kwargs: dict,
        extract_agent_info: Callable
    ) -> Any:
        """
        Common pattern for handling agent execution with START/END events.
        
        Args:
            wrapped: Original method being wrapped
            instance: Agent instance
            args: Method arguments
            kwargs: Method keyword arguments
            extract_agent_info: Function to extract agent ID and name
            
        Returns:
            Result from wrapped method
        """
        # Extract agent information
        agent_id, agent_name = extract_agent_info(instance)
        
        # Check for agent handoff
        self.track_agent_handoff(agent_id, agent_name, send_event=True)
        
        # Create span for this execution
        with self.tracer.start_span(f"{self.get_framework_name()}.agent.execute") as span:
            span.set_attribute("agent_id", agent_id)
            span.set_attribute("agent_name", agent_name)
            span.set_attribute("framework", self.get_framework_name())
            
            try:
                # Send AGENT_START event
                start_event = self.event_builder.create_agent_start(
                    agent_id=agent_id,
                    agent_name=agent_name,
                    role=getattr(instance, "role", "agent"),
                    instructions=self._extract_instructions(instance),
                    tools=self._extract_tools(instance),
                    metadata={
                        "framework": self.get_framework_name()
                    }
                )
                
                await self._send_event_async(start_event)
                
                # Store span ID for END event
                span_key = f"agent_{agent_id}"
                self.store_span_id(span_key, start_event.span_id)
                
                # Execute original method
                result = await wrapped(*args, **kwargs) if asyncio.iscoroutinefunction(wrapped) else wrapped(*args, **kwargs)
                
                # Send AGENT_END event
                end_event = self.event_builder.create_agent_end(
                    agent_id=agent_id,
                    agent_name=agent_name,
                    status=EventStatus.EVENT_STATUS_COMPLETED,
                    span_id=self.get_span_id(span_key),  # Use same span_id as START
                    metadata={
                        "framework": self.get_framework_name(),
                        "result_type": type(result).__name__ if result else None
                    }
                )
                
                await self._send_event_async(end_event)
                
                # Clear span ID
                self.clear_span_id(span_key)
                
                return result
                
            except Exception as e:
                # Send ERROR event
                error_event = self.event_builder.create_error(
                    error_message=str(e),
                    error_code=type(e).__name__,
                    recoverable=self._is_recoverable_error(e),
                    agent_id=agent_id,
                    agent_name=agent_name
                )
                
                await self._send_event_async(error_event)
                
                # Clear span ID on error
                self.clear_span_id(f"agent_{agent_id}")
                
                raise
    
    def _extract_instructions(self, instance: Any) -> Optional[str]:
        """
        Extract instructions from agent instance.
        
        Args:
            instance: Agent instance
            
        Returns:
            Instructions string if found
        """
        # Try common attribute names
        for attr in ["instructions", "instruction", "prompt", "system_prompt", "backstory"]:
            if hasattr(instance, attr):
                value = getattr(instance, attr)
                if value:
                    return str(value)
        return None
    
    def _extract_tools(self, instance: Any) -> list:
        """
        Extract tools from agent instance.
        
        Args:
            instance: Agent instance
            
        Returns:
            List of tool names
        """
        tools = []
        
        # Try to get tools attribute
        if hasattr(instance, "tools"):
            agent_tools = getattr(instance, "tools", [])
            if agent_tools:
                for tool in agent_tools:
                    if hasattr(tool, "name"):
                        tools.append(tool.name)
                    elif hasattr(tool, "__name__"):
                        tools.append(tool.__name__)
                    else:
                        tools.append(str(tool))
        
        return tools
    
    def _is_recoverable_error(self, error: Exception) -> bool:
        """
        Determine if an error is recoverable.
        
        Args:
            error: Exception to check
            
        Returns:
            True if error is potentially recoverable
        """
        error_msg = str(error).lower()
        
        # Non-recoverable patterns
        non_recoverable = [
            "invalid api key",
            "authentication failed",
            "permission denied",
            "not found",
            "invalid",
            "syntax error",
            "type error",
            "value error"
        ]
        
        for pattern in non_recoverable:
            if pattern in error_msg:
                return False
        
        # Recoverable patterns
        recoverable = [
            "rate limit",
            "timeout",
            "connection",
            "temporary",
            "503",
            "429",
            "network",
            "unavailable",
            "retry"
        ]
        
        for pattern in recoverable:
            if pattern in error_msg:
                return True
        
        # Default to recoverable for unknown errors
        return True