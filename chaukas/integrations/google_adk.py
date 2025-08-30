"""
Google ADK Python integration for Chaukas instrumentation.
"""

import asyncio
import logging
from typing import Any, Dict, Optional
from functools import wraps

from ..core.tracer import ChaukasTracer
from ..core.events import EventType

logger = logging.getLogger(__name__)


class GoogleADKWrapper:
    """Wrapper for Google ADK instrumentation."""
    
    def __init__(self, tracer: ChaukasTracer):
        self.tracer = tracer
    
    def wrap_agent_run(self, wrapped, instance, args, kwargs):
        """Wrap Agent.run method."""
        
        @wraps(wrapped)
        async def async_wrapper(*args, **kwargs):
            with self.tracer.start_span("google_adk.agent.run") as span:
                span.set_attribute("agent_type", "google_adk")
                span.set_attribute("agent_name", getattr(instance, "name", "unknown"))
                
                try:
                    # Extract agent configuration
                    agent_data = {
                        "agent_id": getattr(instance, "_id", str(id(instance))),
                        "agent_name": getattr(instance, "name", "unknown"),
                        "model": getattr(instance, "model", None),
                        "description": getattr(instance, "description", None),
                        "instructions": getattr(instance, "instruction", None),
                        "tools": self._extract_tools(instance),
                    }
                    
                    # Send agent start event
                    await self.tracer.send_event(
                        event_type=EventType.AGENT_START,
                        source="google_adk",
                        data=agent_data,
                        normalize_for="google",
                    )
                    
                    # Execute original method
                    result = await wrapped(*args, **kwargs)
                    
                    # Send agent end event
                    end_data = agent_data.copy()
                    end_data.update({
                        "result_type": type(result).__name__,
                        "result_content": str(result)[:500] if result else None,
                    })
                    
                    await self.tracer.send_event(
                        event_type=EventType.AGENT_END,
                        source="google_adk",
                        data=end_data,
                        normalize_for="google",
                    )
                    
                    return result
                    
                except Exception as e:
                    # Send agent error event
                    error_data = {
                        "agent_id": getattr(instance, "_id", str(id(instance))),
                        "agent_name": getattr(instance, "name", "unknown"),
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }
                    
                    await self.tracer.send_event(
                        event_type=EventType.AGENT_ERROR,
                        source="google_adk",
                        data=error_data,
                        normalize_for="google",
                    )
                    
                    raise
        
        @wraps(wrapped)
        def sync_wrapper(*args, **kwargs):
            return asyncio.create_task(async_wrapper(*args, **kwargs))
        
        if asyncio.iscoroutinefunction(wrapped):
            return async_wrapper
        else:
            return sync_wrapper
    
    def wrap_llm_agent_run(self, wrapped, instance, args, kwargs):
        """Wrap LlmAgent.run method with LLM-specific instrumentation."""
        
        @wraps(wrapped)
        async def async_wrapper(*args, **kwargs):
            with self.tracer.start_span("google_adk.llm_agent.run") as span:
                span.set_attribute("agent_type", "google_adk_llm")
                span.set_attribute("agent_name", getattr(instance, "name", "unknown"))
                span.set_attribute("model", getattr(instance, "model", "unknown"))
                
                try:
                    # Extract LLM request data
                    request_data = {
                        "provider": "google",
                        "model": getattr(instance, "model", "unknown"),
                        "agent_name": getattr(instance, "name", "unknown"),
                        "input": args[0] if args else None,
                        "instructions": getattr(instance, "instruction", None),
                    }
                    
                    # Send LLM request event
                    await self.tracer.send_event(
                        event_type=EventType.LLM_REQUEST,
                        source="google_adk",
                        data=request_data,
                        normalize_for="google",
                    )
                    
                    # Execute original method
                    result = await wrapped(*args, **kwargs)
                    
                    # Send LLM response event
                    response_data = {
                        "provider": "google",
                        "model": getattr(instance, "model", "unknown"),
                        "agent_name": getattr(instance, "name", "unknown"),
                        "content": str(result)[:1000] if result else None,
                        "result_type": type(result).__name__,
                    }
                    
                    await self.tracer.send_event(
                        event_type=EventType.LLM_RESPONSE,
                        source="google_adk",
                        data=response_data,
                        normalize_for="google",
                    )
                    
                    return result
                    
                except Exception as e:
                    # Send LLM error event
                    error_data = {
                        "provider": "google",
                        "model": getattr(instance, "model", "unknown"),
                        "agent_name": getattr(instance, "name", "unknown"),
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }
                    
                    await self.tracer.send_event(
                        event_type=EventType.LLM_RESPONSE,
                        source="google_adk",
                        data=error_data,
                        normalize_for="google",
                    )
                    
                    raise
        
        @wraps(wrapped)
        def sync_wrapper(*args, **kwargs):
            return asyncio.create_task(async_wrapper(*args, **kwargs))
        
        if asyncio.iscoroutinefunction(wrapped):
            return async_wrapper
        else:
            return sync_wrapper
    
    def _extract_tools(self, agent_instance) -> list:
        """Extract tools from agent instance."""
        try:
            tools = getattr(agent_instance, "tools", [])
            if not tools:
                return []
            
            tool_names = []
            for tool in tools:
                if hasattr(tool, "name"):
                    tool_names.append(tool.name)
                elif hasattr(tool, "__name__"):
                    tool_names.append(tool.__name__)
                else:
                    tool_names.append(str(tool))
            
            return tool_names
            
        except Exception as e:
            logger.warning(f"Failed to extract tools: {e}")
            return []