"""
OpenAI Agents SDK integration for Chaukas instrumentation.
"""

import asyncio
import logging
from typing import Any, Dict, Optional
from functools import wraps

from ..core.tracer import ChaukasTracer
from ..core.events import EventType

logger = logging.getLogger(__name__)


class OpenAIAgentsWrapper:
    """Wrapper for OpenAI Agents SDK instrumentation."""
    
    def __init__(self, tracer: ChaukasTracer):
        self.tracer = tracer
    
    def wrap_agent_run(self, wrapped, instance, args, kwargs):
        """Wrap Agent.run method."""
        
        @wraps(wrapped)
        async def async_wrapper(*args, **kwargs):
            with self.tracer.start_span("openai_agent.run") as span:
                span.set_attribute("agent_type", "openai")
                span.set_attribute("agent_name", getattr(instance, "name", "unknown"))
                
                try:
                    # Extract agent configuration
                    agent_data = {
                        "agent_id": getattr(instance, "id", str(id(instance))),
                        "agent_name": getattr(instance, "name", "unknown"),
                        "instructions": getattr(instance, "instructions", None),
                        "model": getattr(instance, "model", None),
                        "tools": [tool.name for tool in getattr(instance, "tools", [])],
                    }
                    
                    # Send agent start event
                    await self.tracer.send_event(
                        event_type=EventType.AGENT_START,
                        source="openai_agents",
                        data=agent_data,
                        normalize_for="openai",
                    )
                    
                    # Execute original method
                    result = await wrapped(*args, **kwargs)
                    
                    # Send agent end event
                    end_data = agent_data.copy()
                    end_data.update({
                        "result_type": type(result).__name__,
                        "messages_count": len(getattr(result, "messages", [])),
                    })
                    
                    await self.tracer.send_event(
                        event_type=EventType.AGENT_END,
                        source="openai_agents",
                        data=end_data,
                        normalize_for="openai",
                    )
                    
                    return result
                    
                except Exception as e:
                    # Send agent error event
                    error_data = {
                        "agent_id": getattr(instance, "id", str(id(instance))),
                        "agent_name": getattr(instance, "name", "unknown"),
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }
                    
                    await self.tracer.send_event(
                        event_type=EventType.AGENT_ERROR,
                        source="openai_agents",
                        data=error_data,
                        normalize_for="openai",
                    )
                    
                    raise
        
        @wraps(wrapped)
        def sync_wrapper(*args, **kwargs):
            # For sync version, create async task
            return asyncio.create_task(async_wrapper(*args, **kwargs))
        
        # Check if original is async
        if asyncio.iscoroutinefunction(wrapped):
            return async_wrapper
        else:
            return sync_wrapper
    
    def wrap_runner_run_once(self, wrapped, instance, args, kwargs):
        """Wrap Runner.run_once method."""
        
        @wraps(wrapped)
        async def async_wrapper(*args, **kwargs):
            with self.tracer.start_span("openai_runner.run_once") as span:
                span.set_attribute("runner_type", "openai")
                
                try:
                    # Extract runner/agent info
                    agent = getattr(instance, "agent", None)
                    agent_name = getattr(agent, "name", "unknown") if agent else "unknown"
                    
                    span.set_attribute("agent_name", agent_name)
                    
                    # Send LLM request event (if we can detect it)
                    if args and len(args) > 0:
                        messages = args[0] if isinstance(args[0], list) else []
                        
                        llm_request_data = {
                            "provider": "openai",
                            "model": getattr(agent, "model", "unknown") if agent else "unknown",
                            "messages": self._serialize_messages(messages),
                            "agent_name": agent_name,
                        }
                        
                        await self.tracer.send_event(
                            event_type=EventType.LLM_REQUEST,
                            source="openai_agents",
                            data=llm_request_data,
                            normalize_for="openai",
                        )
                    
                    # Execute original method
                    result = await wrapped(*args, **kwargs)
                    
                    # Send LLM response event
                    if result:
                        llm_response_data = {
                            "provider": "openai",
                            "model": getattr(agent, "model", "unknown") if agent else "unknown",
                            "content": getattr(result, "content", None),
                            "tool_calls": self._extract_tool_calls(result),
                            "finish_reason": getattr(result, "finish_reason", None),
                            "agent_name": agent_name,
                        }
                        
                        await self.tracer.send_event(
                            event_type=EventType.LLM_RESPONSE,
                            source="openai_agents",
                            data=llm_response_data,
                            normalize_for="openai",
                        )
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"Error in runner.run_once wrapper: {e}")
                    raise
        
        @wraps(wrapped)
        def sync_wrapper(*args, **kwargs):
            return asyncio.create_task(async_wrapper(*args, **kwargs))
        
        if asyncio.iscoroutinefunction(wrapped):
            return async_wrapper
        else:
            return sync_wrapper
    
    def _serialize_messages(self, messages) -> list:
        """Serialize messages for logging, handling various formats."""
        try:
            if not messages:
                return []
            
            serialized = []
            for msg in messages[:10]:  # Limit to first 10 messages
                if hasattr(msg, "__dict__"):
                    serialized.append({
                        "role": getattr(msg, "role", "unknown"),
                        "content": str(getattr(msg, "content", ""))[:1000],  # Truncate content
                    })
                elif isinstance(msg, dict):
                    serialized.append({
                        "role": msg.get("role", "unknown"),
                        "content": str(msg.get("content", ""))[:1000],
                    })
                else:
                    serialized.append({"content": str(msg)[:1000]})
            
            return serialized
            
        except Exception as e:
            logger.warning(f"Failed to serialize messages: {e}")
            return [{"error": "serialization_failed"}]
    
    def _extract_tool_calls(self, result) -> Optional[list]:
        """Extract tool calls from LLM response."""
        try:
            if hasattr(result, "tool_calls") and result.tool_calls:
                return [
                    {
                        "id": getattr(call, "id", None),
                        "function": getattr(call, "function", {}).get("name"),
                        "arguments": getattr(call, "function", {}).get("arguments"),
                    }
                    for call in result.tool_calls
                ]
            return None
        except Exception as e:
            logger.warning(f"Failed to extract tool calls: {e}")
            return None