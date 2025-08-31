"""
OpenAI Agents SDK integration for Chaukas instrumentation.
Uses proto-compliant events with proper distributed tracing.
"""

import asyncio
import logging
from typing import Any, Dict, Optional, List
from functools import wraps

from chaukas.spec.common.v1.events_pb2 import EventStatus

from chaukas.sdk.core.tracer import ChaukasTracer
from chaukas.sdk.core.event_builder import EventBuilder
from chaukas.sdk.core.agent_mapper import AgentMapper

logger = logging.getLogger(__name__)


class OpenAIAgentsWrapper:
    """Wrapper for OpenAI Agents SDK instrumentation using proto events."""
    
    def __init__(self, tracer: ChaukasTracer):
        self.tracer = tracer
        self.event_builder = EventBuilder()
    
    def wrap_agent_run(self, wrapped, instance, args, kwargs):
        """Wrap Agent.run method with proto-compliant events."""
        
        @wraps(wrapped)
        async def async_wrapper(*args, **kwargs):
            # Extract agent info using mapper
            agent_id, agent_name = AgentMapper.map_openai_agent(instance)
            
            with self.tracer.start_span("openai_agent.run") as span:
                span.set_attribute("agent_type", "openai")
                span.set_attribute("agent_name", agent_name)
                
                try:
                    # Send AGENT_START event
                    start_event = self.event_builder.create_agent_start(
                        agent_id=agent_id,
                        agent_name=agent_name,
                        role="openai_agent",
                        instructions=getattr(instance, "instructions", None),
                        tools=[tool.name for tool in getattr(instance, "tools", [])],
                        metadata={
                            "model": getattr(instance, "model", None),
                            "framework": "openai_agents",
                        }
                    )
                    
                    await self.tracer.client.send_event(start_event)
                    
                    # Execute original method
                    result = await wrapped(*args, **kwargs)
                    
                    # Send AGENT_END event
                    end_event = self.event_builder.create_agent_end(
                        agent_id=agent_id,
                        agent_name=agent_name,
                        status=EventStatus.EVENT_STATUS_COMPLETED,
                        metadata={
                            "result_type": type(result).__name__,
                            "messages_count": len(getattr(result, "messages", [])),
                            "framework": "openai_agents",
                        }
                    )
                    
                    await self.tracer.client.send_event(end_event)
                    
                    return result
                    
                except Exception as e:
                    # Send ERROR event
                    error_event = self.event_builder.create_error(
                        error_message=str(e),
                        error_code=type(e).__name__,
                        recoverable=True,
                        agent_id=agent_id,
                        agent_name=agent_name
                    )
                    
                    await self.tracer.client.send_event(error_event)
                    
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
        """Wrap Runner.run_once method with proto-compliant MODEL_INVOCATION events."""
        
        @wraps(wrapped)
        async def async_wrapper(*args, **kwargs):
            # Extract agent info
            agent = getattr(instance, "agent", None)
            if agent:
                agent_id, agent_name = AgentMapper.map_openai_agent(agent)
            else:
                agent_id, agent_name = "unknown", "unknown"
            
            with self.tracer.start_span("openai_runner.run_once") as span:
                span.set_attribute("runner_type", "openai")
                span.set_attribute("agent_name", agent_name)
                
                try:
                    # Send MODEL_INVOCATION_START event
                    if args and len(args) > 0:
                        messages = args[0] if isinstance(args[0], list) else []
                        model = getattr(agent, "model", "unknown") if agent else "unknown"
                        
                        start_event = self.event_builder.create_model_invocation_start(
                            provider="openai",
                            model=model,
                            messages=self._serialize_messages(messages),
                            agent_id=agent_id,
                            agent_name=agent_name,
                            temperature=getattr(agent, "temperature", None) if agent else None,
                            max_tokens=getattr(agent, "max_tokens", None) if agent else None,
                            tools=[{"name": tool.name} for tool in getattr(agent, "tools", [])] if agent else None
                        )
                        
                        await self.tracer.client.send_event(start_event)
                    
                    # Execute original method
                    result = await wrapped(*args, **kwargs)
                    
                    # Send MODEL_INVOCATION_END event
                    if result:
                        model = getattr(agent, "model", "unknown") if agent else "unknown"
                        
                        end_event = self.event_builder.create_model_invocation_end(
                            provider="openai",
                            model=model,
                            response_content=getattr(result, "content", None),
                            tool_calls=self._extract_tool_calls(result),
                            finish_reason=getattr(result, "finish_reason", None),
                            agent_id=agent_id,
                            agent_name=agent_name,
                            # Note: Token counts not readily available in OpenAI Agents
                        )
                        
                        await self.tracer.client.send_event(end_event)
                    
                    return result
                    
                except Exception as e:
                    # Send MODEL_INVOCATION_END with error
                    model = getattr(agent, "model", "unknown") if agent else "unknown"
                    
                    error_event = self.event_builder.create_model_invocation_end(
                        provider="openai",
                        model=model,
                        agent_id=agent_id,
                        agent_name=agent_name,
                        error=str(e)
                    )
                    
                    await self.tracer.client.send_event(error_event)
                    
                    raise
        
        @wraps(wrapped)
        def sync_wrapper(*args, **kwargs):
            return asyncio.create_task(async_wrapper(*args, **kwargs))
        
        if asyncio.iscoroutinefunction(wrapped):
            return async_wrapper
        else:
            return sync_wrapper
    
    def _serialize_messages(self, messages) -> List[Dict[str, str]]:
        """Serialize messages for proto LLMInvocation, handling various formats."""
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
                    serialized.append({
                        "role": "unknown",
                        "content": str(msg)[:1000]
                    })
            
            return serialized
            
        except Exception as e:
            logger.warning(f"Failed to serialize messages: {e}")
            return [{"role": "error", "content": "serialization_failed"}]
    
    def _extract_tool_calls(self, result) -> Optional[List[Dict[str, Any]]]:
        """Extract tool calls from LLM response for proto ToolCall format."""
        try:
            if hasattr(result, "tool_calls") and result.tool_calls:
                return [
                    {
                        "id": getattr(call, "id", None),
                        "name": getattr(call, "function", {}).get("name", "unknown"),
                        "arguments": getattr(call, "function", {}).get("arguments", {}),
                    }
                    for call in result.tool_calls
                ]
            return None
        except Exception as e:
            logger.warning(f"Failed to extract tool calls: {e}")
            return None