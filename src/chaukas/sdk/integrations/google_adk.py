"""
Google ADK Python integration for Chaukas instrumentation.
Uses proto-compliant events with proper MODEL_INVOCATION tracking.
"""

import asyncio
import logging
from functools import wraps
from typing import Any, Dict, List, Optional

from chaukas.spec.common.v1.events_pb2 import EventStatus

from chaukas.sdk.core.agent_mapper import AgentMapper
from chaukas.sdk.core.event_builder import EventBuilder
from chaukas.sdk.core.tracer import ChaukasTracer

logger = logging.getLogger(__name__)


class GoogleADKWrapper:
    """Wrapper for Google ADK instrumentation using proto events."""

    def __init__(self, tracer: ChaukasTracer):
        self.tracer = tracer
        self.event_builder = EventBuilder()

    def wrap_agent_run(self, wrapped, instance, args, kwargs):
        """Wrap Agent.run method."""

        @wraps(wrapped)
        async def async_wrapper(*args, **kwargs):
            with self.tracer.start_span("google_adk.agent.run") as span:
                span.set_attribute("agent_type", "google_adk")
                span.set_attribute("agent_name", getattr(instance, "name", "unknown"))

                try:
                    # Extract agent info using mapper
                    agent_id, agent_name = AgentMapper.map_google_adk_agent(instance)

                    # Send AGENT_START event
                    start_event = self.event_builder.create_agent_start(
                        agent_id=agent_id,
                        agent_name=agent_name,
                        role="google_adk_agent",
                        instructions=getattr(instance, "instruction", None),
                        tools=self._extract_tools(instance),
                        metadata={
                            "framework": "google_adk",
                            "model": getattr(instance, "model", None),
                            "description": getattr(instance, "description", None),
                        },
                    )

                    await self.tracer.client.send_event(start_event)

                    # Store span_id for AGENT_END event
                    agent_span_id = start_event.span_id

                    # Execute original method
                    result = await wrapped(*args, **kwargs)

                    # Send AGENT_END event with same span_id as START
                    end_event = self.event_builder.create_agent_end(
                        agent_id=agent_id,
                        agent_name=agent_name,
                        status=EventStatus.EVENT_STATUS_COMPLETED,
                        span_id=agent_span_id,  # Use same span_id as AGENT_START
                        metadata={
                            "framework": "google_adk",
                            "result_type": type(result).__name__,
                            "result_content": str(result)[:500] if result else None,
                        },
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
                        agent_name=agent_name,
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

    def wrap_llm_agent_run(self, wrapped, instance, args, kwargs):
        """Wrap LlmAgent.run method with proto MODEL_INVOCATION events."""

        @wraps(wrapped)
        async def async_wrapper(*args, **kwargs):
            # Extract agent info using mapper
            agent_id, agent_name = AgentMapper.map_google_adk_agent(instance)

            with self.tracer.start_span("google_adk.llm_agent.run") as span:
                span.set_attribute("agent_type", "google_adk_llm")
                span.set_attribute("agent_name", agent_name)
                span.set_attribute("model", getattr(instance, "model", "unknown"))

                try:
                    # Send MODEL_INVOCATION_START event
                    model = getattr(instance, "model", "unknown")
                    input_text = args[0] if args else ""

                    start_event = self.event_builder.create_model_invocation_start(
                        provider="google",
                        model=model,
                        messages=[{"role": "user", "content": str(input_text)}],
                        agent_id=agent_id,
                        agent_name=agent_name,
                    )

                    await self.tracer.client.send_event(start_event)

                    # Store span_id for MODEL_INVOCATION_END event
                    model_span_id = start_event.span_id

                    # Execute original method
                    result = await wrapped(*args, **kwargs)

                    # Send MODEL_INVOCATION_END event with same span_id
                    end_event = self.event_builder.create_model_invocation_end(
                        provider="google",
                        model=model,
                        response_content=str(result)[:1000] if result else None,
                        span_id=model_span_id,  # Use same span_id as START
                        agent_id=agent_id,
                        agent_name=agent_name,
                        # Note: Token counts typically not available in Google ADK
                    )

                    await self.tracer.client.send_event(end_event)

                    return result

                except Exception as e:
                    # Send MODEL_INVOCATION_END with error
                    model = getattr(instance, "model", "unknown")

                    error_event = self.event_builder.create_model_invocation_end(
                        provider="google",
                        model=model,
                        span_id=(
                            model_span_id if "model_span_id" in locals() else None
                        ),  # Use same span if available
                        agent_id=agent_id,
                        agent_name=agent_name,
                        error=str(e),
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
