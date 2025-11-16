"""
OpenAI Agents SDK integration for Chaukas instrumentation.
Provides 100% event coverage using hooks and monkey patching.
"""

import asyncio
import functools
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

# Optional performance monitoring
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# OpenAI Agents SDK tool types for proper type checking
try:
    from agents.tool import (
        CodeInterpreterTool,
        ComputerTool,
        FileSearchTool,
        HostedMCPTool,
        LocalShellTool,
        WebSearchTool,
    )

    OPENAI_TOOLS_AVAILABLE = True
except ImportError:
    # Fallback if specific tool types aren't available
    OPENAI_TOOLS_AVAILABLE = False
    FileSearchTool = None
    WebSearchTool = None
    CodeInterpreterTool = None
    HostedMCPTool = None
    LocalShellTool = None
    ComputerTool = None

# OpenAI SDK exception types for proper error handling
try:
    from openai import (
        APIConnectionError,
        APIError,
        AuthenticationError,
        RateLimitError,
        Timeout,
    )

    OPENAI_EXCEPTIONS_AVAILABLE = True
except ImportError:
    # Fallback if OpenAI SDK exceptions aren't available
    OPENAI_EXCEPTIONS_AVAILABLE = False
    APIError = Exception
    RateLimitError = Exception
    APIConnectionError = Exception
    Timeout = Exception
    AuthenticationError = Exception

from chaukas.spec.common.v1.events_pb2 import EventStatus

from chaukas.sdk.core.agent_mapper import AgentMapper
from chaukas.sdk.core.event_builder import EventBuilder
from chaukas.sdk.core.tracer import ChaukasTracer

logger = logging.getLogger(__name__)

# Constants for OpenAI integration
FRAMEWORK_NAME = "openai_agents"
PROVIDER_NAME = "openai"
MCP_PROTOCOL_VERSION = "1.0"

# Role constants for message formatting
ROLE_SYSTEM = "system"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"

# Default values
DEFAULT_AGENT_NAME = "unnamed_agent"
DEFAULT_MODEL_NAME = "unknown"


class OpenAIAgentsWrapper:
    """Wrapper for OpenAI Agents SDK instrumentation using proto events."""

    def __init__(self, tracer: ChaukasTracer):
        self.tracer = tracer
        self.event_builder = EventBuilder()
        self._session_active = False
        self._session_span_id = None
        self._start_time = None
        self._start_metrics = None
        self._agent_stack = []  # Stack of (agent_id, agent_name, span_id) tuples
        self._original_runner_run = None
        self._original_runner_run_sync = None
        self._original_runner_run_streamed = None
        # MCP patching
        self._original_mcp_get_prompt = None
        self._original_mcp_call_tool = None
        # Track agent state for STATE_UPDATE events
        self._agent_states = {}

    def patch_runner(self):
        """Apply patches to OpenAI Runner methods."""
        try:
            from agents import Runner

            # Store original methods
            self._original_runner_run = Runner.run
            self._original_runner_run_sync = Runner.run_sync
            self._original_runner_run_streamed = Runner.run_streamed

            wrapper = self

            # Patch async run method
            @functools.wraps(wrapper._original_runner_run)
            async def instrumented_run(starting_agent, input=None, **kwargs):
                """Instrumented version of Runner.run()."""
                # Handle both 'input' and 'input_data' parameter names
                input_data = (
                    input if input is not None else kwargs.pop("input_data", None)
                )

                # Start session if not active
                if not wrapper._session_active:
                    wrapper._start_session(starting_agent)

                # Send INPUT_RECEIVED event
                input_event = wrapper.event_builder.create_input_received(
                    content=str(input_data)[:1000] if input_data else "",
                    metadata={
                        "method": "Runner.run",
                        "agent": (
                            starting_agent.name
                            if hasattr(starting_agent, "name")
                            else None
                        ),
                    },
                )
                await wrapper.tracer.client.send_event(input_event)

                # Inject our custom hooks
                hooks = kwargs.get("hooks")
                custom_hooks = wrapper.create_custom_hooks()

                # Merge with existing hooks if provided
                if hooks:
                    # Chain our hooks with user's hooks
                    merged_hooks = wrapper.merge_hooks(hooks, custom_hooks)
                    kwargs["hooks"] = merged_hooks
                else:
                    kwargs["hooks"] = custom_hooks

                try:
                    # Execute original method
                    result = await wrapper._original_runner_run(
                        starting_agent, input_data, **kwargs
                    )

                    # Send OUTPUT_EMITTED event
                    output_event = wrapper.event_builder.create_output_emitted(
                        content=str(result)[:1000] if result else "No output",
                        metadata={
                            "method": "Runner.run",
                            "agent": (
                                starting_agent.name
                                if hasattr(starting_agent, "name")
                                else None
                            ),
                        },
                    )
                    await wrapper.tracer.client.send_event(output_event)

                    return result

                except Exception as e:
                    # Handle errors
                    await wrapper._handle_error(e, starting_agent)
                    raise
                finally:
                    # End session if this was the first call
                    if wrapper._session_active and wrapper._session_span_id:
                        await wrapper._end_session()

            # Patch sync run method
            @functools.wraps(wrapper._original_runner_run_sync)
            def instrumented_run_sync(starting_agent, input=None, **kwargs):
                """Instrumented version of Runner.run_sync()."""
                # Handle both 'input' and 'input_data' parameter names
                input_data = (
                    input if input is not None else kwargs.pop("input_data", None)
                )

                # Start session if not active
                if not wrapper._session_active:
                    wrapper._start_session(starting_agent)

                # Send INPUT_RECEIVED event
                input_event = wrapper.event_builder.create_input_received(
                    content=str(input_data)[:1000] if input_data else "",
                    metadata={
                        "method": "Runner.run_sync",
                        "agent": (
                            starting_agent.name
                            if hasattr(starting_agent, "name")
                            else None
                        ),
                    },
                )
                wrapper._send_event_sync(input_event)

                # Inject our custom hooks
                hooks = kwargs.get("hooks")
                custom_hooks = wrapper.create_custom_hooks()

                if hooks:
                    merged_hooks = wrapper.merge_hooks(hooks, custom_hooks)
                    kwargs["hooks"] = merged_hooks
                else:
                    kwargs["hooks"] = custom_hooks

                try:
                    # Execute original method
                    result = wrapper._original_runner_run_sync(
                        starting_agent, input_data, **kwargs
                    )

                    # Send OUTPUT_EMITTED event
                    output_event = wrapper.event_builder.create_output_emitted(
                        content=str(result)[:1000] if result else "No output",
                        metadata={
                            "method": "Runner.run_sync",
                            "agent": (
                                starting_agent.name
                                if hasattr(starting_agent, "name")
                                else None
                            ),
                        },
                    )
                    wrapper._send_event_sync(output_event)

                    return result

                except Exception as e:
                    # Handle errors synchronously
                    wrapper._handle_error_sync(e, starting_agent)
                    raise
                finally:
                    # End session if this was the first call
                    if wrapper._session_active and wrapper._session_span_id:
                        wrapper._end_session_sync()

            # Patch streamed run method
            @functools.wraps(wrapper._original_runner_run_streamed)
            async def instrumented_run_streamed(starting_agent, input=None, **kwargs):
                """Instrumented version of Runner.run_streamed()."""
                # Handle both 'input' and 'input_data' parameter names
                input_data = (
                    input if input is not None else kwargs.pop("input_data", None)
                )

                # Start session if not active
                if not wrapper._session_active:
                    wrapper._start_session(starting_agent)

                # Send INPUT_RECEIVED event
                input_event = wrapper.event_builder.create_input_received(
                    content=str(input_data)[:1000] if input_data else "",
                    metadata={
                        "method": "Runner.run_streamed",
                        "agent": (
                            starting_agent.name
                            if hasattr(starting_agent, "name")
                            else None
                        ),
                    },
                )
                await wrapper.tracer.client.send_event(input_event)

                # Inject our custom hooks
                hooks = kwargs.get("hooks")
                custom_hooks = wrapper.create_custom_hooks()

                if hooks:
                    merged_hooks = wrapper.merge_hooks(hooks, custom_hooks)
                    kwargs["hooks"] = merged_hooks
                else:
                    kwargs["hooks"] = custom_hooks

                try:
                    # Execute original method - returns an async generator
                    async for chunk in wrapper._original_runner_run_streamed(
                        starting_agent, input_data, **kwargs
                    ):
                        yield chunk

                    # Send OUTPUT_EMITTED event for streamed response
                    output_event = wrapper.event_builder.create_output_emitted(
                        content="[Streamed response completed]",
                        metadata={
                            "method": "Runner.run_streamed",
                            "agent": (
                                starting_agent.name
                                if hasattr(starting_agent, "name")
                                else None
                            ),
                        },
                    )
                    await wrapper.tracer.client.send_event(output_event)

                except Exception as e:
                    await wrapper._handle_error(e, starting_agent)
                    raise
                finally:
                    if wrapper._session_active and wrapper._session_span_id:
                        await wrapper._end_session()

            # Apply patches
            Runner.run = instrumented_run
            Runner.run_sync = instrumented_run_sync
            Runner.run_streamed = instrumented_run_streamed

            logger.info("Successfully patched OpenAI Runner methods")

            # Emit SYSTEM_EVENT for successful patching
            self._emit_system_event_sync(
                "OpenAI Runner methods successfully patched", "INFO"
            )

            return True

        except ImportError:
            logger.warning("OpenAI Agents SDK not installed, skipping Runner patching")
            return False
        except Exception as e:
            logger.error(f"Failed to patch Runner: {e}")
            return False

    def patch_mcp_server(self):
        """Apply patches to MCP Server methods to capture MCP_CALL events."""
        logger.debug("patch_mcp_server called")
        try:
            from agents.mcp import MCPServer
            from agents.mcp.server import _MCPServerWithClientSession

            logger.debug(f"Successfully imported MCPServer: {MCPServer}")
            logger.debug(
                f"Successfully imported _MCPServerWithClientSession: {_MCPServerWithClientSession}"
            )
        except (ImportError, AttributeError) as e:
            logger.debug(f"MCP not available, skipping MCP server patching: {e}")
            return False

        try:
            # Store original methods from _MCPServerWithClientSession (used by MCPServerStreamableHttp)
            self._original_mcp_get_prompt = _MCPServerWithClientSession.get_prompt
            self._original_mcp_call_tool = _MCPServerWithClientSession.call_tool

            wrapper = self

            # Patch get_prompt method
            @functools.wraps(wrapper._original_mcp_get_prompt)
            async def instrumented_get_prompt(self, name: str, arguments: dict = None):
                """Instrumented version of MCPServer.get_prompt()."""
                start_time = time.time()
                server_name = self.name if hasattr(self, "name") else "mcp_server"
                server_url = (
                    getattr(self, "url", None)
                    or getattr(self, "_url", None)
                    or "mcp://local"
                )

                # Send MCP_CALL_START event
                mcp_start = wrapper.event_builder.create_mcp_call_start(
                    server_name=server_name,
                    server_url=str(server_url),
                    operation="get_prompt",
                    method="get_prompt",
                    request={"prompt_name": name, "arguments": arguments or {}},
                    protocol_version=MCP_PROTOCOL_VERSION,
                )
                await wrapper.tracer.client.send_event(mcp_start)

                try:
                    # Execute original method
                    result = await wrapper._original_mcp_get_prompt(
                        self, name, arguments
                    )

                    # Calculate execution time
                    execution_time_ms = (time.time() - start_time) * 1000

                    # Send MCP_CALL_END event
                    mcp_end = wrapper.event_builder.create_mcp_call_end(
                        server_name=server_name,
                        server_url=str(server_url),
                        operation="get_prompt",
                        method="get_prompt",
                        response={
                            "prompt_name": name,
                            "message_count": (
                                len(result.messages)
                                if hasattr(result, "messages")
                                else 0
                            ),
                        },
                        execution_time_ms=execution_time_ms,
                        error=None,
                        span_id=mcp_start.span_id,
                    )
                    await wrapper.tracer.client.send_event(mcp_end)

                    return result

                except Exception as e:
                    # Calculate execution time
                    execution_time_ms = (time.time() - start_time) * 1000

                    # Send MCP_CALL_END event with error
                    mcp_end = wrapper.event_builder.create_mcp_call_end(
                        server_name=server_name,
                        server_url=str(server_url),
                        operation="get_prompt",
                        method="get_prompt",
                        response={},
                        execution_time_ms=execution_time_ms,
                        error=str(e),
                        span_id=mcp_start.span_id,
                    )
                    await wrapper.tracer.client.send_event(mcp_end)
                    raise

            # Patch call_tool method
            @functools.wraps(wrapper._original_mcp_call_tool)
            async def instrumented_call_tool(
                self, tool_name: str, arguments: dict = None
            ):
                """Instrumented version of MCPServer.call_tool()."""
                start_time = time.time()
                server_name = self.name if hasattr(self, "name") else "mcp_server"
                server_url = (
                    getattr(self, "url", None)
                    or getattr(self, "_url", None)
                    or "mcp://local"
                )

                # Send MCP_CALL_START event
                mcp_start = wrapper.event_builder.create_mcp_call_start(
                    server_name=server_name,
                    server_url=str(server_url),
                    operation="call_tool",
                    method="call_tool",
                    request={"tool_name": tool_name, "arguments": arguments or {}},
                    protocol_version=MCP_PROTOCOL_VERSION,
                )
                await wrapper.tracer.client.send_event(mcp_start)

                try:
                    # Execute original method
                    result = await wrapper._original_mcp_call_tool(
                        self, tool_name, arguments
                    )

                    # Calculate execution time
                    execution_time_ms = (time.time() - start_time) * 1000

                    # Send MCP_CALL_END event
                    response_data = {
                        "tool_name": tool_name,
                        "content_count": (
                            len(result.content) if hasattr(result, "content") else 0
                        ),
                    }
                    if hasattr(result, "content") and result.content:
                        # Include first content item (truncated)
                        first_content = result.content[0]
                        if hasattr(first_content, "text"):
                            response_data["preview"] = str(first_content.text)[:200]

                    mcp_end = wrapper.event_builder.create_mcp_call_end(
                        server_name=server_name,
                        server_url=str(server_url),
                        operation="call_tool",
                        method="call_tool",
                        response=response_data,
                        execution_time_ms=execution_time_ms,
                        error=None,
                        span_id=mcp_start.span_id,
                    )
                    await wrapper.tracer.client.send_event(mcp_end)

                    return result

                except Exception as e:
                    # Calculate execution time
                    execution_time_ms = (time.time() - start_time) * 1000

                    # Send MCP_CALL_END event with error
                    mcp_end = wrapper.event_builder.create_mcp_call_end(
                        server_name=server_name,
                        server_url=str(server_url),
                        operation="call_tool",
                        method="call_tool",
                        response={},
                        execution_time_ms=execution_time_ms,
                        error=str(e),
                        span_id=mcp_start.span_id,
                    )
                    await wrapper.tracer.client.send_event(mcp_end)
                    raise

            # Apply patches to _MCPServerWithClientSession (which MCPServerStreamableHttp inherits from)
            _MCPServerWithClientSession.get_prompt = instrumented_get_prompt
            _MCPServerWithClientSession.call_tool = instrumented_call_tool

            logger.info(
                "Successfully patched MCP Server methods (_MCPServerWithClientSession)"
            )
            return True

        except ImportError:
            logger.debug("MCP not available, skipping MCP server patching")
            return False
        except Exception as e:
            logger.error(f"Failed to patch MCP Server: {e}")
            return False

    def create_custom_hooks(self):
        """Create custom RunHooks implementation for event capture."""
        from agents.lifecycle import RunHooksBase

        wrapper = self

        class ChaukasRunHooks(RunHooksBase):
            """Custom hooks for Chaukas event capture."""

            async def on_agent_start(self, context, agent):
                """Called when an agent starts execution."""
                try:
                    # Extract agent info
                    agent_id = agent.name if hasattr(agent, "name") else str(id(agent))
                    agent_name = (
                        agent.name if hasattr(agent, "name") else "unnamed_agent"
                    )

                    # Check if this is first time seeing this agent
                    is_first_run = agent_id not in wrapper._agent_states

                    # Track agent state changes and emit STATE_UPDATE (always emit on first run or changes)
                    state_diff = wrapper._track_agent_state(agent_id, agent)
                    if state_diff:
                        await wrapper._emit_state_update(
                            agent_id, agent_name, state_diff
                        )

                    # Emit SYSTEM_EVENT for agent initialization (first run only)
                    if is_first_run:
                        await wrapper._emit_system_event(
                            f"Agent '{agent_name}' initialized with model {agent.model if hasattr(agent, 'model') else 'unknown'}",
                            "INFO",
                            agent_id=agent_id,
                            agent_name=agent_name,
                        )

                    # Send AGENT_START event
                    agent_start = wrapper.event_builder.create_agent_start(
                        agent_id=agent_id,
                        agent_name=agent_name,
                        role="assistant",
                        instructions=(
                            agent.instructions
                            if hasattr(agent, "instructions")
                            else None
                        ),
                        tools=[
                            tool.name if hasattr(tool, "name") else str(tool)
                            for tool in (agent.tools if hasattr(agent, "tools") else [])
                        ],
                        metadata={
                            "model": agent.model if hasattr(agent, "model") else None,
                            "framework": FRAMEWORK_NAME,
                        },
                    )
                    await wrapper.tracer.client.send_event(agent_start)

                    # Push agent onto stack for hierarchical tracking
                    wrapper._agent_stack.append(
                        (agent_id, agent_name, agent_start.span_id)
                    )
                except Exception as e:
                    logger.error(f"Error in on_agent_start hook: {e}")

            async def on_agent_end(self, context, agent, output):
                """Called when an agent ends execution."""
                try:
                    agent_id = agent.name if hasattr(agent, "name") else str(id(agent))
                    agent_name = (
                        agent.name if hasattr(agent, "name") else "unnamed_agent"
                    )

                    # Find and pop the agent from the stack
                    span_id = None
                    for i, (
                        stack_agent_id,
                        stack_agent_name,
                        stack_span_id,
                    ) in enumerate(wrapper._agent_stack):
                        if stack_agent_id == agent_id:
                            span_id = stack_span_id
                            wrapper._agent_stack.pop(i)
                            break

                    # If not found in stack, log warning but continue
                    if span_id is None:
                        logger.warning(
                            f"Agent {agent_name} not found in stack during on_agent_end"
                        )
                        # Create a new span_id as fallback
                        span_id = wrapper.event_builder._generate_span_id()

                    # Send AGENT_END event
                    agent_end = wrapper.event_builder.create_agent_end(
                        agent_id=agent_id,
                        agent_name=agent_name,
                        status=EventStatus.EVENT_STATUS_COMPLETED,
                        span_id=span_id,
                        metadata={
                            "output": str(output)[:500] if output else None,
                            "framework": FRAMEWORK_NAME,
                        },
                    )
                    await wrapper.tracer.client.send_event(agent_end)
                except Exception as e:
                    logger.error(f"Error in on_agent_end hook: {e}")

            async def on_llm_start(self, context, agent, system_prompt, input_items):
                """Called when an LLM invocation starts."""
                try:
                    agent_id = wrapper._get_agent_id(agent)
                    agent_name = wrapper._get_agent_name(agent)
                    model = wrapper._get_model(agent)

                    # Convert input_items to messages format
                    messages = []
                    if system_prompt:
                        messages.append({"role": ROLE_SYSTEM, "content": system_prompt})

                    for item in input_items:
                        # Handle different item types
                        if hasattr(item, "role") and hasattr(item, "content"):
                            messages.append(
                                {"role": item.role, "content": str(item.content)}
                            )
                        else:
                            messages.append({"role": ROLE_USER, "content": str(item)})

                    # Send MODEL_INVOCATION_START event
                    llm_start = wrapper.event_builder.create_model_invocation_start(
                        provider=PROVIDER_NAME,
                        model=model,
                        messages=messages,
                        agent_id=agent_id,
                        agent_name=agent_name,
                        temperature=None,  # Could extract from agent config if available
                        max_tokens=None,
                        tools=[
                            tool.name if hasattr(tool, "name") else str(tool)
                            for tool in (agent.tools if hasattr(agent, "tools") else [])
                        ],
                    )
                    await wrapper.tracer.client.send_event(llm_start)

                    # Store span_id for matching END event
                    context._chaukas_llm_span_id = llm_start.span_id
                except Exception as e:
                    logger.error(f"Error in on_llm_start hook: {e}", exc_info=True)

            async def on_llm_end(self, context, agent, response):
                """Called when an LLM invocation ends."""
                try:
                    agent_id = wrapper._get_agent_id(agent)
                    agent_name = wrapper._get_agent_name(agent)
                    model = wrapper._get_model(agent)

                    # Extract response details using helper
                    response_content, tool_calls, finish_reason = (
                        wrapper._parse_llm_response(response)
                    )

                    # Get span_id from context
                    span_id = getattr(context, "_chaukas_llm_span_id", None)

                    # Send MODEL_INVOCATION_END event
                    llm_end = wrapper.event_builder.create_model_invocation_end(
                        provider=PROVIDER_NAME,
                        model=model,
                        response_content=response_content,
                        tool_calls=tool_calls if tool_calls else None,
                        finish_reason=finish_reason,
                        prompt_tokens=(
                            response.usage.prompt_tokens
                            if hasattr(response, "usage")
                            and hasattr(response.usage, "prompt_tokens")
                            else None
                        ),
                        completion_tokens=(
                            response.usage.completion_tokens
                            if hasattr(response, "usage")
                            and hasattr(response.usage, "completion_tokens")
                            else None
                        ),
                        total_tokens=(
                            response.usage.total_tokens
                            if hasattr(response, "usage")
                            and hasattr(response.usage, "total_tokens")
                            else None
                        ),
                        duration_ms=None,
                        span_id=span_id,
                        agent_id=agent_id,
                        agent_name=agent_name,
                        error=None,
                    )
                    await wrapper.tracer.client.send_event(llm_end)

                    # Check for content filtering (POLICY_DECISION)
                    # Valid OpenAI finish_reason values: 'stop', 'length', 'tool_calls', 'content_filter', 'function_call'
                    finish_reason = (
                        response.finish_reason
                        if hasattr(response, "finish_reason")
                        else None
                    )
                    if finish_reason == "content_filter":
                        await wrapper._emit_policy_decision(
                            policy_id="openai_content_policy",
                            outcome="blocked",
                            rule_ids=["content_filter"],
                            rationale=f"Response blocked due to: {finish_reason}",
                            agent_id=agent_id,
                            agent_name=agent_name,
                        )
                    elif finish_reason == "length":
                        await wrapper._emit_policy_decision(
                            policy_id="openai_length_policy",
                            outcome="truncated",
                            rule_ids=["max_tokens_limit"],
                            rationale="Response truncated due to length limit",
                            agent_id=agent_id,
                            agent_name=agent_name,
                        )

                    # Check for tool calls and emit TOOL_CALL_START events
                    if tool_calls:
                        for tc in tool_calls:
                            tool_start = wrapper.event_builder.create_tool_call_start(
                                tool_name=tc["name"],
                                arguments=(
                                    json.loads(tc["arguments"])
                                    if tc["arguments"]
                                    else {}
                                ),
                                call_id=tc["id"],
                                agent_id=agent_id,
                                agent_name=agent_name,
                            )
                            await wrapper.tracer.client.send_event(tool_start)
                            # Store span_id for END event
                            if not hasattr(context, "_chaukas_tool_spans"):
                                context._chaukas_tool_spans = {}
                            context._chaukas_tool_spans[tc["id"]] = tool_start.span_id

                except Exception as e:
                    logger.error(f"Error in on_llm_end hook: {e}", exc_info=True)

            async def on_tool_start(self, context, agent, tool):
                """Called when a tool execution starts."""
                try:
                    agent_id = agent.name if hasattr(agent, "name") else str(id(agent))
                    agent_name = (
                        agent.name if hasattr(agent, "name") else "unnamed_agent"
                    )

                    # Check if this is an MCP tool
                    is_mcp = wrapper._is_mcp_tool(tool)

                    if is_mcp:
                        # Send MCP_CALL_START event
                        mcp_start = wrapper.event_builder.create_mcp_call_start(
                            server_name=(
                                tool.name if hasattr(tool, "name") else "mcp_server"
                            ),
                            server_url=(
                                tool.server_url
                                if hasattr(tool, "server_url")
                                else "mcp://local"
                            ),
                            operation="tool_execution",
                            method=(
                                tool.method if hasattr(tool, "method") else "execute"
                            ),
                            request={},
                            protocol_version=MCP_PROTOCOL_VERSION,
                            agent_id=agent_id,
                            agent_name=agent_name,
                        )
                        await wrapper.tracer.client.send_event(mcp_start)
                        # Store for END event
                        if not hasattr(context, "_chaukas_mcp_spans"):
                            context._chaukas_mcp_spans = {}
                        context._chaukas_mcp_spans[
                            tool.name if hasattr(tool, "name") else str(tool)
                        ] = mcp_start.span_id
                    elif not wrapper._is_internal_tool(tool):
                        # Regular tool - but only if we haven't already sent TOOL_CALL_START from LLM response
                        # This handles tools that are executed without LLM involvement
                        tool_name = tool.name if hasattr(tool, "name") else str(tool)
                        if (
                            not hasattr(context, "_chaukas_tool_spans")
                            or tool_name not in context._chaukas_tool_spans
                        ):
                            tool_start = wrapper.event_builder.create_tool_call_start(
                                tool_name=tool_name,
                                arguments={},
                                call_id=None,
                                agent_id=agent_id,
                                agent_name=agent_name,
                            )
                            await wrapper.tracer.client.send_event(tool_start)
                            if not hasattr(context, "_chaukas_tool_spans"):
                                context._chaukas_tool_spans = {}
                            context._chaukas_tool_spans[tool_name] = tool_start.span_id

                    # Track data access for certain tool types
                    if wrapper._is_data_access_tool(tool):
                        data_event = wrapper.event_builder.create_data_access(
                            datasource=wrapper._get_datasource_name(tool),
                            document_ids=None,
                            chunk_ids=None,
                            pii_categories=None,
                            agent_id=agent_id,
                            agent_name=agent_name,
                        )
                        await wrapper.tracer.client.send_event(data_event)

                except Exception as e:
                    logger.error(f"Error in on_tool_start hook: {e}")

            async def on_tool_end(self, context, agent, tool, result):
                """Called when a tool execution ends."""
                try:
                    agent_id = agent.name if hasattr(agent, "name") else str(id(agent))
                    agent_name = (
                        agent.name if hasattr(agent, "name") else "unnamed_agent"
                    )

                    tool_name = tool.name if hasattr(tool, "name") else str(tool)

                    # Check if this is an MCP tool
                    is_mcp = wrapper._is_mcp_tool(tool)

                    if is_mcp:
                        # Send MCP_CALL_END event
                        span_id = None
                        if hasattr(context, "_chaukas_mcp_spans"):
                            span_id = context._chaukas_mcp_spans.get(tool_name)

                        mcp_end = wrapper.event_builder.create_mcp_call_end(
                            server_name=tool_name,
                            server_url=(
                                tool.server_url
                                if hasattr(tool, "server_url")
                                else "mcp://local"
                            ),
                            operation="tool_execution",
                            method=(
                                tool.method if hasattr(tool, "method") else "execute"
                            ),
                            response={"result": str(result)[:1000]} if result else {},
                            execution_time_ms=None,
                            error=None,
                            span_id=span_id,
                            agent_id=agent_id,
                            agent_name=agent_name,
                        )
                        await wrapper.tracer.client.send_event(mcp_end)
                    elif not wrapper._is_internal_tool(tool):
                        # Regular tool - O(1) dict lookup instead of O(n) loop
                        span_id = None
                        if hasattr(context, "_chaukas_tool_spans"):
                            # Try to get by tool ID first (from LLM response), then by name
                            tool_id = tool.id if hasattr(tool, "id") else None
                            span_id = context._chaukas_tool_spans.get(
                                tool_id
                            ) or context._chaukas_tool_spans.get(tool_name)

                        tool_end = wrapper.event_builder.create_tool_call_end(
                            tool_name=tool_name,
                            call_id=tool.id if hasattr(tool, "id") else None,
                            output=str(result)[:1000] if result else None,
                            error=None,
                            execution_time_ms=None,
                            span_id=span_id,
                            agent_id=agent_id,
                            agent_name=agent_name,
                        )
                        await wrapper.tracer.client.send_event(tool_end)

                except Exception as e:
                    logger.error(f"Error in on_tool_end hook: {e}")

            async def on_handoff(self, context, from_agent, to_agent):
                """Called when control is handed off from one agent to another."""
                try:
                    from_id = (
                        from_agent.name
                        if hasattr(from_agent, "name")
                        else str(id(from_agent))
                    )
                    from_name = (
                        from_agent.name
                        if hasattr(from_agent, "name")
                        else "unnamed_agent"
                    )
                    to_id = (
                        to_agent.name
                        if hasattr(to_agent, "name")
                        else str(id(to_agent))
                    )
                    to_name = (
                        to_agent.name if hasattr(to_agent, "name") else "unnamed_agent"
                    )

                    # Send AGENT_HANDOFF event
                    handoff_event = wrapper.event_builder.create_agent_handoff(
                        from_agent_id=from_id,
                        from_agent_name=from_name,
                        to_agent_id=to_id,
                        to_agent_name=to_name,
                        reason="Agent transfer",
                        handoff_type="direct",
                        handoff_data={"framework": FRAMEWORK_NAME},
                    )
                    await wrapper.tracer.client.send_event(handoff_event)

                    # Find and emit AGENT_END for the delegating agent
                    # This is necessary because OpenAI SDK doesn't call on_agent_end for agents that delegate
                    from_span_id = None
                    for i, (
                        stack_agent_id,
                        stack_agent_name,
                        stack_span_id,
                    ) in enumerate(wrapper._agent_stack):
                        if stack_agent_id == from_id:
                            from_span_id = stack_span_id
                            wrapper._agent_stack.pop(i)
                            break

                    if from_span_id:
                        # Emit AGENT_END for the delegating agent
                        agent_end = wrapper.event_builder.create_agent_end(
                            agent_id=from_id,
                            agent_name=from_name,
                            status=EventStatus.EVENT_STATUS_COMPLETED,
                            span_id=from_span_id,
                            metadata={
                                "handoff_to": to_name,
                                "framework": FRAMEWORK_NAME,
                                "reason": "delegated via handoff",
                            },
                        )
                        await wrapper.tracer.client.send_event(agent_end)
                    else:
                        logger.warning(
                            f"Could not find agent {from_name} in stack during handoff"
                        )

                except Exception as e:
                    logger.error(f"Error in on_handoff hook: {e}")

        return ChaukasRunHooks()

    def merge_hooks(self, user_hooks, chaukas_hooks):
        """Merge user-provided hooks with Chaukas hooks."""
        # Create a new hooks instance that calls both
        from agents.lifecycle import RunHooksBase

        class MergedHooks(RunHooksBase):
            """Merged hooks that call both user and Chaukas hooks."""

            async def on_agent_start(self, context, agent):
                if hasattr(chaukas_hooks, "on_agent_start"):
                    await chaukas_hooks.on_agent_start(context, agent)
                if hasattr(user_hooks, "on_agent_start"):
                    await user_hooks.on_agent_start(context, agent)

            async def on_agent_end(self, context, agent, output):
                if hasattr(chaukas_hooks, "on_agent_end"):
                    await chaukas_hooks.on_agent_end(context, agent, output)
                if hasattr(user_hooks, "on_agent_end"):
                    await user_hooks.on_agent_end(context, agent, output)

            async def on_llm_start(self, context, agent, system_prompt, input_items):
                if hasattr(chaukas_hooks, "on_llm_start"):
                    await chaukas_hooks.on_llm_start(
                        context, agent, system_prompt, input_items
                    )
                if hasattr(user_hooks, "on_llm_start"):
                    await user_hooks.on_llm_start(
                        context, agent, system_prompt, input_items
                    )

            async def on_llm_end(self, context, agent, response):
                if hasattr(chaukas_hooks, "on_llm_end"):
                    await chaukas_hooks.on_llm_end(context, agent, response)
                if hasattr(user_hooks, "on_llm_end"):
                    await user_hooks.on_llm_end(context, agent, response)

            async def on_tool_start(self, context, agent, tool):
                if hasattr(chaukas_hooks, "on_tool_start"):
                    await chaukas_hooks.on_tool_start(context, agent, tool)
                if hasattr(user_hooks, "on_tool_start"):
                    await user_hooks.on_tool_start(context, agent, tool)

            async def on_tool_end(self, context, agent, tool, result):
                if hasattr(chaukas_hooks, "on_tool_end"):
                    await chaukas_hooks.on_tool_end(context, agent, tool, result)
                if hasattr(user_hooks, "on_tool_end"):
                    await user_hooks.on_tool_end(context, agent, tool, result)

            async def on_handoff(self, context, from_agent, to_agent):
                if hasattr(chaukas_hooks, "on_handoff"):
                    await chaukas_hooks.on_handoff(context, from_agent, to_agent)
                if hasattr(user_hooks, "on_handoff"):
                    await user_hooks.on_handoff(context, from_agent, to_agent)

        return MergedHooks()

    def _start_session(self, agent):
        """Start a new session."""
        try:
            self._session_active = True
            self._start_time = time.time()
            self._start_metrics = self._get_performance_metrics()

            # Send SESSION_START event
            session_start = self.event_builder.create_session_start(
                metadata={
                    "framework": FRAMEWORK_NAME,
                    "agent_name": agent.name if hasattr(agent, "name") else None,
                    "model": agent.model if hasattr(agent, "model") else None,
                }
            )
            self._send_event_sync(session_start)
            self._session_span_id = session_start.span_id

            # Set session context for all subsequent events
            session_tokens = self.tracer.set_session_context(
                session_start.session_id, session_start.trace_id
            )
            parent_token = self.tracer.set_parent_span_context(self._session_span_id)

        except Exception as e:
            logger.error(f"Error starting session: {e}")

    def _create_session_end_event(self):
        """Create SESSION_END event with metrics (shared helper)."""
        duration_ms = (
            (time.time() - self._start_time) * 1000 if self._start_time else None
        )
        end_metrics = self._get_performance_metrics()

        return self.event_builder.create_session_end(
            span_id=self._session_span_id,
            metadata={
                "framework": FRAMEWORK_NAME,
                "duration_ms": duration_ms,
                "cpu_percent": end_metrics.get("cpu_percent"),
                "memory_mb": end_metrics.get("memory_mb"),
                "success": True,
            },
        )

    async def _end_session(self):
        """End the current session (async)."""
        try:
            if not self._session_active:
                return

            session_end = self._create_session_end_event()
            await self.tracer.client.send_event(session_end)

            self._session_active = False
            self._session_span_id = None

        except Exception as e:
            logger.error(f"Error ending session: {e}")

    def _end_session_sync(self):
        """End the current session (sync)."""
        try:
            if not self._session_active:
                return

            session_end = self._create_session_end_event()
            self._send_event_sync(session_end)

            self._session_active = False
            self._session_span_id = None

        except Exception as e:
            logger.error(f"Error ending session: {e}")

    def _create_error_events(self, error: Exception, agent):
        """Create error-related events (shared helper). Returns tuple of (policy_event, error_event, agent_end_event)."""
        agent_id = self._get_agent_id(agent)
        agent_name = self._get_agent_name(agent)
        error_msg = str(error)

        # Create POLICY_DECISION event for rate limits
        policy_event = None
        if OPENAI_EXCEPTIONS_AVAILABLE and isinstance(error, RateLimitError):
            policy_event = self.event_builder.create_policy_decision(
                policy_id="openai_rate_limit",
                outcome="blocked",
                rule_ids=["rate_limit"],
                rationale=f"Request blocked due to rate limit: {error_msg[:100]}",
                agent_id=agent_id,
                agent_name=agent_name,
            )

        # Create ERROR event
        error_event = self.event_builder.create_error(
            error_message=error_msg,
            error_code=type(error).__name__,
            recoverable=False,  # We can't determine this reliably from outside SDK
            agent_id=agent_id,
            agent_name=agent_name,
        )

        # Pop agent from stack and create AGENT_END event
        span_id = None
        for i, (stack_agent_id, stack_agent_name, stack_span_id) in enumerate(
            self._agent_stack
        ):
            if stack_agent_id == agent_id:
                span_id = stack_span_id
                self._agent_stack.pop(i)
                break

        agent_end_event = None
        if span_id:
            agent_end_event = self.event_builder.create_agent_end(
                agent_id=agent_id,
                agent_name=agent_name,
                status=EventStatus.EVENT_STATUS_FAILED,
                span_id=span_id,
                metadata={
                    "error": error_msg[:200],
                    "error_type": type(error).__name__,
                    "framework": FRAMEWORK_NAME,
                },
            )

        return policy_event, error_event, agent_end_event

    async def _handle_error(self, error: Exception, agent):
        """
        Handle errors and emit ERROR events (async).

        Note: RETRY events are NOT emitted here because we cannot observe
        OpenAI SDK's internal retry attempts. The SDK retries (408, 429, 500+)
        happen invisibly within the HTTP client layer.
        """
        try:
            policy_event, error_event, agent_end_event = self._create_error_events(
                error, agent
            )

            if policy_event:
                await self.tracer.client.send_event(policy_event)
            await self.tracer.client.send_event(error_event)
            if agent_end_event:
                await self.tracer.client.send_event(agent_end_event)

        except Exception as e:
            logger.error(f"Error handling error: {e}")
            import traceback

            traceback.print_exc()

    def _handle_error_sync(self, error: Exception, agent):
        """
        Handle errors and emit ERROR events (sync).

        Note: RETRY events are NOT emitted here because we cannot observe
        OpenAI SDK's internal retry attempts. The SDK retries (408, 429, 500+)
        happen invisibly within the HTTP client layer.
        """
        try:
            policy_event, error_event, agent_end_event = self._create_error_events(
                error, agent
            )

            if policy_event:
                self._send_event_sync(policy_event)
            self._send_event_sync(error_event)
            if agent_end_event:
                self._send_event_sync(agent_end_event)

        except Exception as e:
            logger.error(f"Error handling error: {e}")

    # Helper methods for common attribute access patterns
    def _get_agent_id(self, agent) -> str:
        """Extract agent ID from agent object."""
        return agent.name if hasattr(agent, "name") else str(id(agent))

    def _get_agent_name(self, agent) -> str:
        """Extract agent name from agent object."""
        return agent.name if hasattr(agent, "name") else DEFAULT_AGENT_NAME

    def _get_model(self, agent) -> str:
        """Extract model name from agent object."""
        return agent.model if hasattr(agent, "model") else DEFAULT_MODEL_NAME

    def _get_tool_name(self, tool) -> str:
        """Extract tool name from tool object."""
        return tool.name if hasattr(tool, "name") else str(tool)

    def _parse_llm_response(self, response):
        """Parse LLM response safely with validation. Returns tuple of (content, tool_calls, finish_reason)."""
        response_content = None
        tool_calls = []
        finish_reason = None

        # Extract content
        if hasattr(response, "content") and response.content:
            response_content = str(response.content)

        # Extract tool calls
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tc in response.tool_calls:
                tool_calls.append(
                    {
                        "id": tc.id if hasattr(tc, "id") else None,
                        "name": (
                            tc.function.name
                            if hasattr(tc, "function") and hasattr(tc.function, "name")
                            else None
                        ),
                        "arguments": (
                            tc.function.arguments
                            if hasattr(tc, "function")
                            and hasattr(tc.function, "arguments")
                            else None
                        ),
                    }
                )

        # Extract and validate finish_reason
        if hasattr(response, "finish_reason"):
            finish_reason = response.finish_reason
            # Validate against known OpenAI finish_reason values
            valid_reasons = [
                "stop",
                "length",
                "tool_calls",
                "content_filter",
                "function_call",
            ]
            if finish_reason and finish_reason not in valid_reasons:
                logger.warning(f"Unknown finish_reason value: {finish_reason}")

        return response_content, tool_calls, finish_reason

    def _is_mcp_tool(self, tool) -> bool:
        """Check if a tool is an MCP tool."""
        # Use proper type checking if available
        if OPENAI_TOOLS_AVAILABLE and HostedMCPTool is not None:
            if isinstance(tool, HostedMCPTool):
                return True

        # Fallback: Check for MCP-related attributes for backwards compatibility
        if hasattr(tool, "server_url") or hasattr(tool, "protocol"):
            return True

        return False

    def _is_internal_tool(self, tool) -> bool:
        """Check if a tool is an internal/system tool that shouldn't be tracked."""
        tool_name = tool.name if hasattr(tool, "name") else str(tool)
        internal_tools = ["transfer_to_agent", "handoff", "system"]
        return any(internal in tool_name.lower() for internal in internal_tools)

    def _is_data_access_tool(self, tool) -> bool:
        """Check if a tool involves data access."""
        # Use proper type checking if available
        if OPENAI_TOOLS_AVAILABLE:
            if isinstance(
                tool,
                (
                    FileSearchTool,
                    WebSearchTool,
                    CodeInterpreterTool,
                    LocalShellTool,
                    ComputerTool,
                ),
            ):
                return True

        # Fallback: Check type name for backwards compatibility
        tool_type = type(tool).__name__
        data_tools = [
            "FileSearchTool",
            "WebSearchTool",
            "CodeInterpreterTool",
            "LocalShellTool",
            "ComputerTool",
        ]
        return any(dt in tool_type for dt in data_tools)

    def _get_datasource_name(self, tool) -> str:
        """Get the datasource name for a data access tool."""
        # Use proper type checking if available
        if OPENAI_TOOLS_AVAILABLE:
            if isinstance(tool, FileSearchTool):
                return "file_search"
            elif isinstance(tool, WebSearchTool):
                return "web_search"
            elif isinstance(tool, CodeInterpreterTool):
                return "code_interpreter"
            elif isinstance(tool, LocalShellTool):
                return "local_shell"
            elif isinstance(tool, ComputerTool):
                return "computer"

        # Fallback: Check type name for backwards compatibility
        tool_type = type(tool).__name__
        if "FileSearch" in tool_type:
            return "file_search"
        elif "WebSearch" in tool_type:
            return "web_search"
        elif "CodeInterpreter" in tool_type:
            return "code_interpreter"
        elif "LocalShell" in tool_type:
            return "local_shell"
        elif "Computer" in tool_type:
            return "computer"
        else:
            return "unknown"

    def _send_event_sync(self, event):
        """Helper to send event from sync context."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self.tracer.client.send_event(event))
            else:
                loop.run_until_complete(self.tracer.client.send_event(event))
        except RuntimeError:
            asyncio.run(self.tracer.client.send_event(event))

    def _get_performance_metrics(self):
        """Collect current performance metrics."""
        if not PSUTIL_AVAILABLE:
            return {}

        try:
            process = psutil.Process(os.getpid())
            return {
                "cpu_percent": process.cpu_percent(),
                "memory_mb": process.memory_info().rss / 1024 / 1024,
                "num_threads": process.num_threads(),
            }
        except Exception:
            return {}

    def unpatch_runner(self):
        """Restore original Runner methods."""
        if self._original_runner_run:
            try:
                from agents import Runner

                Runner.run = self._original_runner_run
                Runner.run_sync = self._original_runner_run_sync
                Runner.run_streamed = self._original_runner_run_streamed
                self._original_runner_run = None
                self._original_runner_run_sync = None
                self._original_runner_run_streamed = None
                logger.info("Successfully unpatched Runner methods")
            except Exception as e:
                logger.error(f"Failed to unpatch Runner: {e}")

    # New event type support - POLICY_DECISION, STATE_UPDATE, SYSTEM_EVENT

    def _emit_system_event_sync(self, message: str, severity_str: str = "INFO"):
        """Emit a SYSTEM_EVENT synchronously."""
        try:
            from chaukas.spec.common.v1.events_pb2 import Severity

            severity_map = {
                "DEBUG": Severity.SEVERITY_DEBUG,
                "INFO": Severity.SEVERITY_INFO,
                "WARNING": Severity.SEVERITY_WARNING,
                "ERROR": Severity.SEVERITY_ERROR,
                "CRITICAL": Severity.SEVERITY_CRITICAL,
            }
            severity = severity_map.get(severity_str, Severity.SEVERITY_INFO)

            system_event = self.event_builder.create_system_event(
                message=message,
                severity=severity,
                metadata={"framework": FRAMEWORK_NAME},
            )
            self._send_event_sync(system_event)
        except Exception as e:
            logger.debug(f"Failed to emit system event: {e}")

    async def _emit_system_event(
        self,
        message: str,
        severity_str: str = "INFO",
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
    ):
        """Emit a SYSTEM_EVENT asynchronously."""
        try:
            from chaukas.spec.common.v1.events_pb2 import Severity

            severity_map = {
                "DEBUG": Severity.SEVERITY_DEBUG,
                "INFO": Severity.SEVERITY_INFO,
                "WARNING": Severity.SEVERITY_WARNING,
                "ERROR": Severity.SEVERITY_ERROR,
                "CRITICAL": Severity.SEVERITY_CRITICAL,
            }
            severity = severity_map.get(severity_str, Severity.SEVERITY_INFO)

            system_event = self.event_builder.create_system_event(
                message=message,
                severity=severity,
                metadata={"framework": FRAMEWORK_NAME},
                agent_id=agent_id,
                agent_name=agent_name,
            )
            await self.tracer.client.send_event(system_event)
        except Exception as e:
            logger.debug(f"Failed to emit system event: {e}")

    async def _emit_state_update(
        self, agent_id: str, agent_name: str, state_data: Dict[str, Any]
    ):
        """Emit a STATE_UPDATE event."""
        try:
            state_event = self.event_builder.create_state_update(
                state_data=state_data, agent_id=agent_id, agent_name=agent_name
            )
            await self.tracer.client.send_event(state_event)
        except Exception as e:
            logger.debug(f"Failed to emit state update event: {e}")

    async def _emit_policy_decision(
        self,
        policy_id: str,
        outcome: str,
        rule_ids: List[str],
        rationale: Optional[str] = None,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
    ):
        """Emit a POLICY_DECISION event."""
        try:
            policy_event = self.event_builder.create_policy_decision(
                policy_id=policy_id,
                outcome=outcome,
                rule_ids=rule_ids,
                rationale=rationale,
                agent_id=agent_id,
                agent_name=agent_name,
            )
            await self.tracer.client.send_event(policy_event)
        except Exception as e:
            logger.debug(f"Failed to emit policy decision event: {e}")

    def _track_agent_state(self, agent_id: str, agent) -> Dict[str, Any]:
        """Track agent state changes and return state diff."""
        current_state = {
            "model": agent.model if hasattr(agent, "model") else None,
            "instructions": (
                agent.instructions if hasattr(agent, "instructions") else None
            ),
            "tools_count": (
                len(agent.tools) if hasattr(agent, "tools") and agent.tools else 0
            ),
            "temperature": getattr(agent, "temperature", None),
            "max_tokens": getattr(agent, "max_tokens", None),
        }

        # Check if state has changed
        previous_state = self._agent_states.get(agent_id, {})
        state_diff = {}

        for key, value in current_state.items():
            if key not in previous_state or previous_state[key] != value:
                state_diff[key] = {"old": previous_state.get(key), "new": value}

        # Update stored state
        self._agent_states[agent_id] = current_state

        return state_diff if state_diff else None
