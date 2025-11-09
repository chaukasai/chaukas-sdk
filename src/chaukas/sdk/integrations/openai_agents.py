"""
OpenAI Agents SDK integration for Chaukas instrumentation.
Provides 100% event coverage using hooks and monkey patching.
"""

import functools
import logging
import time
import json
from typing import Any, Dict, Optional, List, Union
from datetime import datetime
import os

# Optional performance monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

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
        self._session_active = False
        self._session_span_id = None
        self._start_time = None
        self._start_metrics = None
        self._current_agent_span_id = None
        self._original_runner_run = None
        self._original_runner_run_sync = None
        self._original_runner_run_streamed = None
        # Track retry attempts
        self._llm_retry_attempts = {}
        self._tool_retry_attempts = {}
        self._agent_retry_attempts = {}
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
                input_data = input if input is not None else kwargs.pop('input_data', None)

                # Start session if not active
                if not wrapper._session_active:
                    wrapper._start_session(starting_agent)
                
                # Send INPUT_RECEIVED event
                input_event = wrapper.event_builder.create_input_received(
                    content=str(input_data)[:1000] if input_data else "",
                    metadata={"method": "Runner.run", "agent": starting_agent.name if hasattr(starting_agent, 'name') else None}
                )
                await wrapper.tracer.client.send_event(input_event)
                
                # Inject our custom hooks
                hooks = kwargs.get('hooks')
                custom_hooks = wrapper.create_custom_hooks()
                
                # Merge with existing hooks if provided
                if hooks:
                    # Chain our hooks with user's hooks
                    merged_hooks = wrapper.merge_hooks(hooks, custom_hooks)
                    kwargs['hooks'] = merged_hooks
                else:
                    kwargs['hooks'] = custom_hooks
                
                try:
                    # Execute original method
                    result = await wrapper._original_runner_run(starting_agent, input_data, **kwargs)
                    
                    # Send OUTPUT_EMITTED event
                    output_event = wrapper.event_builder.create_output_emitted(
                        content=str(result)[:1000] if result else "No output",
                        metadata={"method": "Runner.run", "agent": starting_agent.name if hasattr(starting_agent, 'name') else None}
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
                input_data = input if input is not None else kwargs.pop('input_data', None)

                # Start session if not active
                if not wrapper._session_active:
                    wrapper._start_session(starting_agent)
                
                # Send INPUT_RECEIVED event
                input_event = wrapper.event_builder.create_input_received(
                    content=str(input_data)[:1000] if input_data else "",
                    metadata={"method": "Runner.run_sync", "agent": starting_agent.name if hasattr(starting_agent, 'name') else None}
                )
                wrapper._send_event_sync(input_event)
                
                # Inject our custom hooks
                hooks = kwargs.get('hooks')
                custom_hooks = wrapper.create_custom_hooks()
                
                if hooks:
                    merged_hooks = wrapper.merge_hooks(hooks, custom_hooks)
                    kwargs['hooks'] = merged_hooks
                else:
                    kwargs['hooks'] = custom_hooks
                
                try:
                    # Execute original method
                    result = wrapper._original_runner_run_sync(starting_agent, input_data, **kwargs)
                    
                    # Send OUTPUT_EMITTED event
                    output_event = wrapper.event_builder.create_output_emitted(
                        content=str(result)[:1000] if result else "No output",
                        metadata={"method": "Runner.run_sync", "agent": starting_agent.name if hasattr(starting_agent, 'name') else None}
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
                input_data = input if input is not None else kwargs.pop('input_data', None)

                # Start session if not active
                if not wrapper._session_active:
                    wrapper._start_session(starting_agent)
                
                # Send INPUT_RECEIVED event
                input_event = wrapper.event_builder.create_input_received(
                    content=str(input_data)[:1000] if input_data else "",
                    metadata={"method": "Runner.run_streamed", "agent": starting_agent.name if hasattr(starting_agent, 'name') else None}
                )
                await wrapper.tracer.client.send_event(input_event)
                
                # Inject our custom hooks
                hooks = kwargs.get('hooks')
                custom_hooks = wrapper.create_custom_hooks()
                
                if hooks:
                    merged_hooks = wrapper.merge_hooks(hooks, custom_hooks)
                    kwargs['hooks'] = merged_hooks
                else:
                    kwargs['hooks'] = custom_hooks
                
                try:
                    # Execute original method - returns an async generator
                    async for chunk in wrapper._original_runner_run_streamed(starting_agent, input_data, **kwargs):
                        yield chunk
                    
                    # Send OUTPUT_EMITTED event for streamed response
                    output_event = wrapper.event_builder.create_output_emitted(
                        content="[Streamed response completed]",
                        metadata={"method": "Runner.run_streamed", "agent": starting_agent.name if hasattr(starting_agent, 'name') else None}
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
            self._emit_system_event_sync("OpenAI Runner methods successfully patched", "INFO")

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
            logger.debug(f"Successfully imported _MCPServerWithClientSession: {_MCPServerWithClientSession}")
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
                server_name = self.name if hasattr(self, 'name') else "mcp_server"
                server_url = getattr(self, 'url', None) or getattr(self, '_url', None) or "mcp://local"

                # Send MCP_CALL_START event
                mcp_start = wrapper.event_builder.create_mcp_call_start(
                    server_name=server_name,
                    server_url=str(server_url),
                    operation="get_prompt",
                    method="get_prompt",
                    request={"prompt_name": name, "arguments": arguments or {}},
                    protocol_version="1.0"
                )
                await wrapper.tracer.client.send_event(mcp_start)

                try:
                    # Execute original method
                    result = await wrapper._original_mcp_get_prompt(self, name, arguments)

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
                            "message_count": len(result.messages) if hasattr(result, 'messages') else 0
                        },
                        execution_time_ms=execution_time_ms,
                        error=None,
                        span_id=mcp_start.span_id
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
                        span_id=mcp_start.span_id
                    )
                    await wrapper.tracer.client.send_event(mcp_end)
                    raise

            # Patch call_tool method
            @functools.wraps(wrapper._original_mcp_call_tool)
            async def instrumented_call_tool(self, tool_name: str, arguments: dict = None):
                """Instrumented version of MCPServer.call_tool()."""
                start_time = time.time()
                server_name = self.name if hasattr(self, 'name') else "mcp_server"
                server_url = getattr(self, 'url', None) or getattr(self, '_url', None) or "mcp://local"

                # Send MCP_CALL_START event
                mcp_start = wrapper.event_builder.create_mcp_call_start(
                    server_name=server_name,
                    server_url=str(server_url),
                    operation="call_tool",
                    method="call_tool",
                    request={"tool_name": tool_name, "arguments": arguments or {}},
                    protocol_version="1.0"
                )
                await wrapper.tracer.client.send_event(mcp_start)

                try:
                    # Execute original method
                    result = await wrapper._original_mcp_call_tool(self, tool_name, arguments)

                    # Calculate execution time
                    execution_time_ms = (time.time() - start_time) * 1000

                    # Send MCP_CALL_END event
                    response_data = {
                        "tool_name": tool_name,
                        "content_count": len(result.content) if hasattr(result, 'content') else 0
                    }
                    if hasattr(result, 'content') and result.content:
                        # Include first content item (truncated)
                        first_content = result.content[0]
                        if hasattr(first_content, 'text'):
                            response_data["preview"] = str(first_content.text)[:200]

                    mcp_end = wrapper.event_builder.create_mcp_call_end(
                        server_name=server_name,
                        server_url=str(server_url),
                        operation="call_tool",
                        method="call_tool",
                        response=response_data,
                        execution_time_ms=execution_time_ms,
                        error=None,
                        span_id=mcp_start.span_id
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
                        span_id=mcp_start.span_id
                    )
                    await wrapper.tracer.client.send_event(mcp_end)
                    raise

            # Apply patches to _MCPServerWithClientSession (which MCPServerStreamableHttp inherits from)
            _MCPServerWithClientSession.get_prompt = instrumented_get_prompt
            _MCPServerWithClientSession.call_tool = instrumented_call_tool

            logger.info("Successfully patched MCP Server methods (_MCPServerWithClientSession)")
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
                    agent_id = agent.name if hasattr(agent, 'name') else str(id(agent))
                    agent_name = agent.name if hasattr(agent, 'name') else "unnamed_agent"

                    # Check if this is first time seeing this agent
                    is_first_run = agent_id not in wrapper._agent_states

                    # Track agent state changes and emit STATE_UPDATE (always emit on first run or changes)
                    state_diff = wrapper._track_agent_state(agent_id, agent)
                    if state_diff:
                        await wrapper._emit_state_update(agent_id, agent_name, state_diff)

                    # Emit SYSTEM_EVENT for agent initialization (first run only)
                    if is_first_run:
                        await wrapper._emit_system_event(
                            f"Agent '{agent_name}' initialized with model {agent.model if hasattr(agent, 'model') else 'unknown'}",
                            "INFO",
                            agent_id=agent_id,
                            agent_name=agent_name
                        )

                    # Send AGENT_START event
                    agent_start = wrapper.event_builder.create_agent_start(
                        agent_id=agent_id,
                        agent_name=agent_name,
                        role="assistant",
                        instructions=agent.instructions if hasattr(agent, 'instructions') else None,
                        tools=[tool.name if hasattr(tool, 'name') else str(tool) for tool in (agent.tools if hasattr(agent, 'tools') else [])],
                        metadata={
                            "model": agent.model if hasattr(agent, 'model') else None,
                            "framework": "openai_agents"
                        }
                    )
                    await wrapper.tracer.client.send_event(agent_start)
                    wrapper._current_agent_span_id = agent_start.span_id
                except Exception as e:
                    logger.error(f"Error in on_agent_start hook: {e}")
            
            async def on_agent_end(self, context, agent, output):
                """Called when an agent ends execution."""
                try:
                    agent_id = agent.name if hasattr(agent, 'name') else str(id(agent))
                    agent_name = agent.name if hasattr(agent, 'name') else "unnamed_agent"
                    
                    # Clear retry counter on successful completion
                    agent_key = f"{agent_id}_{agent.model if hasattr(agent, 'model') else 'unknown'}"
                    wrapper._agent_retry_attempts.pop(agent_key, None)
                    
                    # Send AGENT_END event
                    agent_end = wrapper.event_builder.create_agent_end(
                        agent_id=agent_id,
                        agent_name=agent_name,
                        status=EventStatus.EVENT_STATUS_COMPLETED,
                        span_id=wrapper._current_agent_span_id,
                        metadata={
                            "output": str(output)[:500] if output else None,
                            "framework": "openai_agents"
                        }
                    )
                    await wrapper.tracer.client.send_event(agent_end)
                except Exception as e:
                    logger.error(f"Error in on_agent_end hook: {e}")
            
            async def on_llm_start(self, context, agent, system_prompt, input_items):
                """Called when an LLM invocation starts."""
                try:
                    agent_id = agent.name if hasattr(agent, 'name') else str(id(agent))
                    agent_name = agent.name if hasattr(agent, 'name') else "unnamed_agent"
                    
                    # Convert input_items to messages format
                    messages = []
                    if system_prompt:
                        messages.append({"role": "system", "content": system_prompt})
                    
                    for item in input_items:
                        # Handle different item types
                        if hasattr(item, 'role') and hasattr(item, 'content'):
                            messages.append({"role": item.role, "content": str(item.content)})
                        else:
                            messages.append({"role": "user", "content": str(item)})
                    
                    # Send MODEL_INVOCATION_START event
                    llm_start = wrapper.event_builder.create_model_invocation_start(
                        provider="openai",
                        model=agent.model if hasattr(agent, 'model') else "unknown",
                        messages=messages,
                        agent_id=agent_id,
                        agent_name=agent_name,
                        temperature=None,  # Could extract from agent config if available
                        max_tokens=None,
                        tools=[tool.name if hasattr(tool, 'name') else str(tool) for tool in (agent.tools if hasattr(agent, 'tools') else [])]
                    )
                    await wrapper.tracer.client.send_event(llm_start)
                    
                    # Store span_id for matching END event
                    context._chaukas_llm_span_id = llm_start.span_id
                except Exception as e:
                    logger.error(f"Error in on_llm_start hook: {e}", exc_info=True)
            
            async def on_llm_end(self, context, agent, response):
                """Called when an LLM invocation ends."""
                try:
                    agent_id = agent.name if hasattr(agent, 'name') else str(id(agent))
                    agent_name = agent.name if hasattr(agent, 'name') else "unnamed_agent"
                    
                    # Clear retry counter on successful completion
                    model = agent.model if hasattr(agent, 'model') else "unknown"
                    llm_key = f"{agent_id}_{model}_openai"
                    wrapper._llm_retry_attempts.pop(llm_key, None)
                    
                    # Extract response details
                    response_content = None
                    tool_calls = []
                    
                    if hasattr(response, 'content'):
                        response_content = str(response.content)
                    
                    if hasattr(response, 'tool_calls') and response.tool_calls:
                        for tc in response.tool_calls:
                            tool_calls.append({
                                "id": tc.id if hasattr(tc, 'id') else None,
                                "name": tc.function.name if hasattr(tc, 'function') and hasattr(tc.function, 'name') else None,
                                "arguments": tc.function.arguments if hasattr(tc, 'function') and hasattr(tc.function, 'arguments') else None
                            })
                    
                    # Get span_id from context
                    span_id = getattr(context, '_chaukas_llm_span_id', None)
                    
                    # Send MODEL_INVOCATION_END event
                    llm_end = wrapper.event_builder.create_model_invocation_end(
                        provider="openai",
                        model=model,
                        response_content=response_content,
                        tool_calls=tool_calls if tool_calls else None,
                        finish_reason=response.finish_reason if hasattr(response, 'finish_reason') else None,
                        prompt_tokens=response.usage.prompt_tokens if hasattr(response, 'usage') and hasattr(response.usage, 'prompt_tokens') else None,
                        completion_tokens=response.usage.completion_tokens if hasattr(response, 'usage') and hasattr(response.usage, 'completion_tokens') else None,
                        total_tokens=response.usage.total_tokens if hasattr(response, 'usage') and hasattr(response.usage, 'total_tokens') else None,
                        duration_ms=None,
                        span_id=span_id,
                        agent_id=agent_id,
                        agent_name=agent_name,
                        error=None
                    )
                    await wrapper.tracer.client.send_event(llm_end)

                    # Check for content filtering/moderation (POLICY_DECISION)
                    finish_reason = response.finish_reason if hasattr(response, 'finish_reason') else None
                    if finish_reason in ['content_filter', 'content_policy', 'moderation']:
                        await wrapper._emit_policy_decision(
                            policy_id="openai_content_policy",
                            outcome="blocked",
                            rule_ids=["content_filter"],
                            rationale=f"Response blocked due to: {finish_reason}",
                            agent_id=agent_id,
                            agent_name=agent_name
                        )
                    elif finish_reason == 'length':
                        await wrapper._emit_policy_decision(
                            policy_id="openai_length_policy",
                            outcome="truncated",
                            rule_ids=["max_tokens_limit"],
                            rationale="Response truncated due to length limit",
                            agent_id=agent_id,
                            agent_name=agent_name
                        )

                    # Check for tool calls and emit TOOL_CALL_START events
                    if tool_calls:
                        for tc in tool_calls:
                            tool_start = wrapper.event_builder.create_tool_call_start(
                                tool_name=tc['name'],
                                arguments=json.loads(tc['arguments']) if tc['arguments'] else {},
                                call_id=tc['id'],
                                agent_id=agent_id,
                                agent_name=agent_name
                            )
                            await wrapper.tracer.client.send_event(tool_start)
                            # Store span_id for END event
                            if not hasattr(context, '_chaukas_tool_spans'):
                                context._chaukas_tool_spans = {}
                            context._chaukas_tool_spans[tc['id']] = tool_start.span_id
                    
                except Exception as e:
                    logger.error(f"Error in on_llm_end hook: {e}", exc_info=True)
            
            async def on_tool_start(self, context, agent, tool):
                """Called when a tool execution starts."""
                try:
                    agent_id = agent.name if hasattr(agent, 'name') else str(id(agent))
                    agent_name = agent.name if hasattr(agent, 'name') else "unnamed_agent"
                    
                    # Check if this is an MCP tool
                    is_mcp = wrapper._is_mcp_tool(tool)
                    
                    if is_mcp:
                        # Send MCP_CALL_START event
                        mcp_start = wrapper.event_builder.create_mcp_call_start(
                            server_name=tool.name if hasattr(tool, 'name') else "mcp_server",
                            server_url=tool.server_url if hasattr(tool, 'server_url') else "mcp://local",
                            operation="tool_execution",
                            method=tool.method if hasattr(tool, 'method') else "execute",
                            request={},
                            protocol_version="1.0",
                            agent_id=agent_id,
                            agent_name=agent_name
                        )
                        await wrapper.tracer.client.send_event(mcp_start)
                        # Store for END event
                        if not hasattr(context, '_chaukas_mcp_spans'):
                            context._chaukas_mcp_spans = {}
                        context._chaukas_mcp_spans[tool.name if hasattr(tool, 'name') else str(tool)] = mcp_start.span_id
                    elif not wrapper._is_internal_tool(tool):
                        # Regular tool - but only if we haven't already sent TOOL_CALL_START from LLM response
                        # This handles tools that are executed without LLM involvement
                        tool_name = tool.name if hasattr(tool, 'name') else str(tool)
                        if not hasattr(context, '_chaukas_tool_spans') or tool_name not in context._chaukas_tool_spans:
                            tool_start = wrapper.event_builder.create_tool_call_start(
                                tool_name=tool_name,
                                arguments={},
                                call_id=None,
                                agent_id=agent_id,
                                agent_name=agent_name
                            )
                            await wrapper.tracer.client.send_event(tool_start)
                            if not hasattr(context, '_chaukas_tool_spans'):
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
                            agent_name=agent_name
                        )
                        await wrapper.tracer.client.send_event(data_event)
                    
                except Exception as e:
                    logger.error(f"Error in on_tool_start hook: {e}")
            
            async def on_tool_end(self, context, agent, tool, result):
                """Called when a tool execution ends."""
                try:
                    agent_id = agent.name if hasattr(agent, 'name') else str(id(agent))
                    agent_name = agent.name if hasattr(agent, 'name') else "unnamed_agent"
                    
                    # Clear retry counter on successful completion
                    tool_name = tool.name if hasattr(tool, 'name') else str(tool)
                    tool_key = f"{agent_id}_{tool_name}"
                    wrapper._tool_retry_attempts.pop(tool_key, None)
                    
                    # Check if this is an MCP tool
                    is_mcp = wrapper._is_mcp_tool(tool)
                    
                    if is_mcp:
                        # Send MCP_CALL_END event
                        span_id = None
                        if hasattr(context, '_chaukas_mcp_spans'):
                            span_id = context._chaukas_mcp_spans.get(tool_name)
                        
                        mcp_end = wrapper.event_builder.create_mcp_call_end(
                            server_name=tool_name,
                            server_url=tool.server_url if hasattr(tool, 'server_url') else "mcp://local",
                            operation="tool_execution",
                            method=tool.method if hasattr(tool, 'method') else "execute",
                            response={"result": str(result)[:1000]} if result else {},
                            execution_time_ms=None,
                            error=None,
                            span_id=span_id,
                            agent_id=agent_id,
                            agent_name=agent_name
                        )
                        await wrapper.tracer.client.send_event(mcp_end)
                    elif not wrapper._is_internal_tool(tool):
                        # Regular tool
                        span_id = None
                        if hasattr(context, '_chaukas_tool_spans'):
                            # Try to get by tool ID first (from LLM response), then by name
                            for key, value in context._chaukas_tool_spans.items():
                                if key == tool_name or (hasattr(tool, 'id') and key == tool.id):
                                    span_id = value
                                    break
                        
                        tool_end = wrapper.event_builder.create_tool_call_end(
                            tool_name=tool_name,
                            call_id=tool.id if hasattr(tool, 'id') else None,
                            output=str(result)[:1000] if result else None,
                            error=None,
                            execution_time_ms=None,
                            span_id=span_id,
                            agent_id=agent_id,
                            agent_name=agent_name
                        )
                        await wrapper.tracer.client.send_event(tool_end)
                    
                except Exception as e:
                    logger.error(f"Error in on_tool_end hook: {e}")
            
            async def on_handoff(self, context, from_agent, to_agent):
                """Called when control is handed off from one agent to another."""
                try:
                    from_id = from_agent.name if hasattr(from_agent, 'name') else str(id(from_agent))
                    from_name = from_agent.name if hasattr(from_agent, 'name') else "unnamed_agent"
                    to_id = to_agent.name if hasattr(to_agent, 'name') else str(id(to_agent))
                    to_name = to_agent.name if hasattr(to_agent, 'name') else "unnamed_agent"
                    
                    # Send AGENT_HANDOFF event
                    handoff_event = wrapper.event_builder.create_agent_handoff(
                        from_agent_id=from_id,
                        from_agent_name=from_name,
                        to_agent_id=to_id,
                        to_agent_name=to_name,
                        reason="Agent transfer",
                        handoff_type="direct",
                        handoff_data={
                            "framework": "openai_agents"
                        }
                    )
                    await wrapper.tracer.client.send_event(handoff_event)
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
                if hasattr(chaukas_hooks, 'on_agent_start'):
                    await chaukas_hooks.on_agent_start(context, agent)
                if hasattr(user_hooks, 'on_agent_start'):
                    await user_hooks.on_agent_start(context, agent)
            
            async def on_agent_end(self, context, agent, output):
                if hasattr(chaukas_hooks, 'on_agent_end'):
                    await chaukas_hooks.on_agent_end(context, agent, output)
                if hasattr(user_hooks, 'on_agent_end'):
                    await user_hooks.on_agent_end(context, agent, output)
            
            async def on_llm_start(self, context, agent, system_prompt, input_items):
                if hasattr(chaukas_hooks, 'on_llm_start'):
                    await chaukas_hooks.on_llm_start(context, agent, system_prompt, input_items)
                if hasattr(user_hooks, 'on_llm_start'):
                    await user_hooks.on_llm_start(context, agent, system_prompt, input_items)
            
            async def on_llm_end(self, context, agent, response):
                if hasattr(chaukas_hooks, 'on_llm_end'):
                    await chaukas_hooks.on_llm_end(context, agent, response)
                if hasattr(user_hooks, 'on_llm_end'):
                    await user_hooks.on_llm_end(context, agent, response)
            
            async def on_tool_start(self, context, agent, tool):
                if hasattr(chaukas_hooks, 'on_tool_start'):
                    await chaukas_hooks.on_tool_start(context, agent, tool)
                if hasattr(user_hooks, 'on_tool_start'):
                    await user_hooks.on_tool_start(context, agent, tool)
            
            async def on_tool_end(self, context, agent, tool, result):
                if hasattr(chaukas_hooks, 'on_tool_end'):
                    await chaukas_hooks.on_tool_end(context, agent, tool, result)
                if hasattr(user_hooks, 'on_tool_end'):
                    await user_hooks.on_tool_end(context, agent, tool, result)
            
            async def on_handoff(self, context, from_agent, to_agent):
                if hasattr(chaukas_hooks, 'on_handoff'):
                    await chaukas_hooks.on_handoff(context, from_agent, to_agent)
                if hasattr(user_hooks, 'on_handoff'):
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
                    "framework": "openai_agents",
                    "agent_name": agent.name if hasattr(agent, 'name') else None,
                    "model": agent.model if hasattr(agent, 'model') else None
                }
            )
            self._send_event_sync(session_start)
            self._session_span_id = session_start.span_id
            
            # Set session context for all subsequent events
            session_tokens = self.tracer.set_session_context(session_start.session_id, session_start.trace_id)
            parent_token = self.tracer.set_parent_span_context(self._session_span_id)
            
        except Exception as e:
            logger.error(f"Error starting session: {e}")
    
    async def _end_session(self):
        """End the current session (async)."""
        try:
            if not self._session_active:
                return
            
            # Calculate performance metrics
            duration_ms = (time.time() - self._start_time) * 1000 if self._start_time else None
            end_metrics = self._get_performance_metrics()
            
            # Send SESSION_END event
            session_end = self.event_builder.create_session_end(
                span_id=self._session_span_id,
                metadata={
                    "framework": "openai_agents",
                    "duration_ms": duration_ms,
                    "cpu_percent": end_metrics.get('cpu_percent'),
                    "memory_mb": end_metrics.get('memory_mb'),
                    "success": True
                }
            )
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
            
            # Calculate performance metrics
            duration_ms = (time.time() - self._start_time) * 1000 if self._start_time else None
            end_metrics = self._get_performance_metrics()
            
            # Send SESSION_END event
            session_end = self.event_builder.create_session_end(
                span_id=self._session_span_id,
                metadata={
                    "framework": "openai_agents",
                    "duration_ms": duration_ms,
                    "cpu_percent": end_metrics.get('cpu_percent'),
                    "memory_mb": end_metrics.get('memory_mb'),
                    "success": True
                }
            )
            self._send_event_sync(session_end)
            
            self._session_active = False
            self._session_span_id = None
            
        except Exception as e:
            logger.error(f"Error ending session: {e}")
    
    async def _handle_error(self, error: Exception, agent):
        """Handle errors and emit appropriate events (async)."""
        try:
            agent_id = agent.name if hasattr(agent, 'name') else str(id(agent))
            agent_name = agent.name if hasattr(agent, 'name') else "unnamed_agent"
            error_msg = str(error)
            
            # Check if this is a retryable error
            if self._is_retryable_error(error_msg):
                # Track retry attempt
                agent_key = f"{agent_id}_{agent.model if hasattr(agent, 'model') else 'unknown'}"
                retry_count = self._agent_retry_attempts.get(agent_key, 0)
                self._agent_retry_attempts[agent_key] = retry_count + 1
                
                # Send RETRY event
                retry_event = self.event_builder.create_retry(
                    attempt=retry_count + 1,
                    strategy="exponential",
                    backoff_ms=1000 * (2 ** retry_count),
                    reason=f"Agent execution failed: {error_msg}",
                    agent_id=agent_id,
                    agent_name=agent_name
                )
                await self.tracer.client.send_event(retry_event)
            
            # Send ERROR event
            error_event = self.event_builder.create_error(
                error_message=error_msg,
                error_code=type(error).__name__,
                recoverable=self._is_retryable_error(error_msg),
                agent_id=agent_id,
                agent_name=agent_name
            )
            await self.tracer.client.send_event(error_event)
            
        except Exception as e:
            logger.error(f"Error handling error: {e}")
    
    def _handle_error_sync(self, error: Exception, agent):
        """Handle errors and emit appropriate events (sync)."""
        try:
            agent_id = agent.name if hasattr(agent, 'name') else str(id(agent))
            agent_name = agent.name if hasattr(agent, 'name') else "unnamed_agent"
            error_msg = str(error)
            
            # Check if this is a retryable error
            if self._is_retryable_error(error_msg):
                # Track retry attempt
                agent_key = f"{agent_id}_{agent.model if hasattr(agent, 'model') else 'unknown'}"
                retry_count = self._agent_retry_attempts.get(agent_key, 0)
                self._agent_retry_attempts[agent_key] = retry_count + 1
                
                # Send RETRY event
                retry_event = self.event_builder.create_retry(
                    attempt=retry_count + 1,
                    strategy="exponential",
                    backoff_ms=1000 * (2 ** retry_count),
                    reason=f"Agent execution failed: {error_msg}",
                    agent_id=agent_id,
                    agent_name=agent_name
                )
                self._send_event_sync(retry_event)
            
            # Send ERROR event
            error_event = self.event_builder.create_error(
                error_message=error_msg,
                error_code=type(error).__name__,
                recoverable=self._is_retryable_error(error_msg),
                agent_id=agent_id,
                agent_name=agent_name
            )
            self._send_event_sync(error_event)
            
        except Exception as e:
            logger.error(f"Error handling error: {e}")
    
    def _is_mcp_tool(self, tool) -> bool:
        """Check if a tool is an MCP tool."""
        # Check for HostedMCPTool type
        tool_type = type(tool).__name__
        if 'MCP' in tool_type or 'mcp' in tool_type:
            return True
        
        # Check for MCP-related attributes
        if hasattr(tool, 'server_url') or hasattr(tool, 'protocol'):
            return True
        
        return False
    
    def _is_internal_tool(self, tool) -> bool:
        """Check if a tool is an internal/system tool that shouldn't be tracked."""
        tool_name = tool.name if hasattr(tool, 'name') else str(tool)
        internal_tools = ['transfer_to_agent', 'handoff', 'system']
        return any(internal in tool_name.lower() for internal in internal_tools)
    
    def _is_data_access_tool(self, tool) -> bool:
        """Check if a tool involves data access."""
        tool_type = type(tool).__name__
        data_tools = ['FileSearchTool', 'WebSearchTool', 'CodeInterpreterTool']
        return any(dt in tool_type for dt in data_tools)
    
    def _get_datasource_name(self, tool) -> str:
        """Get the datasource name for a data access tool."""
        tool_type = type(tool).__name__
        if 'FileSearch' in tool_type:
            return "file_search"
        elif 'WebSearch' in tool_type:
            return "web_search"
        elif 'CodeInterpreter' in tool_type:
            return "code_interpreter"
        else:
            return "unknown"
    
    def _is_retryable_error(self, error_msg: str) -> bool:
        """Check if an error is retryable."""
        retryable_patterns = [
            "rate limit",
            "timeout",
            "connection",
            "temporary",
            "503",
            "429",
            "network",
            "unavailable"
        ]
        
        error_lower = error_msg.lower()
        return any(pattern in error_lower for pattern in retryable_patterns)
    
    def _send_event_sync(self, event):
        """Helper to send event from sync context."""
        import asyncio
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
                'cpu_percent': process.cpu_percent(),
                'memory_mb': process.memory_info().rss / 1024 / 1024,
                'num_threads': process.num_threads()
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
                "CRITICAL": Severity.SEVERITY_CRITICAL
            }
            severity = severity_map.get(severity_str, Severity.SEVERITY_INFO)

            system_event = self.event_builder.create_system_event(
                message=message,
                severity=severity,
                metadata={"framework": "openai_agents"}
            )
            self._send_event_sync(system_event)
        except Exception as e:
            logger.debug(f"Failed to emit system event: {e}")

    async def _emit_system_event(self, message: str, severity_str: str = "INFO", agent_id: Optional[str] = None, agent_name: Optional[str] = None):
        """Emit a SYSTEM_EVENT asynchronously."""
        try:
            from chaukas.spec.common.v1.events_pb2 import Severity

            severity_map = {
                "DEBUG": Severity.SEVERITY_DEBUG,
                "INFO": Severity.SEVERITY_INFO,
                "WARNING": Severity.SEVERITY_WARNING,
                "ERROR": Severity.SEVERITY_ERROR,
                "CRITICAL": Severity.SEVERITY_CRITICAL
            }
            severity = severity_map.get(severity_str, Severity.SEVERITY_INFO)

            system_event = self.event_builder.create_system_event(
                message=message,
                severity=severity,
                metadata={"framework": "openai_agents"},
                agent_id=agent_id,
                agent_name=agent_name
            )
            await self.tracer.client.send_event(system_event)
        except Exception as e:
            logger.debug(f"Failed to emit system event: {e}")

    async def _emit_state_update(self, agent_id: str, agent_name: str, state_data: Dict[str, Any]):
        """Emit a STATE_UPDATE event."""
        try:
            state_event = self.event_builder.create_state_update(
                state_data=state_data,
                agent_id=agent_id,
                agent_name=agent_name
            )
            await self.tracer.client.send_event(state_event)
        except Exception as e:
            logger.debug(f"Failed to emit state update event: {e}")

    async def _emit_policy_decision(self, policy_id: str, outcome: str, rule_ids: List[str], rationale: Optional[str] = None, agent_id: Optional[str] = None, agent_name: Optional[str] = None):
        """Emit a POLICY_DECISION event."""
        try:
            policy_event = self.event_builder.create_policy_decision(
                policy_id=policy_id,
                outcome=outcome,
                rule_ids=rule_ids,
                rationale=rationale,
                agent_id=agent_id,
                agent_name=agent_name
            )
            await self.tracer.client.send_event(policy_event)
        except Exception as e:
            logger.debug(f"Failed to emit policy decision event: {e}")

    def _track_agent_state(self, agent_id: str, agent) -> Dict[str, Any]:
        """Track agent state changes and return state diff."""
        current_state = {
            "model": agent.model if hasattr(agent, 'model') else None,
            "instructions": agent.instructions if hasattr(agent, 'instructions') else None,
            "tools_count": len(agent.tools) if hasattr(agent, 'tools') and agent.tools else 0,
            "temperature": getattr(agent, 'temperature', None),
            "max_tokens": getattr(agent, 'max_tokens', None),
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