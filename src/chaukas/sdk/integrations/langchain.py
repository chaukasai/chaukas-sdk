"""
LangChain integration for Chaukas instrumentation.
Uses BaseCallbackHandler for comprehensive event capture with ~95% event coverage.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional

# Optional performance monitoring
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from chaukas.spec.common.v1.events_pb2 import EventStatus

from chaukas.sdk.core.event_builder import EventBuilder
from chaukas.sdk.core.tracer import ChaukasTracer

# Try to import LangChain's BaseCallbackHandler
try:
    from langchain_core.callbacks.base import BaseCallbackHandler
except ImportError:
    try:
        from langchain.callbacks.base import BaseCallbackHandler
    except ImportError:
        # LangChain not installed, create a dummy base class
        logger = logging.getLogger(__name__)
        logger.debug("LangChain not installed, using dummy BaseCallbackHandler")
        BaseCallbackHandler = object

logger = logging.getLogger(__name__)


class LangChainWrapper:
    """Wrapper for LangChain instrumentation using BaseCallbackHandler."""

    def __init__(self, tracer: ChaukasTracer):
        self.tracer = tracer
        self.event_builder = EventBuilder()
        self.callback_handler = None
        self._session_active = False
        self._session_span_id = None
        self._start_time = None
        self._start_metrics = None
        self._original_default_callbacks = None

    def get_callback_handler(self):
        """Get or create the Chaukas callback handler instance."""
        if self.callback_handler is None:
            self.callback_handler = ChaukasCallbackHandler(self)
        return self.callback_handler

    def auto_instrument(self):
        """
        Automatically inject Chaukas callback by patching LangChain's Runnable classes.
        This enables one-line setup: just call enable_chaukas() and all LangChain
        operations are automatically instrumented.
        """
        try:
            # Import LangChain's base Runnable and RunnableSequence
            try:
                from langchain_core.runnables import Runnable
                from langchain_core.runnables.base import RunnableSequence
            except ImportError:
                try:
                    from langchain.schema.runnable import Runnable, RunnableSequence
                except ImportError:
                    # If RunnableSequence import fails, just use Runnable
                    from langchain.schema.runnable import Runnable

                    RunnableSequence = None

            # Get our callback handler
            chaukas_callback = self.get_callback_handler()

            # Store original methods
            self._original_runnable_invoke = Runnable.invoke
            self._original_runnable_ainvoke = (
                Runnable.ainvoke if hasattr(Runnable, "ainvoke") else None
            )

            if RunnableSequence:
                self._original_seq_invoke = RunnableSequence.invoke
                self._original_seq_ainvoke = (
                    RunnableSequence.ainvoke
                    if hasattr(RunnableSequence, "ainvoke")
                    else None
                )

            # Store references for closure to avoid confusing self capture
            original_runnable_invoke = self._original_runnable_invoke
            original_runnable_ainvoke = self._original_runnable_ainvoke
            original_seq_invoke = (
                self._original_seq_invoke if RunnableSequence else None
            )
            original_seq_ainvoke = (
                self._original_seq_ainvoke if RunnableSequence else None
            )

            # Create wrapped invoke method
            def wrapped_invoke(self_runnable, input, config=None, **kwargs):
                """Wrapped invoke that automatically includes Chaukas callback."""
                if config is None:
                    config = {}
                if "callbacks" not in config:
                    config["callbacks"] = []
                if chaukas_callback not in config["callbacks"]:
                    config["callbacks"].append(chaukas_callback)

                # Call the original method
                if RunnableSequence and isinstance(self_runnable, RunnableSequence):
                    return original_seq_invoke(
                        self_runnable, input, config=config, **kwargs
                    )
                else:
                    return original_runnable_invoke(
                        self_runnable, input, config=config, **kwargs
                    )

            # Create wrapped async invoke method
            async def wrapped_ainvoke(self_runnable, input, config=None, **kwargs):
                """Wrapped async invoke that automatically includes Chaukas callback."""
                if config is None:
                    config = {}
                if "callbacks" not in config:
                    config["callbacks"] = []
                if chaukas_callback not in config["callbacks"]:
                    config["callbacks"].append(chaukas_callback)

                # Call the original method
                if RunnableSequence and isinstance(self_runnable, RunnableSequence):
                    return await original_seq_ainvoke(
                        self_runnable, input, config=config, **kwargs
                    )
                else:
                    return await original_runnable_ainvoke(
                        self_runnable, input, config=config, **kwargs
                    )

            # Patch the methods
            Runnable.invoke = wrapped_invoke
            if self._original_runnable_ainvoke:
                Runnable.ainvoke = wrapped_ainvoke

            # Also patch RunnableSequence if available
            if RunnableSequence:
                RunnableSequence.invoke = wrapped_invoke
                if self._original_seq_ainvoke:
                    RunnableSequence.ainvoke = wrapped_ainvoke

            logger.info(
                "LangChain auto-instrumentation enabled - all operations will be tracked automatically"
            )
            return True

        except ImportError as e:
            logger.debug(f"LangChain not installed: {e}")
            return False
        except Exception as e:
            logger.warning(
                f"Failed to auto-instrument LangChain: {e}. You can still use chaukas.get_langchain_callback() manually."
            )
            return False

    def remove_auto_instrument(self):
        """Remove automatic instrumentation by restoring original methods."""
        try:
            try:
                from langchain_core.runnables import Runnable
                from langchain_core.runnables.base import RunnableSequence
            except ImportError:
                try:
                    from langchain.schema.runnable import Runnable, RunnableSequence
                except ImportError:
                    from langchain.schema.runnable import Runnable

                    RunnableSequence = None

            if hasattr(self, "_original_runnable_invoke"):
                Runnable.invoke = self._original_runnable_invoke
            if (
                hasattr(self, "_original_runnable_ainvoke")
                and self._original_runnable_ainvoke
            ):
                Runnable.ainvoke = self._original_runnable_ainvoke

            if RunnableSequence:
                if hasattr(self, "_original_seq_invoke"):
                    RunnableSequence.invoke = self._original_seq_invoke
                if (
                    hasattr(self, "_original_seq_ainvoke")
                    and self._original_seq_ainvoke
                ):
                    RunnableSequence.ainvoke = self._original_seq_ainvoke

            logger.info("LangChain auto-instrumentation removed")
        except Exception as e:
            logger.debug(f"Failed to remove auto-instrumentation: {e}")

    def _send_event_sync(self, event):
        """Helper to send event from sync context."""
        import asyncio

        try:
            # Try to get the running loop
            asyncio.get_running_loop()
            # We're in an async context, schedule the task
            asyncio.create_task(self.tracer.client.send_event(event))
        except RuntimeError:
            # No running loop - create a temporary one without global manipulation
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(self.tracer.client.send_event(event))
            finally:
                loop.close()

    async def _send_event_async(self, event):
        """Helper to send event from async context."""
        try:
            await self.tracer.client.send_event(event)
        except Exception as e:
            logger.error(f"Failed to send event: {e}")

    def _get_performance_metrics(self):
        """Collect current performance metrics."""
        if not PSUTIL_AVAILABLE:
            return {}

        try:
            import os

            process = psutil.Process(os.getpid())
            return {
                "cpu_percent": process.cpu_percent(),
                "memory_mb": process.memory_info().rss / 1024 / 1024,
                "num_threads": process.num_threads(),
            }
        except Exception:
            return {}


class ChaukasCallbackHandler(BaseCallbackHandler):
    """
    LangChain BaseCallbackHandler implementation for Chaukas event capture.

    Provides comprehensive event coverage by hooking into LangChain's callback system:
    - Chain lifecycle → SESSION events
    - LLM calls → MODEL_INVOCATION events
    - Agent actions → AGENT events
    - Tool usage → TOOL_CALL events
    - Retriever calls → DATA_ACCESS events
    - Errors → ERROR events
    - Retries → RETRY events
    """

    def __init__(self, wrapper: LangChainWrapper):
        """Initialize the callback handler."""
        super().__init__()

        self.wrapper = wrapper
        self.event_builder = wrapper.event_builder
        self.tracer = wrapper.tracer

        # Standard LangChain callback handler attributes
        self.raise_error = False  # Don't raise errors from callbacks
        self.run_inline = False  # Run callbacks in same thread/process

        # Track span IDs for START/END event pairing
        self._chain_spans = {}  # Map chain run_id to span_id
        self._llm_spans = {}  # Map LLM run_id to span_id
        self._tool_spans = {}  # Map tool run_id to span_id
        self._agent_spans = {}  # Map agent run_id to span_id
        self._retriever_spans = {}  # Map retriever run_id to span_id

        # Track retry attempts
        self._llm_retry_attempts = {}
        self._tool_retry_attempts = {}
        self._chain_retry_attempts = {}

        # Track session context
        self._session_started = False
        self._session_span_id = None
        self._root_chain_id = None

        # Track agent context for handoffs
        self._last_active_agent = None

        # Track start times for duration calculation
        self._start_times = {}

    # Chain lifecycle callbacks → SESSION/AGENT events

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Run when chain starts.
        Maps to SESSION_START (for root chains) or AGENT_START (for agent chains).
        """
        try:
            self._start_times[str(run_id)] = time.time()

            # Handle None serialized parameter
            if serialized is None:
                chain_type = "unknown"
            else:
                chain_type = serialized.get("name", "unknown")
            is_agent_chain = "agent" in chain_type.lower()
            is_root_chain = parent_run_id is None

            # Root chain → SESSION_START
            if is_root_chain and not self._session_started:
                self._session_started = True
                self._root_chain_id = str(run_id)

                session_start = self.event_builder.create_session_start(
                    metadata={
                        "framework": "langchain",
                        "chain_type": chain_type,
                        "chain_id": str(run_id),
                        "inputs": str(inputs)[:500],
                        "tags": tags or [],
                    }
                )
                self.wrapper._send_event_sync(session_start)
                self._session_span_id = session_start.span_id
                self._chain_spans[str(run_id)] = session_start.span_id

                # Set session context
                self.tracer.set_session_context(
                    session_start.session_id, session_start.trace_id
                )
                self.tracer.set_parent_span_context(self._session_span_id)

                # Emit INPUT_RECEIVED
                if inputs:
                    input_event = self.event_builder.create_input_received(
                        content=str(inputs)[:1000],
                        metadata={"chain_type": chain_type, "run_id": str(run_id)},
                    )
                    self.wrapper._send_event_sync(input_event)

            # Agent chain → AGENT_START
            elif is_agent_chain:
                agent_id = str(run_id)
                agent_name = chain_type

                # Check for agent handoff (only between different agent types)
                # Compare only agent_name to avoid false positives from new run_ids
                if self._last_active_agent:
                    last_id, last_name = self._last_active_agent
                    # Only trigger handoff if switching to a different agent type
                    if last_name != agent_name:
                        handoff_event = self.event_builder.create_agent_handoff(
                            from_agent_id=last_id,
                            from_agent_name=last_name,
                            to_agent_id=agent_id,
                            to_agent_name=agent_name,
                            reason="Chain delegation",
                            handoff_type="sequential",
                            handoff_data={
                                "framework": "langchain",
                                "chain_type": chain_type,
                            },
                        )
                        self.wrapper._send_event_sync(handoff_event)

                self._last_active_agent = (agent_id, agent_name)

                agent_start = self.event_builder.create_agent_start(
                    agent_id=agent_id,
                    agent_name=agent_name,
                    role="agent",
                    instructions=metadata.get("instructions") if metadata else None,
                    tools=tags or [],
                    metadata={
                        "framework": "langchain",
                        "chain_type": chain_type,
                        "inputs": str(inputs)[:500],
                    },
                )
                self.wrapper._send_event_sync(agent_start)
                self._agent_spans[agent_id] = agent_start.span_id
                self._chain_spans[str(run_id)] = agent_start.span_id

        except Exception as e:
            logger.error(f"Error in on_chain_start: {e}", exc_info=True)

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> Any:
        """
        Run when chain ends.
        Maps to SESSION_END (for root chains) or AGENT_END (for agent chains).
        """
        try:
            run_id_str = str(run_id)
            span_id = self._chain_spans.pop(run_id_str, None)

            # Calculate duration
            duration_ms = None
            if run_id_str in self._start_times:
                duration_ms = (time.time() - self._start_times.pop(run_id_str)) * 1000

            # Clear retry counter
            self._chain_retry_attempts.pop(run_id_str, None)

            # Root chain → SESSION_END
            if run_id_str == self._root_chain_id:
                # Emit OUTPUT_EMITTED
                if outputs:
                    output_event = self.event_builder.create_output_emitted(
                        content=str(outputs)[:1000], metadata={"run_id": run_id_str}
                    )
                    self.wrapper._send_event_sync(output_event)

                # Emit SESSION_END
                metrics = self.wrapper._get_performance_metrics()
                session_end = self.event_builder.create_session_end(
                    span_id=span_id,
                    metadata={
                        "framework": "langchain",
                        "success": True,
                        "duration_ms": duration_ms,
                        "cpu_percent": metrics.get("cpu_percent"),
                        "memory_mb": metrics.get("memory_mb"),
                    },
                )
                self.wrapper._send_event_sync(session_end)

                self._session_started = False
                self._root_chain_id = None

            # Agent chain → AGENT_END
            elif run_id_str in self._agent_spans:
                agent_id = run_id_str
                agent_span_id = self._agent_spans.pop(agent_id, span_id)

                agent_end = self.event_builder.create_agent_end(
                    agent_id=agent_id,
                    agent_name="agent",
                    status=EventStatus.EVENT_STATUS_COMPLETED,
                    span_id=agent_span_id,
                    metadata={
                        "framework": "langchain",
                        "outputs": str(outputs)[:500],
                        "duration_ms": duration_ms,
                    },
                )
                self.wrapper._send_event_sync(agent_end)

        except Exception as e:
            logger.error(f"Error in on_chain_end: {e}", exc_info=True)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Run when chain errors. Maps to ERROR + RETRY events."""
        try:
            run_id_str = str(run_id)
            error_msg = str(error)

            # Check for retry
            if self._is_retryable_error(error_msg):
                retry_count = self._chain_retry_attempts.get(run_id_str, 0)
                self._chain_retry_attempts[run_id_str] = retry_count + 1

                if retry_count < 3:
                    retry_event = self.event_builder.create_retry(
                        attempt=retry_count + 1,
                        strategy="exponential",
                        backoff_ms=1000 * (2**retry_count),
                        reason=f"Chain execution failed: {error_msg}",
                    )
                    self.wrapper._send_event_sync(retry_event)

            # Emit ERROR
            error_event = self.event_builder.create_error(
                error_message=error_msg,
                error_code=type(error).__name__,
                recoverable=self._is_retryable_error(error_msg),
            )
            self.wrapper._send_event_sync(error_event)

            # Clean up
            self._chain_spans.pop(run_id_str, None)
            if run_id_str in self._start_times:
                self._start_times.pop(run_id_str)

        except Exception as e:
            logger.error(f"Error in on_chain_error: {e}", exc_info=True)

    # LLM callbacks → MODEL_INVOCATION events

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when LLM starts. Maps to MODEL_INVOCATION_START."""
        try:
            self._start_times[str(run_id)] = time.time()

            # Extract LLM details
            model = serialized.get("name", "unknown") if serialized else "unknown"
            provider = self._extract_provider(model)

            # Convert prompts to messages format
            messages = [{"role": "user", "content": prompt} for prompt in prompts]

            # Get agent context if available
            agent_id, agent_name = self._get_current_agent_context()

            llm_start = self.event_builder.create_model_invocation_start(
                provider=provider,
                model=model,
                messages=messages,
                agent_id=agent_id,
                agent_name=agent_name,
                temperature=kwargs.get("temperature"),
                max_tokens=kwargs.get("max_tokens"),
                tools=tags,
            )
            self.wrapper._send_event_sync(llm_start)
            self._llm_spans[str(run_id)] = llm_start.span_id

        except Exception as e:
            logger.error(f"Error in on_llm_start: {e}", exc_info=True)

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[Any]],
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when chat model starts. Maps to MODEL_INVOCATION_START."""
        try:
            self._start_times[str(run_id)] = time.time()

            # Extract model details
            model = serialized.get("name", "unknown") if serialized else "unknown"
            provider = self._extract_provider(model)

            # Flatten and convert messages
            formatted_messages = []
            for message_list in messages:
                for msg in message_list:
                    if hasattr(msg, "type") and hasattr(msg, "content"):
                        formatted_messages.append(
                            {"role": msg.type, "content": str(msg.content)}
                        )
                    else:
                        formatted_messages.append({"role": "user", "content": str(msg)})

            # Get agent context
            agent_id, agent_name = self._get_current_agent_context()

            llm_start = self.event_builder.create_model_invocation_start(
                provider=provider,
                model=model,
                messages=formatted_messages,
                agent_id=agent_id,
                agent_name=agent_name,
                temperature=kwargs.get("temperature"),
                max_tokens=kwargs.get("max_tokens"),
                tools=tags,
            )
            self.wrapper._send_event_sync(llm_start)
            self._llm_spans[str(run_id)] = llm_start.span_id

        except Exception as e:
            logger.error(f"Error in on_chat_model_start: {e}", exc_info=True)

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Run when LLM ends. Maps to MODEL_INVOCATION_END."""
        try:
            run_id_str = str(run_id)
            span_id = self._llm_spans.pop(run_id_str, None)

            # Calculate duration
            duration_ms = None
            if run_id_str in self._start_times:
                duration_ms = (time.time() - self._start_times.pop(run_id_str)) * 1000

            # Clear retry counter
            llm_key = f"{run_id_str}"
            self._llm_retry_attempts.pop(llm_key, None)

            # Extract response details
            response_content = None
            tool_calls = None
            finish_reason = None
            prompt_tokens = None
            completion_tokens = None
            total_tokens = None

            if hasattr(response, "generations") and response.generations:
                gen = response.generations[0][0]
                if hasattr(gen, "text"):
                    response_content = gen.text
                if hasattr(gen, "message") and hasattr(gen.message, "content"):
                    response_content = gen.message.content
                # Extract tool calls if present
                if hasattr(gen, "message") and hasattr(
                    gen.message, "additional_kwargs"
                ):
                    tool_calls_data = gen.message.additional_kwargs.get("tool_calls")
                    if tool_calls_data:
                        tool_calls = tool_calls_data

            # Extract token usage
            if hasattr(response, "llm_output") and response.llm_output:
                token_usage = response.llm_output.get("token_usage", {})
                prompt_tokens = token_usage.get("prompt_tokens")
                completion_tokens = token_usage.get("completion_tokens")
                total_tokens = token_usage.get("total_tokens")

            # Get agent context
            agent_id, agent_name = self._get_current_agent_context()

            # Extract model name and provider
            model_name = kwargs.get("name", "unknown")
            llm_end = self.event_builder.create_model_invocation_end(
                provider=self._extract_provider(model_name),
                model=model_name,
                response_content=response_content,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                duration_ms=duration_ms,
                span_id=span_id,
                agent_id=agent_id,
                agent_name=agent_name,
                error=None,
            )
            self.wrapper._send_event_sync(llm_end)

        except Exception as e:
            logger.error(f"Error in on_llm_end: {e}", exc_info=True)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Run when LLM errors. Maps to ERROR + RETRY + MODEL_INVOCATION_END."""
        try:
            run_id_str = str(run_id)
            span_id = self._llm_spans.pop(run_id_str, None)
            error_msg = str(error)

            # Check for retry
            if self._is_retryable_error(error_msg):
                llm_key = run_id_str
                retry_count = self._llm_retry_attempts.get(llm_key, 0)
                self._llm_retry_attempts[llm_key] = retry_count + 1

                if retry_count < 3:
                    retry_event = self.event_builder.create_retry(
                        attempt=retry_count + 1,
                        strategy="exponential",
                        backoff_ms=1000 * (2**retry_count),
                        reason=f"LLM call failed: {error_msg}",
                    )
                    self.wrapper._send_event_sync(retry_event)

            # Get agent context
            agent_id, agent_name = self._get_current_agent_context()

            # Emit ERROR
            error_event = self.event_builder.create_error(
                error_message=error_msg,
                error_code=type(error).__name__,
                recoverable=self._is_retryable_error(error_msg),
                agent_id=agent_id,
                agent_name=agent_name,
            )
            self.wrapper._send_event_sync(error_event)

            # Emit MODEL_INVOCATION_END with error
            model_name = kwargs.get("name", "unknown")
            llm_end = self.event_builder.create_model_invocation_end(
                provider=self._extract_provider(model_name),
                model=model_name,
                response_content=None,
                tool_calls=None,
                finish_reason="error",
                span_id=span_id,
                agent_id=agent_id,
                agent_name=agent_name,
                error=error_msg,
            )
            self.wrapper._send_event_sync(llm_end)

            # Clean up
            if run_id_str in self._start_times:
                self._start_times.pop(run_id_str)

        except Exception as e:
            logger.error(f"Error in on_llm_error: {e}", exc_info=True)

    # Tool callbacks → TOOL_CALL events

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when tool starts. Maps to TOOL_CALL_START."""
        try:
            self._start_times[str(run_id)] = time.time()

            tool_name = serialized.get("name", "unknown") if serialized else "unknown"

            # Parse arguments
            arguments = {}
            if inputs:
                arguments = inputs
            elif input_str:
                try:
                    arguments = (
                        json.loads(input_str)
                        if input_str.startswith("{")
                        else {"input": input_str}
                    )
                except (json.JSONDecodeError, ValueError):
                    arguments = {"input": input_str}

            # Get agent context
            agent_id, agent_name = self._get_current_agent_context()

            tool_start = self.event_builder.create_tool_call_start(
                tool_name=tool_name,
                arguments=arguments,
                call_id=str(run_id),
                agent_id=agent_id,
                agent_name=agent_name,
            )
            self.wrapper._send_event_sync(tool_start)
            self._tool_spans[str(run_id)] = tool_start.span_id

        except Exception as e:
            logger.error(f"Error in on_tool_start: {e}", exc_info=True)

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Run when tool ends. Maps to TOOL_CALL_END."""
        try:
            run_id_str = str(run_id)
            span_id = self._tool_spans.pop(run_id_str, None)

            # Calculate duration
            duration_ms = None
            if run_id_str in self._start_times:
                duration_ms = (time.time() - self._start_times.pop(run_id_str)) * 1000

            # Clear retry counter
            tool_key = run_id_str
            self._tool_retry_attempts.pop(tool_key, None)

            # Get agent context
            agent_id, agent_name = self._get_current_agent_context()

            tool_end = self.event_builder.create_tool_call_end(
                tool_name=kwargs.get("name", "unknown"),
                call_id=run_id_str,
                output=str(output)[:1000] if output else None,
                error=None,
                execution_time_ms=duration_ms,
                span_id=span_id,
                agent_id=agent_id,
                agent_name=agent_name,
            )
            self.wrapper._send_event_sync(tool_end)

        except Exception as e:
            logger.error(f"Error in on_tool_end: {e}", exc_info=True)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Run when tool errors. Maps to ERROR + RETRY + TOOL_CALL_END."""
        try:
            run_id_str = str(run_id)
            span_id = self._tool_spans.pop(run_id_str, None)
            error_msg = str(error)

            # Check for retry
            if self._is_retryable_error(error_msg):
                tool_key = run_id_str
                retry_count = self._tool_retry_attempts.get(tool_key, 0)
                self._tool_retry_attempts[tool_key] = retry_count + 1

                if retry_count < 2:
                    retry_event = self.event_builder.create_retry(
                        attempt=retry_count + 1,
                        strategy="linear",
                        backoff_ms=500 * (retry_count + 1),
                        reason=f"Tool execution failed: {error_msg}",
                    )
                    self.wrapper._send_event_sync(retry_event)

            # Get agent context
            agent_id, agent_name = self._get_current_agent_context()

            # Emit ERROR
            error_event = self.event_builder.create_error(
                error_message=error_msg,
                error_code=type(error).__name__,
                recoverable=self._is_retryable_error(error_msg),
                agent_id=agent_id,
                agent_name=agent_name,
            )
            self.wrapper._send_event_sync(error_event)

            # Emit TOOL_CALL_END with error
            tool_end = self.event_builder.create_tool_call_end(
                tool_name=kwargs.get("name", "unknown"),
                call_id=run_id_str,
                output=None,
                error=error_msg,
                execution_time_ms=None,
                span_id=span_id,
                agent_id=agent_id,
                agent_name=agent_name,
            )
            self.wrapper._send_event_sync(tool_end)

            # Clean up
            if run_id_str in self._start_times:
                self._start_times.pop(run_id_str)

        except Exception as e:
            logger.error(f"Error in on_tool_error: {e}", exc_info=True)

    # Agent action callbacks → AGENT events

    def on_agent_action(
        self,
        action: Any,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> Any:
        """
        Run on agent action.
        Can emit AGENT_START if agent hasn't started yet.
        """
        try:
            # Extract action details
            tool = action.tool if hasattr(action, "tool") else "unknown"
            tool_input = action.tool_input if hasattr(action, "tool_input") else {}

            logger.debug(f"Agent action: tool={tool}, input={tool_input}")

        except Exception as e:
            logger.error(f"Error in on_agent_action: {e}", exc_info=True)

    def on_agent_finish(
        self,
        finish: Any,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> Any:
        """
        Run when agent ends.
        Maps to OUTPUT_EMITTED (output is captured in chain_end).
        """
        try:
            # Extract finish details
            output = finish.return_values if hasattr(finish, "return_values") else {}

            # Emit OUTPUT_EMITTED if we have output
            if output:
                output_event = self.event_builder.create_output_emitted(
                    content=str(output)[:1000], metadata={"run_id": str(run_id)}
                )
                self.wrapper._send_event_sync(output_event)

        except Exception as e:
            logger.error(f"Error in on_agent_finish: {e}", exc_info=True)

    # Retriever callbacks → DATA_ACCESS events

    def on_retriever_start(
        self,
        serialized: Dict[str, Any],
        query: str,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when retriever starts. Maps to DATA_ACCESS."""
        try:
            self._start_times[str(run_id)] = time.time()

            retriever_name = (
                serialized.get("name", "unknown") if serialized else "unknown"
            )

            # Get agent context
            agent_id, agent_name = self._get_current_agent_context()

            data_event = self.event_builder.create_data_access(
                datasource=retriever_name,
                document_ids=None,
                chunk_ids=None,
                pii_categories=None,
                agent_id=agent_id,
                agent_name=agent_name,
            )
            self.wrapper._send_event_sync(data_event)
            self._retriever_spans[str(run_id)] = data_event.span_id

        except Exception as e:
            logger.error(f"Error in on_retriever_start: {e}", exc_info=True)

    def on_retriever_end(
        self,
        documents: Any,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Run when retriever ends. Maps to DATA_ACCESS with results."""
        try:
            run_id_str = str(run_id)
            # Clean up span tracking
            self._retriever_spans.pop(run_id_str, None)

            # Extract document IDs
            doc_ids = []
            if documents:
                for doc in documents:
                    if hasattr(doc, "metadata") and "id" in doc.metadata:
                        doc_ids.append(doc.metadata["id"])

            # Get agent context
            agent_id, agent_name = self._get_current_agent_context()

            data_event = self.event_builder.create_data_access(
                datasource="retriever",
                document_ids=doc_ids if doc_ids else None,
                chunk_ids=None,
                pii_categories=None,
                agent_id=agent_id,
                agent_name=agent_name,
            )
            self.wrapper._send_event_sync(data_event)

            # Clean up
            if run_id_str in self._start_times:
                self._start_times.pop(run_id_str)

        except Exception as e:
            logger.error(f"Error in on_retriever_end: {e}", exc_info=True)

    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Run when retriever errors. Maps to ERROR."""
        try:
            run_id_str = str(run_id)
            error_msg = str(error)

            # Get agent context
            agent_id, agent_name = self._get_current_agent_context()

            # Emit ERROR
            error_event = self.event_builder.create_error(
                error_message=error_msg,
                error_code=type(error).__name__,
                recoverable=self._is_retryable_error(error_msg),
                agent_id=agent_id,
                agent_name=agent_name,
            )
            self.wrapper._send_event_sync(error_event)

            # Clean up
            self._retriever_spans.pop(run_id_str, None)
            if run_id_str in self._start_times:
                self._start_times.pop(run_id_str)

        except Exception as e:
            logger.error(f"Error in on_retriever_error: {e}", exc_info=True)

    # Retry callback → RETRY event

    def on_retry(
        self,
        retry_state: Any,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Run on retry. Maps to RETRY event."""
        try:
            attempt = (
                retry_state.attempt_number
                if hasattr(retry_state, "attempt_number")
                else 1
            )

            retry_event = self.event_builder.create_retry(
                attempt=attempt,
                strategy="exponential",
                backoff_ms=1000 * (2 ** (attempt - 1)),
                reason=f"Retry attempt {attempt}",
            )
            self.wrapper._send_event_sync(retry_event)

        except Exception as e:
            logger.error(f"Error in on_retry: {e}", exc_info=True)

    # Helper methods

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
            "unavailable",
        ]
        error_lower = error_msg.lower()
        return any(pattern in error_lower for pattern in retryable_patterns)

    def _extract_provider(self, model_name: str) -> str:
        """Extract provider from model name."""
        model_lower = model_name.lower()
        if "openai" in model_lower or "gpt" in model_lower:
            return "openai"
        elif "anthropic" in model_lower or "claude" in model_lower:
            return "anthropic"
        elif (
            "google" in model_lower or "gemini" in model_lower or "palm" in model_lower
        ):
            return "google"
        elif "cohere" in model_lower:
            return "cohere"
        elif "huggingface" in model_lower:
            return "huggingface"
        else:
            return "unknown"

    def _get_current_agent_context(self) -> tuple:
        """Get current agent ID and name from context."""
        if self._last_active_agent:
            return self._last_active_agent
        return (None, None)
