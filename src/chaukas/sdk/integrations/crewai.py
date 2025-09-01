"""
CrewAI integration for Chaukas instrumentation.
Uses proto-compliant events with proper agent handoff tracking.
"""

import functools
import logging
from typing import Any, Dict, Optional, List
import json
import os
import time

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


class CrewAIWrapper:
    """Wrapper for CrewAI instrumentation using proto events."""
    
    def __init__(self, tracer: ChaukasTracer):
        self.tracer = tracer
        self.event_builder = EventBuilder()
        self._current_agents: List[Any] = []  # Track agents for handoff events
        self._last_active_agent = None  # Track last active agent for handoff detection
        self._original_kickoff = None
        self._original_kickoff_async = None
        self._original_kickoff_for_each = None
        self._original_kickoff_for_each_async = None
        self._original_execute_task = None
        self.event_listener = None  # Event bus listener
        self._start_time = None  # Track execution start time
        self._start_metrics = None  # Track initial performance metrics
        self._session_span_id = None  # Track SESSION_START span_id for hierarchy
        self._current_agent_span_id = None  # Track current agent span for child events
    
    def patch_crew(self):
        """Apply patches to all CrewAI Crew methods."""
        try:
            from crewai import Crew
            import asyncio
            
            # Store original methods
            self._original_kickoff = Crew.kickoff
            self._original_kickoff_async = getattr(Crew, 'kickoff_async', None)
            self._original_kickoff_for_each = getattr(Crew, 'kickoff_for_each', None)
            self._original_kickoff_for_each_async = getattr(Crew, 'kickoff_for_each_async', None)
            
            # Store wrapper reference for use in nested functions
            wrapper = self
            
            def _extract_crew_data(crew_instance, args, kwargs):
                """Extract crew configuration and input data."""
                agents = getattr(crew_instance, "agents", [])
                tasks = getattr(crew_instance, "tasks", [])
                crew_name = getattr(crew_instance, "name", "crew")
                process = getattr(crew_instance, "process", "unknown")
                
                # Extract inputs from kwargs or first arg
                inputs = kwargs.get('inputs') or (args[0] if args else None)
                
                return {
                    'agents': agents,
                    'tasks': tasks, 
                    'crew_name': crew_name,
                    'process': process,
                    'inputs': inputs
                }
            
            # Create sync kickoff wrapper
            @functools.wraps(wrapper._original_kickoff)
            def instrumented_kickoff(crew_instance, *args, **kwargs):
                """Instrumented sync version of Crew.kickoff()."""
                crew_data = _extract_crew_data(crew_instance, args, kwargs)
                
                # Reset last active agent for new crew execution
                wrapper._last_active_agent = None
                
                # Track start time and metrics
                wrapper._start_time = time.time()
                wrapper._start_metrics = wrapper._get_performance_metrics()
                
                # Send session start event
                session_start = self.event_builder.create_session_start(
                    metadata={
                        "crew_name": crew_data['crew_name'],
                        "method": "kickoff",
                        "framework": "crewai",
                        "agents_count": len(crew_data['agents']),
                        "tasks_count": len(crew_data['tasks']),
                    }
                )
                wrapper._send_event_sync(session_start)
                # Store session span_id for END event and as parent for execution events
                wrapper._session_span_id = session_start.span_id
                
                # Set complete context for all subsequent events
                # This ensures all events share the same session_id and trace_id
                session_tokens = wrapper.tracer.set_session_context(session_start.session_id, session_start.trace_id)
                parent_token = wrapper.tracer.set_parent_span_context(wrapper._session_span_id)
                
                try:
                    # Send input received event if inputs provided (will use session as parent)
                    if crew_data['inputs']:
                        input_event = wrapper.event_builder.create_input_received(
                            content=str(crew_data['inputs'])[:1000],
                            metadata={"input_type": "crew_inputs", "method": "kickoff"}
                        )
                        wrapper._send_event_sync(input_event)
                    
                    # Execute original method
                    result = wrapper._original_kickoff(crew_instance, *args, **kwargs)
                    
                    # Send output event
                    output_event = wrapper.event_builder.create_output_emitted(
                        content=str(result)[:1000] if result else "No result",
                        metadata={"output_type": "crew_result", "method": "kickoff"}
                    )
                    wrapper._send_event_sync(output_event)
                    
                    # Calculate performance metrics
                    duration_ms = (time.time() - wrapper._start_time) * 1000 if wrapper._start_time else None
                    end_metrics = wrapper._get_performance_metrics()
                    token_estimate = wrapper._estimate_tokens(crew_data.get('inputs'), result)
                    
                    # Send session end event with enhanced metadata
                    session_end = wrapper.event_builder.create_session_end(
                        span_id=wrapper._session_span_id,  # Use same span_id as SESSION_START
                        metadata={
                            "crew_name": crew_data['crew_name'],
                            "method": "kickoff",
                            "success": True,
                            "result_type": type(result).__name__,
                            "duration_ms": duration_ms,
                            "estimated_tokens": token_estimate,
                            "cpu_percent": end_metrics.get('cpu_percent'),
                            "memory_mb": end_metrics.get('memory_mb')
                        }
                    )
                    wrapper._send_event_sync(session_end)
                        
                    return result
                    
                except Exception as e:
                    # Send error event
                    error_event = wrapper.event_builder.create_error(
                        error_message=str(e),
                        error_code=type(e).__name__,
                        recoverable=True,
                    )
                    wrapper._send_event_sync(error_event)
                    raise
                finally:
                    # Reset contexts
                    wrapper.tracer.reset_parent_span_context(parent_token)
                    wrapper.tracer.reset_session_context(session_tokens)
            
            # Create async kickoff wrapper  
            @functools.wraps(wrapper._original_kickoff_async or wrapper._original_kickoff)
            async def instrumented_kickoff_async(crew_instance, *args, **kwargs):
                """Instrumented async version of Crew.kickoff_async()."""
                crew_data = _extract_crew_data(crew_instance, args, kwargs)
                
                # Send session start event
                session_start = wrapper.event_builder.create_session_start(
                    metadata={
                        "crew_name": crew_data['crew_name'],
                        "method": "kickoff_async",
                        "framework": "crewai",
                        "agents_count": len(crew_data['agents']),
                        "tasks_count": len(crew_data['tasks']),
                    }
                )
                await wrapper.tracer.client.send_event(session_start)
                # Store session span_id for END event and as parent for execution events
                wrapper._session_span_id = session_start.span_id
                
                # Set complete context for all subsequent events
                # This ensures all events share the same session_id and trace_id
                session_tokens = wrapper.tracer.set_session_context(session_start.session_id, session_start.trace_id)
                parent_token = wrapper.tracer.set_parent_span_context(wrapper._session_span_id)
                
                try:
                    # Send input received event if inputs provided (will use session as parent)
                    if crew_data['inputs']:
                        input_event = wrapper.event_builder.create_input_received(
                            content=str(crew_data['inputs'])[:1000],
                            metadata={"input_type": "crew_inputs", "method": "kickoff_async"}
                        )
                        await wrapper.tracer.client.send_event(input_event)
                    
                    # Execute original method
                    if wrapper._original_kickoff_async:
                        result = await wrapper._original_kickoff_async(crew_instance, *args, **kwargs)
                    else:
                        result = wrapper._original_kickoff(crew_instance, *args, **kwargs)
                        
                    # Send output event
                    output_event = wrapper.event_builder.create_output_emitted(
                        content=str(result)[:1000] if result else "No result",
                        metadata={"output_type": "crew_result", "method": "kickoff_async"}
                    )
                    await wrapper.tracer.client.send_event(output_event)
                    
                    # Send session end event
                    session_end = wrapper.event_builder.create_session_end(
                        span_id=wrapper._session_span_id,  # Use same span_id as SESSION_START
                        metadata={
                            "crew_name": crew_data['crew_name'],
                            "method": "kickoff_async",
                            "success": True,
                            "result_type": type(result).__name__
                        }
                    )
                    await wrapper.tracer.client.send_event(session_end)
                    
                    return result
                    
                except Exception as e:
                    # Send error event
                    error_event = wrapper.event_builder.create_error(
                        error_message=str(e),
                        error_code=type(e).__name__,
                        recoverable=True,
                    )
                    await wrapper.tracer.client.send_event(error_event)
                    raise
                finally:
                    # Reset contexts
                    wrapper.tracer.reset_parent_span_context(parent_token)
                    wrapper.tracer.reset_session_context(session_tokens)
            
            # Patch all methods
            Crew.kickoff = instrumented_kickoff
            if wrapper._original_kickoff_async:
                Crew.kickoff_async = instrumented_kickoff_async
            
            # TODO: Add kickoff_for_each and kickoff_for_each_async wrappers
            
            # Initialize and register event bus listener
            if not self.event_listener:
                logger.info("🎯 Creating CrewAIEventBusListener...")
                self.event_listener = CrewAIEventBusListener(self)
                logger.info("🎯 Registering event handlers...")
                self.event_listener.register_handlers()
                logger.info("✓ Event bus listener initialized and handlers registered")
            else:
                logger.debug("Event listener already exists, skipping creation")
            
            logger.info("Successfully patched Crew kickoff methods")
            return True
            
        except ImportError:
            logger.warning("CrewAI not installed, skipping Crew patching")
            return False
        except Exception as e:
            logger.error(f"Failed to patch Crew: {e}")
            return False
    
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
    
    def patch_agent(self):
        """Apply patches to CrewAI Agent class."""
        try:
            from crewai import Agent
            
            # Store original method
            self._original_execute_task = Agent.execute_task
            
            # Create instrumented version
            @functools.wraps(self._original_execute_task)
            def instrumented_execute_task(agent_instance, *args, **kwargs):
                """Instrumented version of Agent.execute_task()."""
                
                # Extract agent info using mapper
                agent_id, agent_name = AgentMapper.map_crewai_agent(agent_instance)
                
                # Don't create a span here - let AGENT_START be the parent for child events
                try:
                    # Extract task data
                    task = args[0] if args else None
                    
                    # Check for agent handoff
                    if self._last_active_agent and self._last_active_agent != (agent_id, agent_name):
                        last_id, last_name = self._last_active_agent
                        # Emit AGENT_HANDOFF event
                        handoff_event = self.event_builder.create_agent_handoff(
                            from_agent_id=last_id,
                            from_agent_name=last_name,
                            to_agent_id=agent_id,
                            to_agent_name=agent_name,
                            reason="Task delegation in sequential process",
                            handoff_type="sequential",
                            handoff_data={
                                "task": getattr(task, "description", None) if task else None,
                                "framework": "crewai"
                            }
                        )
                        self._send_event_sync(handoff_event)
                    
                    # Update last active agent
                    self._last_active_agent = (agent_id, agent_name)
                    
                    # Send agent start event
                    start_event = self.event_builder.create_agent_start(
                        agent_id=agent_id,
                        agent_name=agent_name,
                        role=getattr(agent_instance, "role", "unknown"),
                        instructions=getattr(agent_instance, "goal", None),
                        tools=[str(tool) for tool in getattr(agent_instance, "tools", [])],
                        metadata={
                            "framework": "crewai",
                            "backstory": getattr(agent_instance, "backstory", None),
                            "task_description": getattr(task, "description", None) if task else None,
                            "expected_output": getattr(task, "expected_output", None) if task else None,
                        }
                    )
                    self._send_event_sync(start_event)
                    
                    # Store agent span_id for END event
                    agent_span_id = start_event.span_id
                    
                    # Set the agent's span_id as the parent context for child events
                    # This ensures MODEL/TOOL events are children of AGENT_START
                    from chaukas.sdk.core.tracer import _parent_span_id
                    token = _parent_span_id.set(agent_span_id)
                    
                    try:
                        # Execute original method  
                        result = self._original_execute_task(agent_instance, *args, **kwargs)
                    finally:
                        # Reset parent context
                        _parent_span_id.reset(token)
                    
                    # Send agent end event with same span_id as START
                    end_event = self.event_builder.create_agent_end(
                        agent_id=agent_id,
                        agent_name=agent_name,
                        status=EventStatus.EVENT_STATUS_COMPLETED,
                        span_id=agent_span_id,  # Use same span_id as AGENT_START
                        metadata={
                            "framework": "crewai",
                            "result": str(result)[:500] if result else None,
                            "result_type": type(result).__name__,
                        }
                    )
                    self._send_event_sync(end_event)
                    
                    return result
                    
                except Exception as e:
                    # Send agent error event
                    error_event = self.event_builder.create_error(
                        error_message=str(e),
                        error_code=type(e).__name__,
                        recoverable=True,
                        agent_id=agent_id,
                        agent_name=agent_name
                    )
                    self._send_event_sync(error_event)
                    raise
            
            # Replace the method
            Agent.execute_task = instrumented_execute_task
            
            logger.info("Successfully patched Agent.execute_task")
            return True
            
        except ImportError:
            logger.warning("CrewAI not installed, skipping Agent patching")
            return False
        except Exception as e:
            logger.error(f"Failed to patch Agent: {e}")
            return False
    
    def unpatch_crew(self):
        """Restore original Crew.kickoff method."""
        # Unregister event bus handlers
        if self.event_listener:
            self.event_listener.unregister_handlers()
            self.event_listener = None
        
        if self._original_kickoff:
            try:
                from crewai import Crew
                Crew.kickoff = self._original_kickoff
                self._original_kickoff = None
                logger.info("Successfully unpatched Crew.kickoff")
            except Exception as e:
                logger.error(f"Failed to unpatch Crew: {e}")
    
    def unpatch_agent(self):
        """Restore original Agent.execute_task method."""
        if self._original_execute_task:
            try:
                from crewai import Agent
                Agent.execute_task = self._original_execute_task
                self._original_execute_task = None
                logger.info("Successfully unpatched Agent.execute_task")
            except Exception as e:
                logger.error(f"Failed to unpatch Agent: {e}")
    
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
    
    def _estimate_tokens(self, input_data, output_data):
        """Estimate token usage for the conversation."""
        def count_tokens_approx(data):
            """Rough approximation: ~4 characters per token."""
            if not data:
                return 0
            
            if isinstance(data, str):
                return len(data) // 4
            elif isinstance(data, dict):
                total = 0
                for value in data.values():
                    if isinstance(value, str):
                        total += len(value) // 4
                return total
            elif hasattr(data, '__str__'):
                return len(str(data)) // 4
            return 0
        
        input_tokens = count_tokens_approx(input_data)
        output_tokens = count_tokens_approx(output_data)
        return input_tokens + output_tokens


class CrewAIEventBusListener:
    """Listener for CrewAI's internal event bus to capture granular events."""
    
    def __init__(self, wrapper: CrewAIWrapper):
        self.wrapper = wrapper
        self.event_builder = wrapper.event_builder
        self.tracer = wrapper.tracer
        self.registered_handlers = []
        # Track span_ids for START/END event pairs
        self._model_invocation_spans = {}  # Track active model invocations by unique ID
        self._tool_call_spans = {}  # Map tool call_id to span_id
        self._mcp_call_spans = {}  # Track active MCP calls
        self._llm_call_counter = 0  # Counter for unique LLM call IDs
        
    def register_handlers(self):
        """Register event handlers with CrewAI's event bus."""
        logger.info("🚀 Starting CrewAIEventBusListener handler registration...")
        try:
            from crewai.utilities.events.crewai_event_bus import crewai_event_bus
            logger.info("✓ CrewAI event bus imported successfully")
            
            # Import all event types we want to handle
            from crewai.utilities.events.tool_usage_events import (
                ToolUsageStartedEvent,
                ToolUsageFinishedEvent,
                ToolUsageErrorEvent,
                ToolExecutionErrorEvent
            )
            from crewai.utilities.events.task_events import (
                TaskStartedEvent,
                TaskCompletedEvent,
                TaskFailedEvent
            )
            from crewai.utilities.events.agent_events import (
                AgentExecutionErrorEvent
            )
            from crewai.utilities.events.knowledge_events import (
                KnowledgeRetrievalStartedEvent,
                KnowledgeRetrievalCompletedEvent,
                KnowledgeQueryStartedEvent,
                KnowledgeQueryCompletedEvent
            )
            
            # Import LLM events for MODEL_INVOCATION tracking
            llm_events_available = False
            guardrail_events_available = False
            
            try:
                from crewai.utilities.events.llm_events import (
                    LLMCallStartedEvent,
                    LLMCallCompletedEvent,
                    LLMCallFailedEvent
                )
                llm_events_available = True
                logger.info("✓ LLM events ARE available in this CrewAI version")
                
                # Try to import guardrail events separately as they might not exist
                try:
                    from crewai.utilities.events.llm_events import (
                        LLMGuardrailStartedEvent,
                        LLMGuardrailCompletedEvent
                    )
                    guardrail_events_available = True
                    logger.info("✓ LLM Guardrail events are also available")
                except ImportError:
                    logger.debug("LLM Guardrail events not available (optional)")
                    
            except ImportError as e:
                logger.warning(f"✗ LLM events NOT available in this CrewAI version: {e}")
            
            # Import reasoning events
            try:
                from crewai.utilities.events.agent_events import (
                    AgentReasoningStartedEvent,
                    AgentReasoningCompletedEvent,
                    AgentReasoningFailedEvent,
                    AgentExecutionStartedEvent,
                    AgentExecutionCompletedEvent
                )
                reasoning_events_available = True
            except ImportError:
                logger.debug("Reasoning events not available in this CrewAI version")
                reasoning_events_available = False
            
            # Import flow events
            try:
                from crewai.utilities.events.flow_events import (
                    FlowStartedEvent,
                    FlowFinishedEvent,
                    MethodExecutionStartedEvent,
                    MethodExecutionFinishedEvent
                )
                flow_events_available = True
            except ImportError:
                logger.debug("Flow events not available in this CrewAI version")
                flow_events_available = False
            
            # Import crew training/test events
            try:
                from crewai.utilities.events.crew_events import (
                    CrewTrainStartedEvent,
                    CrewTrainCompletedEvent,
                    CrewTestStartedEvent,
                    CrewTestCompletedEvent
                )
                crew_events_available = True
            except ImportError:
                logger.debug("Crew training/test events not available in this CrewAI version")
                crew_events_available = False
            
            # Register tool usage handlers
            @crewai_event_bus.on(ToolUsageStartedEvent)
            def handle_tool_started(source, event: ToolUsageStartedEvent):
                self._handle_tool_started(event)
            
            @crewai_event_bus.on(ToolUsageFinishedEvent)
            def handle_tool_finished(source, event: ToolUsageFinishedEvent):
                self._handle_tool_finished(event)
            
            @crewai_event_bus.on(ToolUsageErrorEvent)
            def handle_tool_error(source, event: ToolUsageErrorEvent):
                self._handle_tool_error(event)
            
            @crewai_event_bus.on(ToolExecutionErrorEvent)
            def handle_tool_execution_error(source, event: ToolExecutionErrorEvent):
                self._handle_tool_execution_error(event)
            
            # Register task handlers
            @crewai_event_bus.on(TaskStartedEvent)
            def handle_task_started(source, event: TaskStartedEvent):
                self._handle_task_started(event)
            
            @crewai_event_bus.on(TaskCompletedEvent)
            def handle_task_completed(source, event: TaskCompletedEvent):
                self._handle_task_completed(event)
            
            @crewai_event_bus.on(TaskFailedEvent)
            def handle_task_failed(source, event: TaskFailedEvent):
                self._handle_task_failed(event)
            
            # Register agent error handler
            @crewai_event_bus.on(AgentExecutionErrorEvent)
            def handle_agent_error(source, event: AgentExecutionErrorEvent):
                self._handle_agent_error(event)
            
            # Register knowledge/data access handlers
            @crewai_event_bus.on(KnowledgeRetrievalStartedEvent)
            def handle_knowledge_retrieval_started(source, event: KnowledgeRetrievalStartedEvent):
                self._handle_knowledge_retrieval_started(event)
            
            @crewai_event_bus.on(KnowledgeRetrievalCompletedEvent)
            def handle_knowledge_retrieval_completed(source, event: KnowledgeRetrievalCompletedEvent):
                self._handle_knowledge_retrieval_completed(event)
            
            @crewai_event_bus.on(KnowledgeQueryStartedEvent)
            def handle_knowledge_query_started(source, event: KnowledgeQueryStartedEvent):
                self._handle_knowledge_query_started(event)
            
            @crewai_event_bus.on(KnowledgeQueryCompletedEvent)
            def handle_knowledge_query_completed(source, event: KnowledgeQueryCompletedEvent):
                self._handle_knowledge_query_completed(event)
            
            # Register LLM event handlers if available
            if llm_events_available:
                logger.info("Registering LLM event handlers...")
                
                @crewai_event_bus.on(LLMCallStartedEvent)
                def handle_llm_started(source, event: LLMCallStartedEvent):
                    logger.debug(f"📤 LLM Started event received: model={getattr(event, 'model', '?')}")
                    self._handle_llm_started(event)
                
                @crewai_event_bus.on(LLMCallCompletedEvent)
                def handle_llm_completed(source, event: LLMCallCompletedEvent):
                    logger.debug(f"📥 LLM Completed event received")
                    self._handle_llm_completed(event)
                
                @crewai_event_bus.on(LLMCallFailedEvent)
                def handle_llm_failed(source, event: LLMCallFailedEvent):
                    logger.debug(f"❌ LLM Failed event received")
                    self._handle_llm_failed(event)
                
                logger.info("✓ LLM event handlers registered successfully")
                
                # Register guardrail handlers if available
                if guardrail_events_available:
                    @crewai_event_bus.on(LLMGuardrailStartedEvent)
                    def handle_guardrail_started(source, event: LLMGuardrailStartedEvent):
                        self._handle_guardrail_started(event)
                    
                    @crewai_event_bus.on(LLMGuardrailCompletedEvent)
                    def handle_guardrail_completed(source, event: LLMGuardrailCompletedEvent):
                        self._handle_guardrail_completed(event)
                    
                    logger.info("✓ LLM Guardrail handlers also registered")
            else:
                logger.warning("⚠️  LLM event handlers NOT registered (events not available)")
            
            # Register reasoning event handlers if available
            if reasoning_events_available:
                @crewai_event_bus.on(AgentReasoningStartedEvent)
                def handle_reasoning_started(source, event: AgentReasoningStartedEvent):
                    self._handle_reasoning_started(event)
                
                @crewai_event_bus.on(AgentReasoningCompletedEvent)
                def handle_reasoning_completed(source, event: AgentReasoningCompletedEvent):
                    self._handle_reasoning_completed(event)
                
                @crewai_event_bus.on(AgentReasoningFailedEvent)
                def handle_reasoning_failed(source, event: AgentReasoningFailedEvent):
                    self._handle_reasoning_failed(event)
                
                @crewai_event_bus.on(AgentExecutionStartedEvent)
                def handle_agent_execution_started(source, event: AgentExecutionStartedEvent):
                    self._handle_agent_execution_started(event)
                
                @crewai_event_bus.on(AgentExecutionCompletedEvent)
                def handle_agent_execution_completed(source, event: AgentExecutionCompletedEvent):
                    self._handle_agent_execution_completed(event)
            
            # Register flow event handlers if available
            if flow_events_available:
                @crewai_event_bus.on(FlowStartedEvent)
                def handle_flow_started(source, event: FlowStartedEvent):
                    self._handle_flow_started(event)
                
                @crewai_event_bus.on(FlowFinishedEvent)
                def handle_flow_finished(source, event: FlowFinishedEvent):
                    self._handle_flow_finished(event)
            
            # Register crew training/test handlers if available
            if crew_events_available:
                @crewai_event_bus.on(CrewTrainStartedEvent)
                def handle_train_started(source, event: CrewTrainStartedEvent):
                    self._handle_train_started(event)
                
                @crewai_event_bus.on(CrewTrainCompletedEvent)
                def handle_train_completed(source, event: CrewTrainCompletedEvent):
                    self._handle_train_completed(event)
                
                @crewai_event_bus.on(CrewTestStartedEvent)
                def handle_test_started(source, event: CrewTestStartedEvent):
                    self._handle_test_started(event)
                
                @crewai_event_bus.on(CrewTestCompletedEvent)
                def handle_test_completed(source, event: CrewTestCompletedEvent):
                    self._handle_test_completed(event)
            
            # Store handler references for cleanup
            self.registered_handlers = [
                handle_tool_started,
                handle_tool_finished,
                handle_tool_error,
                handle_tool_execution_error,
                handle_task_started,
                handle_task_completed,
                handle_task_failed,
                handle_agent_error,
                handle_knowledge_retrieval_started,
                handle_knowledge_retrieval_completed,
                handle_knowledge_query_started,
                handle_knowledge_query_completed
            ]
            
            # Add LLM handlers if available
            if llm_events_available:
                self.registered_handlers.extend([
                    handle_llm_started,
                    handle_llm_completed,
                    handle_llm_failed
                ])
                # Add guardrail handlers only if available
                if guardrail_events_available:
                    self.registered_handlers.extend([
                        handle_guardrail_started,
                        handle_guardrail_completed
                    ])
            
            # Add reasoning handlers if available
            if reasoning_events_available:
                self.registered_handlers.extend([
                    handle_reasoning_started,
                    handle_reasoning_completed,
                    handle_reasoning_failed,
                    handle_agent_execution_started,
                    handle_agent_execution_completed
                ])
            
            # Add flow handlers if available
            if flow_events_available:
                self.registered_handlers.extend([
                    handle_flow_started,
                    handle_flow_finished
                ])
            
            # Add crew handlers if available
            if crew_events_available:
                self.registered_handlers.extend([
                    handle_train_started,
                    handle_train_completed,
                    handle_test_started,
                    handle_test_completed
                ])
            
            # Add debug logging for all events if debug mode
            if logger.isEnabledFor(logging.DEBUG):
                try:
                    # Register a catch-all debug handler for any event
                    @crewai_event_bus.on(None)  # None means catch all events
                    def debug_all_events(source, event):
                        logger.debug(f"CrewAI Event Emitted: {type(event).__name__} - {event}")
                    
                    self.registered_handlers.append(debug_all_events)
                except:
                    # If catch-all doesn't work, just log what we registered
                    logger.debug(f"Registered handlers for: {[h.__name__ for h in self.registered_handlers]}")
            
            logger.info("Successfully registered CrewAI event bus handlers")
            return True
            
        except ImportError as e:
            logger.warning(f"Could not import CrewAI event bus or events: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to register event bus handlers: {e}")
            return False
    
    def unregister_handlers(self):
        """Unregister all event handlers from CrewAI's event bus."""
        # Note: CrewAI's event bus doesn't provide a clean unregister mechanism,
        # but we can clear our references to allow garbage collection
        self.registered_handlers.clear()
        logger.info("Cleared CrewAI event bus handler references")
    
    def _is_mcp_tool(self, event) -> bool:
        """Detect if a tool is MCP-based."""
        tool_name = getattr(event, 'tool_name', '')
        tool_class = str(getattr(event, 'tool_class', ''))
        
        return any([
            'mcp' in tool_name.lower(),
            'MCP' in tool_class,
            'MCPServerAdapter' in tool_class
        ])
    
    def _extract_agent_context(self, event) -> tuple:
        """Extract agent ID and name from various event types."""
        agent_id = None
        agent_name = None
        
        # Try to get from event attributes
        if hasattr(event, 'agent_id'):
            agent_id = str(event.agent_id) if event.agent_id else None
        if hasattr(event, 'agent_role'):
            agent_name = event.agent_role
        elif hasattr(event, 'agent') and event.agent:
            # Try to extract from agent object
            agent_id = str(getattr(event.agent, 'id', None)) if hasattr(event.agent, 'id') else None
            agent_name = getattr(event.agent, 'role', None)
        
        return agent_id, agent_name
    
    def _serialize_tool_args(self, tool_args) -> str:
        """Safely serialize tool arguments to string."""
        if isinstance(tool_args, str):
            return tool_args
        elif isinstance(tool_args, dict):
            try:
                return json.dumps(tool_args)
            except:
                return str(tool_args)
        else:
            return str(tool_args)
    
    # Tool event handlers
    def _handle_tool_started(self, event):
        """Handle tool usage started event."""
        try:
            agent_id, agent_name = self._extract_agent_context(event)
            
            if self._is_mcp_tool(event):
                # Emit MCP_CALL_START
                # Convert tool_args to dict format for MCP request
                tool_args = getattr(event, 'tool_args', {})
                if isinstance(tool_args, str):
                    request_data = {"input": tool_args}
                elif isinstance(tool_args, dict):
                    request_data = tool_args
                else:
                    request_data = {"input": str(tool_args)}
                
                mcp_event = self.event_builder.create_mcp_call_start(
                    server_name=event.tool_name,
                    server_url="mcp://local",  # Default URL for local MCP tools
                    operation="tool_execution",
                    method="execute",
                    request=request_data,
                    protocol_version="1.0",
                    agent_id=agent_id,
                    agent_name=agent_name
                )
                # Store span_id for END event
                self._mcp_call_spans[event.tool_name] = mcp_event.span_id
                self.wrapper._send_event_sync(mcp_event)
            else:
                # Emit TOOL_CALL_START
                # Convert tool_args to dict format for arguments parameter
                tool_args = getattr(event, 'tool_args', {})
                if isinstance(tool_args, str):
                    arguments = {"input": tool_args}
                elif isinstance(tool_args, dict):
                    arguments = tool_args
                else:
                    arguments = {"input": str(tool_args)}
                
                tool_event = self.event_builder.create_tool_call_start(
                    tool_name=event.tool_name,
                    arguments=arguments,
                    call_id=str(getattr(event, 'id', None)) if hasattr(event, 'id') else None,
                    agent_id=agent_id,
                    agent_name=agent_name
                )
                # Store span_id for END event
                call_id = str(getattr(event, 'id', None)) if hasattr(event, 'id') else event.tool_name
                self._tool_call_spans[call_id] = tool_event.span_id
                self.wrapper._send_event_sync(tool_event)
        except Exception as e:
            logger.error(f"Error handling tool started event: {e}")
    
    def _handle_tool_finished(self, event):
        """Handle tool usage finished event."""
        try:
            agent_id, agent_name = self._extract_agent_context(event)
            
            # Calculate duration if timestamps available
            duration_ms = None
            if hasattr(event, 'started_at') and hasattr(event, 'finished_at'):
                duration = event.finished_at - event.started_at
                duration_ms = duration.total_seconds() * 1000
            
            if self._is_mcp_tool(event):
                # Emit MCP_CALL_END
                # Format response data
                output = getattr(event, 'output', None)
                if output is not None:
                    response_data = {"result": str(output)}
                else:
                    response_data = {}
                
                # Retrieve span_id from START event
                span_id = self._mcp_call_spans.pop(event.tool_name, None)
                
                mcp_event = self.event_builder.create_mcp_call_end(
                    server_name=event.tool_name,
                    server_url="mcp://local",
                    operation="tool_execution",
                    method="execute",
                    response=response_data,
                    execution_time_ms=duration_ms,
                    error=None,
                    span_id=span_id,  # Use same span_id as START
                    agent_id=agent_id,
                    agent_name=agent_name
                )
                self.wrapper._send_event_sync(mcp_event)
            else:
                # Emit TOOL_CALL_END
                # Retrieve span_id from START event
                call_id = str(getattr(event, 'id', None)) if hasattr(event, 'id') else event.tool_name
                span_id = self._tool_call_spans.pop(call_id, None)
                
                tool_event = self.event_builder.create_tool_call_end(
                    tool_name=event.tool_name,
                    call_id=str(getattr(event, 'id', None)) if hasattr(event, 'id') else None,
                    output=str(event.output) if hasattr(event, 'output') else None,
                    error=None,
                    execution_time_ms=duration_ms,
                    span_id=span_id,  # Use same span_id as START
                    agent_id=agent_id,
                    agent_name=agent_name
                )
                self.wrapper._send_event_sync(tool_event)
        except Exception as e:
            logger.error(f"Error handling tool finished event: {e}")
    
    def _handle_tool_error(self, event):
        """Handle tool usage error event."""
        try:
            agent_id, agent_name = self._extract_agent_context(event)
            error_msg = str(event.error) if hasattr(event, 'error') else "Tool execution failed"
            
            if self._is_mcp_tool(event):
                # Emit MCP_CALL_END with error
                mcp_event = self.event_builder.create_mcp_call_end(
                    server_name=event.tool_name,
                    server_url="mcp://local",
                    operation="tool_execution",
                    method="execute",
                    response={"error": error_msg},
                    execution_time_ms=None,
                    error=error_msg,
                    agent_id=agent_id,
                    agent_name=agent_name
                )
                self.wrapper._send_event_sync(mcp_event)
            else:
                # Emit TOOL_CALL_END with error
                tool_event = self.event_builder.create_tool_call_end(
                    tool_name=event.tool_name,
                    call_id=str(getattr(event, 'id', None)) if hasattr(event, 'id') else None,
                    output=None,
                    error=error_msg,
                    execution_time_ms=None,
                    agent_id=agent_id,
                    agent_name=agent_name
                )
                self.wrapper._send_event_sync(tool_event)
        except Exception as e:
            logger.error(f"Error handling tool error event: {e}")
    
    def _handle_tool_execution_error(self, event):
        """Handle tool execution error event."""
        try:
            agent_id, agent_name = self._extract_agent_context(event)
            
            # Emit ERROR event
            error_event = self.event_builder.create_error(
                error_message=str(event.error) if hasattr(event, 'error') else "Tool execution error",
                error_code="TOOL_EXECUTION_ERROR",
                recoverable=True,
                agent_id=agent_id,
                agent_name=agent_name
            )
            self.wrapper._send_event_sync(error_event)
        except Exception as e:
            logger.error(f"Error handling tool execution error event: {e}")
    
    # Task event handlers
    def _handle_task_started(self, event):
        """Handle task started event."""
        # We already capture this in agent_start, so just log for debugging
        logger.debug(f"Task started: {getattr(event, 'task', 'unknown')}")
    
    def _handle_task_completed(self, event):
        """Handle task completed event."""
        # We already capture this in agent_end, so just log for debugging
        logger.debug(f"Task completed: {getattr(event, 'task', 'unknown')}")
    
    def _handle_task_failed(self, event):
        """Handle task failed event."""
        try:
            # Emit ERROR event for task failure
            error_event = self.event_builder.create_error(
                error_message=str(event.error) if hasattr(event, 'error') else "Task execution failed",
                error_code="TASK_FAILED",
                recoverable=True
            )
            self.wrapper._send_event_sync(error_event)
        except Exception as e:
            logger.error(f"Error handling task failed event: {e}")
    
    # Agent event handlers
    def _handle_agent_error(self, event):
        """Handle agent execution error event."""
        try:
            agent_id, agent_name = self._extract_agent_context(event)
            
            # Emit ERROR event
            error_event = self.event_builder.create_error(
                error_message=str(event.error) if hasattr(event, 'error') else "Agent execution error",
                error_code="AGENT_EXECUTION_ERROR",
                recoverable=True,
                agent_id=agent_id,
                agent_name=agent_name
            )
            self.wrapper._send_event_sync(error_event)
        except Exception as e:
            logger.error(f"Error handling agent error event: {e}")
    
    # Knowledge/data access event handlers
    def _handle_knowledge_retrieval_started(self, event):
        """Handle knowledge retrieval started event."""
        try:
            agent_id, agent_name = self._extract_agent_context(event)
            
            # Emit DATA_ACCESS event
            data_event = self.event_builder.create_data_access(
                datasource="knowledge_base",
                document_ids=None,
                chunk_ids=None,
                pii_categories=None,
                agent_id=agent_id,
                agent_name=agent_name
            )
            self.wrapper._send_event_sync(data_event)
        except Exception as e:
            logger.error(f"Error handling knowledge retrieval started: {e}")
    
    def _handle_knowledge_retrieval_completed(self, event):
        """Handle knowledge retrieval completed event."""
        try:
            agent_id, agent_name = self._extract_agent_context(event)
            
            # Extract document IDs if available
            doc_ids = None
            if hasattr(event, 'document_ids'):
                doc_ids = list(event.document_ids) if event.document_ids else None
            
            # Emit DATA_ACCESS event
            data_event = self.event_builder.create_data_access(
                datasource="knowledge_base",
                document_ids=doc_ids,
                chunk_ids=None,
                pii_categories=None,
                agent_id=agent_id,
                agent_name=agent_name
            )
            self.wrapper._send_event_sync(data_event)
        except Exception as e:
            logger.error(f"Error handling knowledge retrieval completed: {e}")
    
    def _handle_knowledge_query_started(self, event):
        """Handle knowledge query started event."""
        try:
            agent_id, agent_name = self._extract_agent_context(event)
            
            # Emit DATA_ACCESS event
            data_event = self.event_builder.create_data_access(
                datasource="knowledge_base",
                document_ids=None,
                chunk_ids=None,
                pii_categories=None,
                agent_id=agent_id,
                agent_name=agent_name
            )
            self.wrapper._send_event_sync(data_event)
        except Exception as e:
            logger.error(f"Error handling knowledge query started: {e}")
    
    def _handle_knowledge_query_completed(self, event):
        """Handle knowledge query completed event."""
        try:
            agent_id, agent_name = self._extract_agent_context(event)
            
            # Extract chunk IDs if available
            chunk_ids = None
            if hasattr(event, 'chunk_ids'):
                chunk_ids = list(event.chunk_ids) if event.chunk_ids else None
            
            # Emit DATA_ACCESS event
            data_event = self.event_builder.create_data_access(
                datasource="knowledge_base",
                document_ids=None,
                chunk_ids=chunk_ids,
                pii_categories=None,
                agent_id=agent_id,
                agent_name=agent_name
            )
            self.wrapper._send_event_sync(data_event)
        except Exception as e:
            logger.error(f"Error handling knowledge query completed: {e}")
    
    # LLM event handlers for MODEL_INVOCATION events
    def _handle_llm_started(self, event):
        """Handle LLM call started event."""
        logger.info(f"🎯 _handle_llm_started called with event type: {type(event).__name__}")
        try:
            agent_id, agent_name = self._extract_agent_context(event)
            
            # Extract LLM details
            provider = getattr(event, 'provider', 'unknown')
            model = getattr(event, 'model', 'unknown')
            messages = getattr(event, 'messages', [])
            temperature = getattr(event, 'temperature', None)
            max_tokens = getattr(event, 'max_tokens', None)
            tools = getattr(event, 'tools', None)
            
            logger.info(f"Creating MODEL_INVOCATION_START: provider={provider}, model={model}")
            
            # Emit MODEL_INVOCATION_START
            llm_event = self.event_builder.create_model_invocation_start(
                provider=provider,
                model=model,
                messages=messages if isinstance(messages, list) else [],
                agent_id=agent_id,
                agent_name=agent_name,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools
            )
            
            # Generate unique key for this LLM call to handle concurrent calls
            self._llm_call_counter += 1
            llm_call_id = f"{agent_id}_{model}_{self._llm_call_counter}"
            
            # Store span_id for END event with unique key
            self._model_invocation_spans[llm_call_id] = llm_event.span_id
            
            # Also store the event details to match with END event
            if not hasattr(event, '_chaukas_llm_call_id'):
                event._chaukas_llm_call_id = llm_call_id
            
            logger.debug(f"Stored MODEL span {llm_event.span_id[-8:]} for call_id {llm_call_id}")
            
            self.wrapper._send_event_sync(llm_event)
            logger.info("✓ MODEL_INVOCATION_START event sent successfully")
        except Exception as e:
            logger.error(f"Error handling LLM started event: {e}", exc_info=True)
    
    def _handle_llm_completed(self, event):
        """Handle LLM call completed event."""
        try:
            agent_id, agent_name = self._extract_agent_context(event)
            
            # Extract LLM response details
            provider = getattr(event, 'provider', 'unknown')
            model = getattr(event, 'model', 'unknown')
            response_content = getattr(event, 'response', None)
            tool_calls = getattr(event, 'tool_calls', None)
            finish_reason = getattr(event, 'finish_reason', None)
            prompt_tokens = getattr(event, 'prompt_tokens', None)
            completion_tokens = getattr(event, 'completion_tokens', None)
            total_tokens = getattr(event, 'total_tokens', None)
            duration_ms = None
            
            # Calculate duration if timestamps available
            if hasattr(event, 'started_at') and hasattr(event, 'completed_at'):
                duration = event.completed_at - event.started_at
                duration_ms = duration.total_seconds() * 1000
            
            # Try to retrieve span_id using the stored call_id
            span_id = None
            llm_call_id = getattr(event, '_chaukas_llm_call_id', None)
            
            if llm_call_id and llm_call_id in self._model_invocation_spans:
                # Found by exact call_id match
                span_id = self._model_invocation_spans.pop(llm_call_id)
                logger.debug(f"Retrieved MODEL span {span_id[-8:]} for call_id {llm_call_id}")
            else:
                # Fallback: try to find the most recent matching span
                # This handles cases where the event object doesn't preserve our custom attribute
                for key in list(self._model_invocation_spans.keys()):
                    if key.startswith(f"{agent_id}_{model}_"):
                        span_id = self._model_invocation_spans.pop(key)
                        logger.debug(f"Retrieved MODEL span {span_id[-8:]} using fallback for {agent_id}/{model}")
                        break
                
                if not span_id:
                    logger.warning(f"Could not find matching MODEL span for {agent_id}/{model}")
            
            # Emit MODEL_INVOCATION_END with same span_id as START
            llm_event = self.event_builder.create_model_invocation_end(
                provider=provider,
                model=model,
                response_content=str(response_content) if response_content else None,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                duration_ms=duration_ms,
                span_id=span_id,  # Use same span_id as START
                agent_id=agent_id,
                agent_name=agent_name,
                error=None
            )
            self.wrapper._send_event_sync(llm_event)
        except Exception as e:
            logger.error(f"Error handling LLM completed event: {e}")
    
    def _handle_llm_failed(self, event):
        """Handle LLM call failed event."""
        try:
            agent_id, agent_name = self._extract_agent_context(event)
            
            # Extract error details
            provider = getattr(event, 'provider', 'unknown')
            model = getattr(event, 'model', 'unknown')
            error_msg = str(event.error) if hasattr(event, 'error') else "LLM call failed"
            
            # Emit MODEL_INVOCATION_END with error
            llm_event = self.event_builder.create_model_invocation_end(
                provider=provider,
                model=model,
                response_content=None,
                tool_calls=None,
                finish_reason="error",
                agent_id=agent_id,
                agent_name=agent_name,
                error=error_msg
            )
            self.wrapper._send_event_sync(llm_event)
        except Exception as e:
            logger.error(f"Error handling LLM failed event: {e}")
    
    # Guardrail event handlers for POLICY_DECISION events
    def _handle_guardrail_started(self, event):
        """Handle LLM guardrail started event."""
        try:
            agent_id, agent_name = self._extract_agent_context(event)
            
            # Extract guardrail details
            policy_id = getattr(event, 'guardrail_id', 'unknown')
            rule_ids = getattr(event, 'rules', [])
            
            # Emit POLICY_DECISION event (as started)
            policy_event = self.event_builder.create_policy_decision(
                policy_id=str(policy_id),
                outcome="evaluating",
                rule_ids=[str(r) for r in rule_ids] if rule_ids else [],
                rationale="Guardrail evaluation started",
                agent_id=agent_id,
                agent_name=agent_name
            )
            self.wrapper._send_event_sync(policy_event)
        except Exception as e:
            logger.error(f"Error handling guardrail started event: {e}")
    
    def _handle_guardrail_completed(self, event):
        """Handle LLM guardrail completed event."""
        try:
            agent_id, agent_name = self._extract_agent_context(event)
            
            # Extract guardrail results
            policy_id = getattr(event, 'guardrail_id', 'unknown')
            outcome = getattr(event, 'outcome', 'unknown')
            rule_ids = getattr(event, 'triggered_rules', [])
            rationale = getattr(event, 'reason', None)
            
            # Emit POLICY_DECISION event
            policy_event = self.event_builder.create_policy_decision(
                policy_id=str(policy_id),
                outcome=str(outcome),
                rule_ids=[str(r) for r in rule_ids] if rule_ids else [],
                rationale=rationale,
                agent_id=agent_id,
                agent_name=agent_name
            )
            self.wrapper._send_event_sync(policy_event)
        except Exception as e:
            logger.error(f"Error handling guardrail completed event: {e}")
    
    # Reasoning event handlers for STATE_UPDATE events
    def _handle_reasoning_started(self, event):
        """Handle agent reasoning started event."""
        try:
            agent_id, agent_name = self._extract_agent_context(event)
            
            # Emit STATE_UPDATE event
            state_event = self.event_builder.create_state_update(
                state_data={
                    "reasoning_status": "started",
                    "task": str(getattr(event, 'task', None)),
                    "context": str(getattr(event, 'context', None))[:500] if hasattr(event, 'context') else None
                },
                agent_id=agent_id,
                agent_name=agent_name
            )
            self.wrapper._send_event_sync(state_event)
        except Exception as e:
            logger.error(f"Error handling reasoning started event: {e}")
    
    def _handle_reasoning_completed(self, event):
        """Handle agent reasoning completed event."""
        try:
            agent_id, agent_name = self._extract_agent_context(event)
            
            # Emit STATE_UPDATE event
            state_event = self.event_builder.create_state_update(
                state_data={
                    "reasoning_status": "completed",
                    "result": str(getattr(event, 'result', None))[:500] if hasattr(event, 'result') else None,
                    "decisions": getattr(event, 'decisions', [])
                },
                agent_id=agent_id,
                agent_name=agent_name
            )
            self.wrapper._send_event_sync(state_event)
        except Exception as e:
            logger.error(f"Error handling reasoning completed event: {e}")
    
    def _handle_reasoning_failed(self, event):
        """Handle agent reasoning failed event."""
        try:
            agent_id, agent_name = self._extract_agent_context(event)
            
            # Emit ERROR event
            error_event = self.event_builder.create_error(
                error_message=str(event.error) if hasattr(event, 'error') else "Agent reasoning failed",
                error_code="REASONING_FAILED",
                recoverable=True,
                agent_id=agent_id,
                agent_name=agent_name
            )
            self.wrapper._send_event_sync(error_event)
        except Exception as e:
            logger.error(f"Error handling reasoning failed event: {e}")
    
    # Enhanced agent execution handlers with handoff detection
    def _handle_agent_execution_started(self, event):
        """Handle agent execution started event - track for handoffs."""
        try:
            agent_id, agent_name = self._extract_agent_context(event)
            
            # Check if this is a handoff (different agent than last one)
            if hasattr(self.wrapper, '_last_active_agent') and self.wrapper._last_active_agent:
                last_agent_id, last_agent_name = self.wrapper._last_active_agent
                if agent_id != last_agent_id:
                    # Emit AGENT_HANDOFF event
                    handoff_event = self.event_builder.create_agent_handoff(
                        from_agent_id=last_agent_id,
                        from_agent_name=last_agent_name,
                        to_agent_id=agent_id,
                        to_agent_name=agent_name,
                        reason="Task delegation",
                        handoff_type="sequential",
                        handoff_data={
                            "task": str(getattr(event, 'task', None)),
                            "context": str(getattr(event, 'context', None))[:500] if hasattr(event, 'context') else None
                        }
                    )
                    self.wrapper._send_event_sync(handoff_event)
            
            # Update last active agent
            self.wrapper._last_active_agent = (agent_id, agent_name)
            
        except Exception as e:
            logger.error(f"Error handling agent execution started: {e}")
    
    def _handle_agent_execution_completed(self, event):
        """Handle agent execution completed event."""
        # Already handled in main agent end event
        pass
    
    # Flow event handlers for SYSTEM events
    def _handle_flow_started(self, event):
        """Handle flow started event."""
        try:
            # Emit SYSTEM event
            system_event = self.event_builder.create_system_event(
                message=f"Flow started: {getattr(event, 'flow_name', 'unknown')}",
                metadata={
                    "flow_name": getattr(event, 'flow_name', None),
                    "flow_id": getattr(event, 'flow_id', None),
                    "framework": "crewai"
                }
            )
            self.wrapper._send_event_sync(system_event)
        except Exception as e:
            logger.error(f"Error handling flow started event: {e}")
    
    def _handle_flow_finished(self, event):
        """Handle flow finished event."""
        try:
            # Emit SYSTEM event
            system_event = self.event_builder.create_system_event(
                message=f"Flow finished: {getattr(event, 'flow_name', 'unknown')}",
                metadata={
                    "flow_name": getattr(event, 'flow_name', None),
                    "flow_id": getattr(event, 'flow_id', None),
                    "success": getattr(event, 'success', True),
                    "framework": "crewai"
                }
            )
            self.wrapper._send_event_sync(system_event)
        except Exception as e:
            logger.error(f"Error handling flow finished event: {e}")
    
    # Crew training/test event handlers
    def _handle_train_started(self, event):
        """Handle crew training started event."""
        try:
            # Emit SYSTEM event
            system_event = self.event_builder.create_system_event(
                message="Crew training started",
                metadata={
                    "crew_name": getattr(event, 'crew_name', None),
                    "training_data_size": getattr(event, 'data_size', None),
                    "framework": "crewai"
                }
            )
            self.wrapper._send_event_sync(system_event)
        except Exception as e:
            logger.error(f"Error handling training started event: {e}")
    
    def _handle_train_completed(self, event):
        """Handle crew training completed event."""
        try:
            # Emit SYSTEM event
            system_event = self.event_builder.create_system_event(
                message="Crew training completed",
                metadata={
                    "crew_name": getattr(event, 'crew_name', None),
                    "metrics": getattr(event, 'metrics', {}),
                    "framework": "crewai"
                }
            )
            self.wrapper._send_event_sync(system_event)
        except Exception as e:
            logger.error(f"Error handling training completed event: {e}")
    
    def _handle_test_started(self, event):
        """Handle crew test started event."""
        try:
            # Emit SYSTEM event
            system_event = self.event_builder.create_system_event(
                message="Crew test started",
                metadata={
                    "crew_name": getattr(event, 'crew_name', None),
                    "test_cases": getattr(event, 'test_cases', None),
                    "framework": "crewai"
                }
            )
            self.wrapper._send_event_sync(system_event)
        except Exception as e:
            logger.error(f"Error handling test started event: {e}")
    
    def _handle_test_completed(self, event):
        """Handle crew test completed event."""
        try:
            # Emit SYSTEM event
            system_event = self.event_builder.create_system_event(
                message="Crew test completed",
                metadata={
                    "crew_name": getattr(event, 'crew_name', None),
                    "test_results": getattr(event, 'results', {}),
                    "framework": "crewai"
                }
            )
            self.wrapper._send_event_sync(system_event)
        except Exception as e:
            logger.error(f"Error handling test completed event: {e}")