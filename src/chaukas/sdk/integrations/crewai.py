"""
CrewAI integration for Chaukas instrumentation.
Uses proto-compliant events with proper agent handoff tracking.
"""

import functools
import logging
from typing import Any, Dict, Optional, List
import json

from chaukas.spec.common.v1.events_pb2 import EventStatus, EventSeverity

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
        self._original_kickoff = None
        self._original_kickoff_async = None
        self._original_kickoff_for_each = None
        self._original_kickoff_for_each_async = None
        self._original_execute_task = None
        self.event_listener = None  # Event bus listener
    
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
            
            def _send_event_sync(self, event):
                """Helper to send event from sync context."""
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # Schedule as task if loop is running
                        asyncio.create_task(self.tracer.client.send_event(event))
                    else:
                        loop.run_until_complete(self.tracer.client.send_event(event))
                except RuntimeError:
                    # No event loop, create one
                    asyncio.run(self.tracer.client.send_event(event))
            
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
            @functools.wraps(self._original_kickoff)
            def instrumented_kickoff(crew_instance, *args, **kwargs):
                """Instrumented sync version of Crew.kickoff()."""
                crew_data = _extract_crew_data(crew_instance, args, kwargs)
                
                with self.tracer.start_span("crewai.crew.kickoff") as span:
                    span.set_attribute("method", "kickoff")
                    span.set_attribute("crew_name", crew_data['crew_name'])
                    span.set_attribute("agents_count", len(crew_data['agents']))
                    
                    try:
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
                        self._send_event_sync(session_start)
                        
                        # Send input received event if inputs provided
                        if crew_data['inputs']:
                            input_event = self.event_builder.create_input_received(
                                content=str(crew_data['inputs'])[:1000],
                                metadata={"input_type": "crew_inputs", "method": "kickoff"}
                            )
                            self._send_event_sync(input_event)
                        
                        # Execute original method
                        result = self._original_kickoff(crew_instance, *args, **kwargs)
                        
                        # Send output event
                        output_event = self.event_builder.create_output_emitted(
                            content=str(result)[:1000] if result else "No result",
                            metadata={"output_type": "crew_result", "method": "kickoff"}
                        )
                        self._send_event_sync(output_event)
                        
                        # Send session end event
                        session_end = self.event_builder.create_session_end(
                            metadata={
                                "crew_name": crew_data['crew_name'],
                                "method": "kickoff",
                                "success": True,
                                "result_type": type(result).__name__
                            }
                        )
                        self._send_event_sync(session_end)
                        
                        return result
                        
                    except Exception as e:
                        # Send error event
                        error_event = self.event_builder.create_error(
                            error_message=str(e),
                            error_code=type(e).__name__,
                            recoverable=True,
                        )
                        self._send_event_sync(error_event)
                        raise
            
            # Create async kickoff wrapper  
            @functools.wraps(self._original_kickoff_async or self._original_kickoff)
            async def instrumented_kickoff_async(crew_instance, *args, **kwargs):
                """Instrumented async version of Crew.kickoff_async()."""
                crew_data = _extract_crew_data(crew_instance, args, kwargs)
                
                with self.tracer.start_span("crewai.crew.kickoff_async") as span:
                    span.set_attribute("method", "kickoff_async")
                    span.set_attribute("crew_name", crew_data['crew_name'])
                    span.set_attribute("agents_count", len(crew_data['agents']))
                    
                    try:
                        # Send session start event
                        session_start = self.event_builder.create_session_start(
                            metadata={
                                "crew_name": crew_data['crew_name'],
                                "method": "kickoff_async",
                                "framework": "crewai",
                                "agents_count": len(crew_data['agents']),
                                "tasks_count": len(crew_data['tasks']),
                            }
                        )
                        await self.tracer.client.send_event(session_start)
                        
                        # Send input received event if inputs provided
                        if crew_data['inputs']:
                            input_event = self.event_builder.create_input_received(
                                content=str(crew_data['inputs'])[:1000],
                                metadata={"input_type": "crew_inputs", "method": "kickoff_async"}
                            )
                            await self.tracer.client.send_event(input_event)
                        
                        # Execute original method
                        if self._original_kickoff_async:
                            result = await self._original_kickoff_async(crew_instance, *args, **kwargs)
                        else:
                            result = self._original_kickoff(crew_instance, *args, **kwargs)
                        
                        # Send output event
                        output_event = self.event_builder.create_output_emitted(
                            content=str(result)[:1000] if result else "No result",
                            metadata={"output_type": "crew_result", "method": "kickoff_async"}
                        )
                        await self.tracer.client.send_event(output_event)
                        
                        # Send session end event
                        session_end = self.event_builder.create_session_end(
                            metadata={
                                "crew_name": crew_data['crew_name'],
                                "method": "kickoff_async",
                                "success": True,
                                "result_type": type(result).__name__
                            }
                        )
                        await self.tracer.client.send_event(session_end)
                        
                        return result
                        
                    except Exception as e:
                        # Send error event
                        error_event = self.event_builder.create_error(
                            error_message=str(e),
                            error_code=type(e).__name__,
                            recoverable=True,
                        )
                        await self.tracer.client.send_event(error_event)
                        raise
            
            # Patch all methods
            Crew.kickoff = instrumented_kickoff
            if self._original_kickoff_async:
                Crew.kickoff_async = instrumented_kickoff_async
            
            # TODO: Add kickoff_for_each and kickoff_for_each_async wrappers
            
            # Initialize and register event bus listener
            if not self.event_listener:
                self.event_listener = CrewAIEventBusListener(self)
                self.event_listener.register_handlers()
            
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
                
                with self.tracer.start_span("crewai.agent.execute_task") as span:
                    span.set_attribute("agent_type", "crewai")
                    span.set_attribute("agent_role", getattr(agent_instance, "role", "unknown"))
                    
                    try:
                        # Extract task data
                        task = args[0] if args else None
                        
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
                        
                        # Execute original method
                        result = self._original_execute_task(agent_instance, *args, **kwargs)
                        
                        # Send agent end event
                        end_event = self.event_builder.create_agent_end(
                            agent_id=agent_id,
                            agent_name=agent_name,
                            status=EventStatus.EVENT_STATUS_COMPLETED,
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


class CrewAIEventBusListener:
    """Listener for CrewAI's internal event bus to capture granular events."""
    
    def __init__(self, wrapper: CrewAIWrapper):
        self.wrapper = wrapper
        self.event_builder = wrapper.event_builder
        self.tracer = wrapper.tracer
        self.registered_handlers = []
        
    def register_handlers(self):
        """Register event handlers with CrewAI's event bus."""
        try:
            from crewai.utilities.events.crewai_event_bus import crewai_event_bus
            
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
                mcp_event = self.event_builder.create_mcp_call_start(
                    server_name=event.tool_name,
                    method_name="execute",
                    arguments=self._serialize_tool_args(event.tool_args),
                    agent_id=agent_id,
                    agent_name=agent_name,
                    metadata={
                        "tool_class": str(event.tool_class) if event.tool_class else None,
                        "run_attempts": event.run_attempts,
                        "task_name": event.task_name,
                        "framework": "crewai"
                    }
                )
                self.wrapper._send_event_sync(mcp_event)
            else:
                # Emit TOOL_CALL_START
                tool_event = self.event_builder.create_tool_call_start(
                    tool_name=event.tool_name,
                    tool_args=self._serialize_tool_args(event.tool_args),
                    agent_id=agent_id,
                    agent_name=agent_name,
                    metadata={
                        "tool_class": str(event.tool_class) if event.tool_class else None,
                        "run_attempts": event.run_attempts,
                        "task_name": event.task_name,
                        "framework": "crewai"
                    }
                )
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
                mcp_event = self.event_builder.create_mcp_call_end(
                    server_name=event.tool_name,
                    method_name="execute",
                    response=str(event.output) if hasattr(event, 'output') else None,
                    success=True,
                    agent_id=agent_id,
                    agent_name=agent_name,
                    metadata={
                        "from_cache": getattr(event, 'from_cache', False),
                        "duration_ms": duration_ms,
                        "framework": "crewai"
                    }
                )
                self.wrapper._send_event_sync(mcp_event)
            else:
                # Emit TOOL_CALL_END
                tool_event = self.event_builder.create_tool_call_end(
                    tool_name=event.tool_name,
                    tool_output=str(event.output) if hasattr(event, 'output') else None,
                    success=True,
                    agent_id=agent_id,
                    agent_name=agent_name,
                    metadata={
                        "from_cache": getattr(event, 'from_cache', False),
                        "duration_ms": duration_ms,
                        "framework": "crewai"
                    }
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
                    method_name="execute",
                    success=False,
                    error_message=error_msg,
                    agent_id=agent_id,
                    agent_name=agent_name,
                    metadata={"framework": "crewai"}
                )
                self.wrapper._send_event_sync(mcp_event)
            else:
                # Emit TOOL_CALL_END with error
                tool_event = self.event_builder.create_tool_call_end(
                    tool_name=event.tool_name,
                    success=False,
                    error_message=error_msg,
                    agent_id=agent_id,
                    agent_name=agent_name,
                    metadata={"framework": "crewai"}
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
                agent_name=agent_name,
                metadata={
                    "tool_name": event.tool_name,
                    "tool_args": str(event.tool_args) if hasattr(event, 'tool_args') else None,
                    "framework": "crewai"
                }
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
                recoverable=True,
                metadata={
                    "task": str(getattr(event, 'task', None)),
                    "framework": "crewai"
                }
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
                agent_name=agent_name,
                metadata={
                    "task": str(getattr(event, 'task', None)),
                    "framework": "crewai"
                }
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
                data_source="knowledge_base",
                operation="retrieval_start",
                agent_id=agent_id,
                agent_name=agent_name,
                metadata={
                    "query": str(getattr(event, 'query', None)),
                    "framework": "crewai"
                }
            )
            self.wrapper._send_event_sync(data_event)
        except Exception as e:
            logger.error(f"Error handling knowledge retrieval started: {e}")
    
    def _handle_knowledge_retrieval_completed(self, event):
        """Handle knowledge retrieval completed event."""
        try:
            agent_id, agent_name = self._extract_agent_context(event)
            
            # Emit DATA_ACCESS event
            data_event = self.event_builder.create_data_access(
                data_source="knowledge_base",
                operation="retrieval_complete",
                agent_id=agent_id,
                agent_name=agent_name,
                metadata={
                    "results_count": getattr(event, 'results_count', None),
                    "framework": "crewai"
                }
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
                data_source="knowledge_base",
                operation="query_start",
                agent_id=agent_id,
                agent_name=agent_name,
                metadata={
                    "query": str(getattr(event, 'query', None)),
                    "framework": "crewai"
                }
            )
            self.wrapper._send_event_sync(data_event)
        except Exception as e:
            logger.error(f"Error handling knowledge query started: {e}")
    
    def _handle_knowledge_query_completed(self, event):
        """Handle knowledge query completed event."""
        try:
            agent_id, agent_name = self._extract_agent_context(event)
            
            # Emit DATA_ACCESS event
            data_event = self.event_builder.create_data_access(
                data_source="knowledge_base",
                operation="query_complete",
                agent_id=agent_id,
                agent_name=agent_name,
                metadata={
                    "success": getattr(event, 'success', True),
                    "framework": "crewai"
                }
            )
            self.wrapper._send_event_sync(data_event)
        except Exception as e:
            logger.error(f"Error handling knowledge query completed: {e}")