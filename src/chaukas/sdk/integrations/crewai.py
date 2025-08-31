"""
CrewAI integration for Chaukas instrumentation.
Uses proto-compliant events with proper agent handoff tracking.
"""

import functools
import logging
from typing import Any, Dict, Optional, List

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
        self._original_kickoff = None
        self._original_kickoff_async = None
        self._original_kickoff_for_each = None
        self._original_kickoff_for_each_async = None
        self._original_execute_task = None
    
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