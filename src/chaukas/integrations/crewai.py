"""
CrewAI integration for Chaukas instrumentation.
Uses proto-compliant events with proper agent handoff tracking.
"""

import asyncio
import logging
from typing import Any, Dict, Optional, List
from functools import wraps

from chaukas.spec.common.v1.events_pb2 import EventStatus

from ..core.tracer import ChaukasTracer
from ..core.event_builder import EventBuilder
from ..core.agent_mapper import AgentMapper

logger = logging.getLogger(__name__)


class CrewAIWrapper:
    """Wrapper for CrewAI instrumentation using proto events."""
    
    def __init__(self, tracer: ChaukasTracer):
        self.tracer = tracer
        self.event_builder = EventBuilder()
        self._current_agents: List[Any] = []  # Track agents for handoff events
    
    def wrap_crew_kickoff(self, wrapped, instance, args, kwargs):
        """Wrap Crew.kickoff method."""
        
        @wraps(wrapped)
        async def async_wrapper(*args, **kwargs):
            with self.tracer.start_span("crewai.crew.kickoff") as span:
                span.set_attribute("crew_type", "crewai")
                
                try:
                    # Extract crew configuration
                    agents = getattr(instance, "agents", [])
                    tasks = getattr(instance, "tasks", [])
                    crew_name = getattr(instance, "name", "crew")
                    process = getattr(instance, "process", "unknown")
                    
                    # Store agents for potential handoff tracking
                    self._current_agents = agents
                    
                    # Send SESSION_START event for the crew execution
                    session_start = self.event_builder.create_session_start(
                        metadata={
                            "crew_name": crew_name,
                            "agents_count": len(agents),
                            "tasks_count": len(tasks),
                            "process_type": str(process),
                            "framework": "crewai",
                            "agents": [
                                {
                                    "agent_id": str(getattr(agent, "id", agent.role)),
                                    "role": getattr(agent, "role", "unknown"),
                                    "goal": getattr(agent, "goal", None),
                                }
                                for agent in agents
                            ],
                            "tasks": [
                                {
                                    "task_id": str(id(task)),
                                    "description": getattr(task, "description", "unknown")[:200],
                                    "assigned_agent": getattr(task.agent, "role", "unknown") if hasattr(task, "agent") else "unknown",
                                }
                                for task in tasks
                            ],
                        }
                    )
                    
                    await self.tracer.client.send_event(session_start)
                    
                    # Execute original method
                    result = await wrapped(*args, **kwargs)
                    
                    # Send SESSION_END event
                    session_end = self.event_builder.create_session_end(
                        metadata={
                            "crew_name": crew_name,
                            "result": str(result)[:500] if result else None,
                            "result_type": type(result).__name__,
                            "framework": "crewai",
                        }
                    )
                    
                    await self.tracer.client.send_event(session_end)
                    
                    return result
                    
                except Exception as e:
                    # Send ERROR event
                    error_event = self.event_builder.create_error(
                        error_message=str(e),
                        error_code=type(e).__name__,
                        recoverable=True,
                    )
                    
                    await self.tracer.client.send_event(error_event)
                    
                    raise
        
        @wraps(wrapped)
        def sync_wrapper(*args, **kwargs):
            # For sync methods, we need to handle differently
            try:
                # Try to run in existing event loop
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Create task for async execution
                    task = asyncio.create_task(async_wrapper(*args, **kwargs))
                    return wrapped(*args, **kwargs)  # Run original sync version
                else:
                    # Run in new event loop
                    return loop.run_until_complete(async_wrapper(*args, **kwargs))
            except RuntimeError:
                # No event loop, run sync version
                return wrapped(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(wrapped):
            return async_wrapper
        else:
            return sync_wrapper
    
    def wrap_agent_execute_task(self, wrapped, instance, args, kwargs):
        """Wrap Agent.execute_task method with proto AGENT events."""
        
        @wraps(wrapped)
        async def async_wrapper(*args, **kwargs):
            # Extract agent info using mapper
            agent_id, agent_name = AgentMapper.map_crewai_agent(instance)
            
            with self.tracer.start_span("crewai.agent.execute_task") as span:
                span.set_attribute("agent_type", "crewai")
                span.set_attribute("agent_role", getattr(instance, "role", "unknown"))
                
                try:
                    # Extract task data
                    task = args[0] if args else None
                    
                    # Send AGENT_START event
                    start_event = self.event_builder.create_agent_start(
                        agent_id=agent_id,
                        agent_name=agent_name,
                        role=getattr(instance, "role", "unknown"),
                        instructions=getattr(instance, "goal", None),
                        tools=[str(tool) for tool in getattr(instance, "tools", [])],
                        metadata={
                            "framework": "crewai",
                            "backstory": getattr(instance, "backstory", None)[:200] if getattr(instance, "backstory", None) else None,
                            "task": {
                                "task_id": str(id(task)) if task else None,
                                "description": getattr(task, "description", None)[:200] if task else None,
                                "expected_output": getattr(task, "expected_output", None)[:200] if task and getattr(task, "expected_output", None) else None,
                            } if task else None
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
                            "framework": "crewai",
                            "result": str(result)[:500] if result else None,
                            "result_type": type(result).__name__,
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
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    return wrapped(*args, **kwargs)
                else:
                    return loop.run_until_complete(async_wrapper(*args, **kwargs))
            except RuntimeError:
                return wrapped(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(wrapped):
            return async_wrapper
        else:
            return sync_wrapper