"""
CrewAI integration for Chaukas instrumentation.
"""

import asyncio
import logging
from typing import Any, Dict, Optional
from functools import wraps

from ..core.tracer import ChaukasTracer
from ..core.events import EventType

logger = logging.getLogger(__name__)


class CrewAIWrapper:
    """Wrapper for CrewAI instrumentation."""
    
    def __init__(self, tracer: ChaukasTracer):
        self.tracer = tracer
    
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
                    process = getattr(instance, "process", "unknown")
                    
                    crew_data = {
                        "crew_id": str(id(instance)),
                        "agents_count": len(agents),
                        "tasks_count": len(tasks),
                        "process_type": str(process),
                        "agents": [
                            {
                                "agent_id": str(id(agent)),
                                "role": getattr(agent, "role", "unknown"),
                                "goal": getattr(agent, "goal", None),
                                "backstory": getattr(agent, "backstory", None)[:200] if getattr(agent, "backstory", None) else None,
                            }
                            for agent in agents
                        ],
                        "tasks": [
                            {
                                "task_id": str(id(task)),
                                "description": getattr(task, "description", "unknown")[:200],
                                "agent": getattr(task, "agent", {}).get("role", "unknown") if hasattr(task, "agent") else "unknown",
                            }
                            for task in tasks
                        ],
                    }
                    
                    # Send session start event
                    await self.tracer.send_event(
                        event_type=EventType.SESSION_START,
                        source="crewai",
                        data=crew_data,
                        normalize_for="crewai",
                    )
                    
                    # Execute original method
                    result = await wrapped(*args, **kwargs)
                    
                    # Send session end event
                    end_data = crew_data.copy()
                    end_data.update({
                        "result": str(result)[:500] if result else None,
                        "result_type": type(result).__name__,
                    })
                    
                    await self.tracer.send_event(
                        event_type=EventType.SESSION_END,
                        source="crewai",
                        data=end_data,
                        normalize_for="crewai",
                    )
                    
                    return result
                    
                except Exception as e:
                    # Send error event
                    error_data = {
                        "crew_id": str(id(instance)),
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }
                    
                    await self.tracer.send_event(
                        event_type=EventType.AGENT_ERROR,
                        source="crewai",
                        data=error_data,
                        normalize_for="crewai",
                    )
                    
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
        """Wrap Agent.execute_task method."""
        
        @wraps(wrapped)
        async def async_wrapper(*args, **kwargs):
            with self.tracer.start_span("crewai.agent.execute_task") as span:
                span.set_attribute("agent_type", "crewai")
                span.set_attribute("agent_role", getattr(instance, "role", "unknown"))
                
                try:
                    # Extract agent and task data
                    task = args[0] if args else None
                    
                    agent_data = {
                        "agent_id": str(id(instance)),
                        "role": getattr(instance, "role", "unknown"),
                        "goal": getattr(instance, "goal", None),
                        "backstory": getattr(instance, "backstory", None)[:200] if getattr(instance, "backstory", None) else None,
                        "tools": [str(tool) for tool in getattr(instance, "tools", [])],
                    }
                    
                    task_data = {}
                    if task:
                        task_data = {
                            "task_id": str(id(task)),
                            "description": getattr(task, "description", "unknown")[:200],
                            "expected_output": getattr(task, "expected_output", None)[:200] if getattr(task, "expected_output", None) else None,
                        }
                    
                    # Send agent start event
                    start_data = {**agent_data, **task_data}
                    await self.tracer.send_event(
                        event_type=EventType.AGENT_START,
                        source="crewai",
                        data=start_data,
                        normalize_for="crewai",
                    )
                    
                    # Execute original method
                    result = await wrapped(*args, **kwargs)
                    
                    # Send agent end event
                    end_data = start_data.copy()
                    end_data.update({
                        "result": str(result)[:500] if result else None,
                        "result_type": type(result).__name__,
                    })
                    
                    await self.tracer.send_event(
                        event_type=EventType.AGENT_END,
                        source="crewai",
                        data=end_data,
                        normalize_for="crewai",
                    )
                    
                    return result
                    
                except Exception as e:
                    # Send agent error event
                    error_data = {
                        "agent_id": str(id(instance)),
                        "role": getattr(instance, "role", "unknown"),
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }
                    
                    await self.tracer.send_event(
                        event_type=EventType.AGENT_ERROR,
                        source="crewai",
                        data=error_data,
                        normalize_for="crewai",
                    )
                    
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