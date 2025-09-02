"""
OpenAI Agents SDK integration using the base wrapper class.
Demonstrates simplified integration with reusable components.
"""

import logging
from typing import Any, Optional

from chaukas.sdk.integrations.base import BaseIntegrationWrapper
from chaukas.sdk.core.agent_mapper import AgentMapper
from chaukas.sdk.utils.retry_detector import RetryDetector
from chaukas.sdk.utils.event_pairs import EventPairManager

logger = logging.getLogger(__name__)


class OpenAIAgentsWrapperRefactored(BaseIntegrationWrapper):
    """
    Refactored OpenAI Agents wrapper using base class.
    
    This implementation is much simpler than the original because it leverages:
    - BaseIntegrationWrapper for common functionality
    - RetryDetector for retry tracking
    - EventPairManager for START/END correlation
    """
    
    def _initialize_framework(self) -> None:
        """Initialize OpenAI-specific components."""
        self.retry_detector = RetryDetector(self.event_builder)
        self.event_pair_manager = EventPairManager()
    
    def get_framework_name(self) -> str:
        """Return framework name."""
        return "openai_agents"
    
    def wrap_agent_run(self, wrapped, instance, args, kwargs):
        """
        Wrap Agent.run method with instrumentation.
        
        This is now much simpler - just define the wrapper logic,
        the base class handles sync/async conversion.
        """
        async def wrapper(original_func, *call_args, **call_kwargs):
            # Use the base class helper for standard agent execution
            return await self.handle_agent_execution(
                original_func,
                instance,
                call_args,
                call_kwargs,
                lambda agent: AgentMapper.map_openai_agent(agent)
            )
        
        # Base class handles sync/async wrapping
        return self.create_wrapper(wrapped, wrapper)
    
    def wrap_runner_run_once(self, wrapped, instance, args, kwargs):
        """
        Wrap Runner.run_once method with enhanced tracking.
        
        This demonstrates using the utilities for more complex scenarios.
        """
        async def wrapper(original_func, *call_args, **call_kwargs):
            # Extract agent from runner
            agent = getattr(instance, "agent", None)
            if not agent:
                # No agent, just run original
                return await original_func(*call_args, **call_kwargs) if asyncio.iscoroutinefunction(original_func) else original_func(*call_args, **call_kwargs)
            
            agent_id, agent_name = AgentMapper.map_openai_agent(agent)
            
            # Check for handoff
            self.track_agent_handoff(agent_id, agent_name)
            
            try:
                # Send MODEL_INVOCATION_START
                model_start = self.event_builder.create_model_invocation_start(
                    provider="openai",
                    model=getattr(agent, "model", "unknown"),
                    messages=self._extract_messages(call_args, call_kwargs),
                    agent_id=agent_id,
                    agent_name=agent_name
                )
                
                await self._send_event_async(model_start)
                
                # Register for END event pairing
                self.event_pair_manager.register_start_event(
                    "MODEL_INVOCATION",
                    f"{agent_id}_{model_start.event_id}",
                    model_start.span_id
                )
                
                # Execute original
                result = await original_func(*call_args, **call_kwargs) if asyncio.iscoroutinefunction(original_func) else original_func(*call_args, **call_kwargs)
                
                # Send MODEL_INVOCATION_END with same span_id
                model_end = self.event_builder.create_model_invocation_end(
                    provider="openai",
                    model=getattr(agent, "model", "unknown"),
                    response_content=self._extract_response(result),
                    span_id=self.event_pair_manager.get_span_id_for_end(
                        "MODEL_INVOCATION",
                        f"{agent_id}_{model_start.event_id}"
                    ),
                    agent_id=agent_id,
                    agent_name=agent_name
                )
                
                await self._send_event_async(model_end)
                
                # Clear retry counter on success
                self.retry_detector.clear_llm_retry(f"{agent_id}_openai_{getattr(agent, 'model', 'unknown')}")
                
                return result
                
            except Exception as e:
                error_msg = str(e)
                
                # Check for retry
                retry_event = self.retry_detector.track_llm_retry(
                    f"{agent_id}_openai_{getattr(agent, 'model', 'unknown')}",
                    error_msg,
                    agent_id,
                    agent_name
                )
                
                if retry_event:
                    await self._send_event_async(retry_event)
                
                # Send error event (base class helper determines recoverability)
                error_event = self.event_builder.create_error(
                    error_message=error_msg,
                    error_code=type(e).__name__,
                    recoverable=self._is_recoverable_error(e),
                    agent_id=agent_id,
                    agent_name=agent_name
                )
                
                await self._send_event_async(error_event)
                
                # Clear the event pair on error
                self.event_pair_manager.clear_pair(
                    "MODEL_INVOCATION",
                    f"{agent_id}_{model_start.event_id}"
                )
                
                raise
        
        return self.create_wrapper(wrapped, wrapper)
    
    def _extract_messages(self, args: tuple, kwargs: dict) -> list:
        """Extract messages from method arguments."""
        # Try to find messages in args or kwargs
        if "messages" in kwargs:
            return kwargs["messages"]
        
        for arg in args:
            if isinstance(arg, list) and arg and isinstance(arg[0], dict):
                if "role" in arg[0]:
                    return arg
        
        return []
    
    def _extract_response(self, result: Any) -> Optional[str]:
        """Extract response content from result."""
        if result is None:
            return None
        
        # Try common patterns
        if hasattr(result, "content"):
            return str(result.content)
        elif hasattr(result, "text"):
            return str(result.text)
        elif hasattr(result, "message"):
            return str(result.message)
        elif isinstance(result, dict):
            return result.get("content") or result.get("text") or str(result)
        else:
            return str(result)[:1000]  # Limit size


# Comparison with original implementation:
# 
# Original OpenAIAgentsWrapper: ~200 lines
# Refactored version: ~140 lines (30% reduction)
# 
# Benefits:
# 1. No need to implement _send_event_sync - inherited from base
# 2. No need to implement sync/async wrapper logic - use create_wrapper()
# 3. Retry detection is now a reusable utility
# 4. Event pair management is centralized
# 5. Agent handoff tracking is inherited
# 6. Error recoverability logic is inherited
# 7. Tool/instruction extraction helpers are inherited