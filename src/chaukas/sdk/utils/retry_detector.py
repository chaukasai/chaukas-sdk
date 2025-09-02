"""
Retry detection and tracking utilities for all integrations.
"""

import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field

from chaukas.spec.common.v1.events_pb2 import Event
from chaukas.sdk.core.event_builder import EventBuilder

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """Configuration for retry detection and tracking."""
    
    max_llm_retries: int = 3
    max_tool_retries: int = 2
    max_task_retries: int = 3
    
    llm_backoff_strategy: str = "exponential"
    tool_backoff_strategy: str = "linear"
    task_backoff_strategy: str = "exponential"
    
    llm_base_delay_ms: int = 1000
    tool_base_delay_ms: int = 500
    task_base_delay_ms: int = 2000


class RetryDetector:
    """
    Manages retry detection and tracking across all integration types.
    
    Features:
    - Detects retryable errors based on patterns
    - Tracks retry attempts per operation
    - Calculates backoff delays
    - Creates RETRY events
    """
    
    def __init__(self, event_builder: EventBuilder, config: Optional[RetryConfig] = None):
        """
        Initialize retry detector.
        
        Args:
            event_builder: EventBuilder for creating RETRY events
            config: Optional retry configuration
        """
        self.event_builder = event_builder
        self.config = config or RetryConfig()
        
        # Track retry attempts by operation key
        self._llm_retry_attempts: Dict[str, int] = {}
        self._tool_retry_attempts: Dict[str, int] = {}
        self._task_retry_attempts: Dict[str, int] = {}
    
    def is_retryable_error(self, error_msg: str) -> bool:
        """
        Check if an error is retryable based on the error message.
        
        Args:
            error_msg: Error message to check
            
        Returns:
            True if error is retryable, False otherwise
        """
        retryable_patterns = [
            "rate limit",
            "timeout",
            "connection",
            "temporary",
            "503",
            "429",
            "network",
            "unavailable",
            "throttl",
            "busy",
            "overload",
            "transient"
        ]
        
        error_lower = error_msg.lower()
        return any(pattern in error_lower for pattern in retryable_patterns)
    
    def calculate_backoff_delay(self, strategy: str, attempt: int, base_delay_ms: int) -> int:
        """
        Calculate backoff delay based on strategy and attempt number.
        
        Args:
            strategy: Backoff strategy ("exponential", "linear", "immediate")
            attempt: Current attempt number (1-based)
            base_delay_ms: Base delay in milliseconds
            
        Returns:
            Delay in milliseconds before next retry
        """
        if strategy == "immediate":
            return 0
        elif strategy == "linear":
            return base_delay_ms * attempt
        elif strategy == "exponential":
            return base_delay_ms * (2 ** (attempt - 1))
        else:
            # Default to linear
            return base_delay_ms * attempt
    
    def track_llm_retry(
        self,
        key: str,
        error_msg: str,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None
    ) -> Optional[Event]:
        """
        Track LLM retry attempt and create RETRY event if applicable.
        
        Args:
            key: Unique key for this LLM operation (e.g., "agent_id_model_provider")
            error_msg: Error message that triggered retry
            agent_id: Optional agent ID
            agent_name: Optional agent name
            
        Returns:
            RETRY event if retry should be attempted, None otherwise
        """
        if not self.is_retryable_error(error_msg):
            return None
        
        # Get current retry count
        retry_count = self._llm_retry_attempts.get(key, 0)
        
        # Check if we've exceeded max retries
        if retry_count >= self.config.max_llm_retries:
            # Clear retry counter
            self._llm_retry_attempts.pop(key, None)
            return None
        
        # Increment retry counter
        retry_count += 1
        self._llm_retry_attempts[key] = retry_count
        
        # Calculate backoff delay
        backoff_ms = self.calculate_backoff_delay(
            self.config.llm_backoff_strategy,
            retry_count,
            self.config.llm_base_delay_ms
        )
        
        # Create RETRY event
        retry_event = self.event_builder.create_retry(
            attempt=retry_count,
            strategy=self.config.llm_backoff_strategy,
            backoff_ms=backoff_ms,
            reason=f"LLM call failed: {error_msg} (attempt {retry_count}/{self.config.max_llm_retries})",
            agent_id=agent_id,
            agent_name=agent_name
        )
        
        return retry_event
    
    def track_tool_retry(
        self,
        key: str,
        error_msg: str,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None
    ) -> Optional[Event]:
        """
        Track tool retry attempt and create RETRY event if applicable.
        
        Args:
            key: Unique key for this tool operation (e.g., "agent_id_tool_name")
            error_msg: Error message that triggered retry
            agent_id: Optional agent ID
            agent_name: Optional agent name
            
        Returns:
            RETRY event if retry should be attempted, None otherwise
        """
        if not self.is_retryable_error(error_msg):
            return None
        
        # Get current retry count
        retry_count = self._tool_retry_attempts.get(key, 0)
        
        # Check if we've exceeded max retries
        if retry_count >= self.config.max_tool_retries:
            # Clear retry counter
            self._tool_retry_attempts.pop(key, None)
            return None
        
        # Increment retry counter
        retry_count += 1
        self._tool_retry_attempts[key] = retry_count
        
        # Calculate backoff delay
        backoff_ms = self.calculate_backoff_delay(
            self.config.tool_backoff_strategy,
            retry_count,
            self.config.tool_base_delay_ms
        )
        
        # Create RETRY event
        retry_event = self.event_builder.create_retry(
            attempt=retry_count,
            strategy=self.config.tool_backoff_strategy,
            backoff_ms=backoff_ms,
            reason=f"Tool execution failed: {error_msg} (attempt {retry_count}/{self.config.max_tool_retries})",
            agent_id=agent_id,
            agent_name=agent_name
        )
        
        return retry_event
    
    def track_task_retry(
        self,
        key: str,
        error_msg: str,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None
    ) -> Optional[Event]:
        """
        Track task retry attempt and create RETRY event if applicable.
        
        Args:
            key: Unique key for this task operation (e.g., "agent_id_task_id")
            error_msg: Error message that triggered retry
            agent_id: Optional agent ID
            agent_name: Optional agent name
            
        Returns:
            RETRY event if retry should be attempted, None otherwise
        """
        if not self.is_retryable_error(error_msg):
            return None
        
        # Get current retry count
        retry_count = self._task_retry_attempts.get(key, 0)
        
        # Check if we've exceeded max retries
        if retry_count >= self.config.max_task_retries:
            # Clear retry counter
            self._task_retry_attempts.pop(key, None)
            return None
        
        # Increment retry counter
        retry_count += 1
        self._task_retry_attempts[key] = retry_count
        
        # Calculate backoff delay
        backoff_ms = self.calculate_backoff_delay(
            self.config.task_backoff_strategy,
            retry_count,
            self.config.task_base_delay_ms
        )
        
        # Create RETRY event
        retry_event = self.event_builder.create_retry(
            attempt=retry_count,
            strategy=self.config.task_backoff_strategy,
            backoff_ms=backoff_ms,
            reason=f"Task execution failed: {error_msg} (attempt {retry_count}/{self.config.max_task_retries})",
            agent_id=agent_id,
            agent_name=agent_name
        )
        
        return retry_event
    
    def clear_llm_retry(self, key: str) -> None:
        """Clear retry counter for successful LLM operation."""
        self._llm_retry_attempts.pop(key, None)
    
    def clear_tool_retry(self, key: str) -> None:
        """Clear retry counter for successful tool operation."""
        self._tool_retry_attempts.pop(key, None)
    
    def clear_task_retry(self, key: str) -> None:
        """Clear retry counter for successful task operation."""
        self._task_retry_attempts.pop(key, None)
    
    def clear_all_retries(self) -> None:
        """Clear all retry counters."""
        self._llm_retry_attempts.clear()
        self._tool_retry_attempts.clear()
        self._task_retry_attempts.clear()
    
    def get_retry_stats(self) -> Dict[str, Dict[str, int]]:
        """
        Get current retry statistics.
        
        Returns:
            Dictionary with retry counts by operation type
        """
        return {
            "llm": dict(self._llm_retry_attempts),
            "tool": dict(self._tool_retry_attempts),
            "task": dict(self._task_retry_attempts)
        }