"""
Tests for refactored utility classes.
"""

import os
import pytest
from unittest.mock import Mock, MagicMock
from datetime import datetime, timedelta, timezone

# Set required environment variables for testing
os.environ["CHAUKAS_TENANT_ID"] = "test-tenant"
os.environ["CHAUKAS_PROJECT_ID"] = "test-project"
os.environ["CHAUKAS_ENDPOINT"] = "http://test-endpoint"
os.environ["CHAUKAS_API_KEY"] = "test-key"

from chaukas.sdk.core.event_builder import EventBuilder
from chaukas.sdk.utils.retry_detector import RetryDetector, RetryConfig
from chaukas.sdk.utils.event_pairs import EventPairManager


class TestRetryDetector:
    """Test the RetryDetector utility."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.event_builder = EventBuilder()
        self.retry_detector = RetryDetector(self.event_builder)
    
    def test_is_retryable_error(self):
        """Test retryable error detection."""
        # Retryable errors
        assert self.retry_detector.is_retryable_error("Rate limit exceeded")
        assert self.retry_detector.is_retryable_error("Connection timeout")
        assert self.retry_detector.is_retryable_error("503 Service Unavailable")
        assert self.retry_detector.is_retryable_error("429 Too Many Requests")
        assert self.retry_detector.is_retryable_error("Network error")
        assert self.retry_detector.is_retryable_error("Service temporarily unavailable")
        
        # Non-retryable errors
        assert not self.retry_detector.is_retryable_error("Invalid API key")
        assert not self.retry_detector.is_retryable_error("Syntax error")
        assert not self.retry_detector.is_retryable_error("Permission denied")
    
    def test_calculate_backoff_delay(self):
        """Test backoff delay calculation."""
        # Immediate backoff
        assert self.retry_detector.calculate_backoff_delay("immediate", 1, 1000) == 0
        assert self.retry_detector.calculate_backoff_delay("immediate", 5, 1000) == 0
        
        # Linear backoff
        assert self.retry_detector.calculate_backoff_delay("linear", 1, 500) == 500
        assert self.retry_detector.calculate_backoff_delay("linear", 2, 500) == 1000
        assert self.retry_detector.calculate_backoff_delay("linear", 3, 500) == 1500
        
        # Exponential backoff
        assert self.retry_detector.calculate_backoff_delay("exponential", 1, 1000) == 1000
        assert self.retry_detector.calculate_backoff_delay("exponential", 2, 1000) == 2000
        assert self.retry_detector.calculate_backoff_delay("exponential", 3, 1000) == 4000
        assert self.retry_detector.calculate_backoff_delay("exponential", 4, 1000) == 8000
    
    def test_track_llm_retry(self):
        """Test LLM retry tracking."""
        key = "agent1_gpt4_openai"
        
        # First retry - should create RETRY event
        retry_event = self.retry_detector.track_llm_retry(
            key, "Rate limit error", "agent1", "TestAgent"
        )
        assert retry_event is not None
        assert retry_event.retry.attempt == 1
        assert retry_event.retry.strategy == "exponential"
        assert retry_event.retry.backoff_ms == 1000
        
        # Second retry
        retry_event = self.retry_detector.track_llm_retry(
            key, "429 Too Many Requests", "agent1", "TestAgent"
        )
        assert retry_event is not None
        assert retry_event.retry.attempt == 2
        assert retry_event.retry.backoff_ms == 2000
        
        # Third retry
        retry_event = self.retry_detector.track_llm_retry(
            key, "Service unavailable", "agent1", "TestAgent"
        )
        assert retry_event is not None
        assert retry_event.retry.attempt == 3
        assert retry_event.retry.backoff_ms == 4000
        
        # Fourth attempt - exceeds max retries
        retry_event = self.retry_detector.track_llm_retry(
            key, "Still failing", "agent1", "TestAgent"
        )
        assert retry_event is None  # No more retries
    
    def test_clear_retry_counters(self):
        """Test clearing retry counters."""
        key = "agent1_tool1"
        
        # Track a retry
        self.retry_detector.track_tool_retry(key, "Timeout error")
        assert key in self.retry_detector._tool_retry_attempts
        
        # Clear the retry
        self.retry_detector.clear_tool_retry(key)
        assert key not in self.retry_detector._tool_retry_attempts
        
        # Clear all retries
        self.retry_detector.track_llm_retry("llm1", "Error")
        self.retry_detector.track_tool_retry("tool1", "Error")
        self.retry_detector.track_task_retry("task1", "Error")
        
        self.retry_detector.clear_all_retries()
        assert len(self.retry_detector._llm_retry_attempts) == 0
        assert len(self.retry_detector._tool_retry_attempts) == 0
        assert len(self.retry_detector._task_retry_attempts) == 0
    
    def test_custom_config(self):
        """Test custom retry configuration."""
        config = RetryConfig(
            max_llm_retries=5,
            llm_backoff_strategy="linear",
            llm_base_delay_ms=2000
        )
        detector = RetryDetector(self.event_builder, config)
        
        # Should allow up to 5 retries
        key = "test_llm"
        for i in range(5):
            retry_event = detector.track_llm_retry(key, "Rate limit error", "agent1", "Test")
            assert retry_event is not None
            assert retry_event.retry.strategy == "linear"
            assert retry_event.retry.backoff_ms == 2000 * (i + 1)
        
        # Sixth attempt should fail
        retry_event = detector.track_llm_retry(key, "Error", "agent1", "Test")
        assert retry_event is None


class TestEventPairManager:
    """Test the EventPairManager utility."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = EventPairManager()
    
    def test_register_and_retrieve_span_id(self):
        """Test registering START event and retrieving span_id for END."""
        # Register START event
        span_id = self.manager.register_start_event(
            "AGENT", "agent123", "span-abc-123"
        )
        assert span_id == "span-abc-123"
        
        # Retrieve for END event
        retrieved_span_id = self.manager.get_span_id_for_end(
            "AGENT", "agent123"
        )
        assert retrieved_span_id == "span-abc-123"
        
        # Should be cleared after retrieval
        retrieved_again = self.manager.get_span_id_for_end(
            "AGENT", "agent123"
        )
        assert retrieved_again is None
    
    def test_orphaned_end_events(self):
        """Test tracking of END events without START."""
        # Try to get span_id for END without START
        span_id = self.manager.get_span_id_for_end(
            "MODEL_INVOCATION", "model123"
        )
        assert span_id is None
        
        # Should be tracked as orphaned
        orphans = self.manager.get_orphaned_ends()
        assert ("MODEL_INVOCATION", "model123") in orphans
    
    def test_clear_pair_on_error(self):
        """Test clearing pairs on error."""
        # Register START
        self.manager.register_start_event(
            "TOOL_CALL", "tool456", "span-def-456"
        )
        
        # Clear on error
        cleared = self.manager.clear_pair("TOOL_CALL", "tool456")
        assert cleared is True
        
        # Should not be retrievable
        span_id = self.manager.get_span_id_for_end(
            "TOOL_CALL", "tool456"
        )
        assert span_id is None
    
    def test_cleanup_stale_pairs(self):
        """Test cleanup of stale START events."""
        # Register a START event
        self.manager.register_start_event(
            "SESSION", "session789", "span-ghi-789"
        )
        
        # Manually set start_time to be old
        key = ("SESSION", "session789")
        self.manager._active_pairs[key].start_time = datetime.now(timezone.utc) - timedelta(hours=2)
        
        # Cleanup with 1 hour max age
        cleaned = self.manager.cleanup_stale_pairs(max_age_seconds=3600)
        assert cleaned == 1
        
        # Should be gone
        assert key not in self.manager._active_pairs
    
    def test_get_stats(self):
        """Test statistics retrieval."""
        # Register some events
        self.manager.register_start_event("AGENT", "a1", "s1")
        self.manager.register_start_event("AGENT", "a2", "s2")
        self.manager.get_span_id_for_end("TOOL", "t1")  # Orphan
        
        stats = self.manager.get_stats()
        assert stats["active_pairs"] == 2
        assert stats["orphaned_ends"] == 1
        assert stats["total_tracked"] == 3
    
    def test_reset(self):
        """Test resetting the manager."""
        # Add some data
        self.manager.register_start_event("AGENT", "a1", "s1")
        self.manager.get_span_id_for_end("TOOL", "t1")
        
        # Reset
        self.manager.reset()
        
        # Should be empty
        assert len(self.manager._active_pairs) == 0
        assert len(self.manager._orphaned_ends) == 0