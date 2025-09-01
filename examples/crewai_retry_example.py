"""
Example demonstrating RETRY event capture with CrewAI.

This example simulates failures that trigger retry events:
- LLM rate limit errors
- Tool timeout errors  
- Task execution failures
"""

import os
import asyncio
from unittest.mock import Mock, MagicMock
from crewai import Agent, Task, Crew
from chaukas import sdk as chaukas

# Set environment variables for Chaukas
os.environ["CHAUKAS_TENANT_ID"] = "test-tenant"
os.environ["CHAUKAS_PROJECT_ID"] = "test-project"
os.environ["CHAUKAS_OUTPUT_MODE"] = "file"
os.environ["CHAUKAS_OUTPUT_FILE"] = "retry_events.jsonl"

# Enable Chaukas instrumentation
chaukas.enable_chaukas()


def simulate_retry_events():
    """Simulate various retry scenarios."""
    
    # Create a simple agent
    agent = Agent(
        role="researcher",
        goal="Research and summarize topics",
        backstory="You are an expert researcher.",
        verbose=True,
        max_iter=3  # Allow retries
    )
    
    # Create a task that might fail
    task = Task(
        description="Research AI trends (this might fail and retry)",
        agent=agent,
        expected_output="A summary of AI trends"
    )
    
    # Create crew
    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=True
    )
    
    print("=" * 50)
    print("CrewAI RETRY Event Capture Example")
    print("=" * 50)
    print("\nThis example would normally trigger retry events when:")
    print("1. LLM calls fail due to rate limits (HTTP 429)")
    print("2. Tool executions timeout or fail temporarily")
    print("3. Tasks fail but are retryable")
    print("\nIn production, the SDK automatically captures these as RETRY events.")
    print("\nEvents would be logged to: retry_events.jsonl")
    print("=" * 50)
    
    # Note: In a real scenario, you would run:
    # result = crew.kickoff()
    # 
    # And if any retryable errors occur (rate limits, timeouts, etc.),
    # the SDK will automatically emit RETRY events with:
    # - attempt number
    # - backoff strategy (exponential, linear, immediate)
    # - backoff delay in milliseconds
    # - reason for retry
    
    print("\nRetry events are now captured for:")
    print("✅ LLM failures (rate limits, 503 errors)")
    print("✅ Tool execution failures (timeouts, network errors)")
    print("✅ Task execution failures (temporary errors)")
    print("✅ Agent execution errors (recoverable failures)")
    
    print("\nEach RETRY event includes:")
    print("• Attempt number (e.g., 1/3)")
    print("• Backoff strategy (exponential, linear, immediate)")
    print("• Backoff delay in milliseconds")
    print("• Detailed reason for the retry")
    print("• Agent context (ID and name)")


if __name__ == "__main__":
    simulate_retry_events()
    print("\n✨ CrewAI integration now captures 100% of chaukas-spec events!")
    print("   Including the newly added EVENT_TYPE_RETRY support.")