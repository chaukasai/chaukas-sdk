"""
Demonstration of RETRY event capture with CrewAI.

This example creates a custom tool that simulates failures to demonstrate
how retry events are captured by the Chaukas SDK.
"""

import os
import asyncio
import random
from typing import Type
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

# Disable CrewAI telemetry
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

# Set environment variables for Chaukas
os.environ["CHAUKAS_TENANT_ID"] = "demo_tenant"
os.environ["CHAUKAS_PROJECT_ID"] = "retry_demo"
os.environ["CHAUKAS_OUTPUT_MODE"] = "file"
os.environ["CHAUKAS_OUTPUT_FILE"] = "retry_events_demo.jsonl"
os.environ["CHAUKAS_BATCH_SIZE"] = "1"  # Write events immediately

# Import and enable Chaukas instrumentation
from chaukas import sdk as chaukas

# Enable instrumentation - this will automatically detect and patch CrewAI
chaukas.enable_chaukas()


class FlakeySearchInput(BaseModel):
    """Input for the flakey search tool."""
    query: str = Field(description="The search query")


class FlakeySearchTool(BaseTool):
    """A tool that sometimes fails to simulate retry scenarios."""
    
    name: str = "flakey_search"
    description: str = "Search for information (may fail and retry)"
    args_schema: Type[BaseModel] = FlakeySearchInput
    attempt_count: int = 0  # Define as a class field for Pydantic
    
    def _run(self, query: str) -> str:
        """Execute the search with simulated failures."""
        self.attempt_count += 1
        
        # Simulate different types of failures
        if self.attempt_count == 1:
            # First attempt: Rate limit error (retryable)
            raise Exception("429 Rate limit exceeded. Please retry after 1 second.")
        elif self.attempt_count == 2:
            # Second attempt: Timeout error (retryable)
            raise Exception("Connection timeout while fetching search results")
        elif self.attempt_count == 3:
            # Third attempt: Success!
            return f"Search results for '{query}': Found 10 relevant articles about {query}"
        else:
            # Backup success case
            return f"Search completed for '{query}'"


class NetworkAnalysisInput(BaseModel):
    """Input for network analysis tool."""
    data: str = Field(description="Data to analyze")


class NetworkAnalysisTool(BaseTool):
    """A tool that simulates network failures."""
    
    name: str = "network_analysis"
    description: str = "Analyze network data (may experience network issues)"
    args_schema: Type[BaseModel] = NetworkAnalysisInput
    call_count: int = 0  # Define as a class field for Pydantic
    
    def _run(self, data: str) -> str:
        """Analyze data with simulated network issues."""
        self.call_count += 1
        
        if self.call_count == 1:
            # Network unavailable error (retryable)
            raise Exception("Network unavailable. Service temporarily down (503)")
        else:
            return f"Analysis complete: {data} shows positive trends"


def main():
    print("=" * 60)
    print("CrewAI RETRY Event Capture Demonstration")
    print("=" * 60)
    print("\nThis demo will simulate failures to trigger RETRY events.")
    print("Watch for retry attempts in the output...")
    print("=" * 60 + "\n")
    
    # Create tools that will fail and retry
    flakey_tool = FlakeySearchTool()
    network_tool = NetworkAnalysisTool()
    
    # Create an agent with the flakey tools
    researcher = Agent(
        role='Research Analyst',
        goal='Gather information despite unreliable tools',
        backstory='You are persistent and retry when tools fail temporarily.',
        tools=[flakey_tool, network_tool],
        verbose=True,
        max_iter=5  # Allow multiple attempts
    )
    
    # Create a simple task
    research_task = Task(
        description="""Use the flakey_search tool to find information about 'AI trends'.
        Then use the network_analysis tool to analyze the results.
        Be persistent if tools fail - they often work on retry.""",
        expected_output="A summary of AI trends with analysis",
        agent=researcher
    )
    
    # Create and run crew
    crew = Crew(
        agents=[researcher],
        tasks=[research_task],
        verbose=True,
        process=Process.sequential
    )
    
    try:
        # Execute crew - this will trigger failures and retries
        print("Starting crew execution (expect to see retry attempts)...\n")
        result = crew.kickoff()
        
        print("\n" + "=" * 60)
        print("EXECUTION COMPLETE")
        print("=" * 60)
        print(f"\nResult: {result}")
        
    except Exception as e:
        print(f"\nExecution failed with error: {e}")
    
    # Flush events to ensure file is written
    print("\n" + "=" * 60)
    print("Flushing events to file...")
    client = chaukas.get_client()
    if client:
        # For sync context, we need to run the async flush
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, create a task
                task = asyncio.create_task(client.flush())
            else:
                # If no loop is running, run until complete
                loop.run_until_complete(client.flush())
        except:
            # Fallback: create new event loop
            asyncio.run(client.flush())
        
        print(f"✅ Events written to: retry_events_demo.jsonl")
    
    print("\n" + "=" * 60)
    print("RETRY Events Captured:")
    print("=" * 60)
    print("The following retry events should be in the output file:")
    print("1. 🔄 RETRY for rate limit error (exponential backoff)")
    print("2. 🔄 RETRY for timeout error (linear backoff)")
    print("3. 🔄 RETRY for network unavailable (exponential backoff)")
    print("\nEach retry event includes:")
    print("• Attempt number and backoff strategy")
    print("• Delay in milliseconds before next attempt")
    print("• Detailed reason for the retry")
    print("• Agent context (ID and name)")
    print("=" * 60)


if __name__ == "__main__":
    main()