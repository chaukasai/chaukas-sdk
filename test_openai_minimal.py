#!/usr/bin/env python
"""Minimal test of OpenAI agents integration."""

import os
import asyncio

# Set required environment variables
os.environ["CHAUKAS_TENANT_ID"] = "test"
os.environ["CHAUKAS_PROJECT_ID"] = "test"
os.environ["CHAUKAS_OUTPUT_MODE"] = "file"
os.environ["CHAUKAS_OUTPUT_FILE"] = "test_output.jsonl"

from chaukas import sdk as chaukas
from agents import Agent, Runner, function_tool

@function_tool
def test_tool(input: str) -> str:
    """Test tool."""
    return f"Result for: {input}"

async def main():
    print("Testing OpenAI Agents integration...")
    
    # Enable Chaukas
    chaukas.enable_chaukas()
    print("Chaukas enabled")
    
    # Create simple agent
    agent = Agent(
        name="test_agent",
        instructions="You are a test agent. Use the test_tool when asked.",
        model="gpt-4o-mini",
        tools=[test_tool]
    )
    
    # Try to run
    try:
        print("Calling Runner.run...")
        result = await Runner.run(agent, "Test message - use the test tool")
        print(f"Success! Result: {result}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())