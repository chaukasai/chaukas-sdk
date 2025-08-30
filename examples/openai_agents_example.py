"""
Example usage with OpenAI Agents SDK.
"""

import os
import asyncio
from openai import OpenAI
from openai.agents import Agent

# Import and enable Chaukas instrumentation
import chaukas

# Set environment variables
os.environ["CHAUKAS_ENDPOINT"] = "https://api.chaukas.com"
os.environ["CHAUKAS_API_KEY"] = "your-api-key-here"

# Enable instrumentation - this will automatically detect and patch OpenAI Agents
chaukas.enable_chaukas()


async def main():
    # Create OpenAI client
    client = OpenAI(api_key="your-openai-api-key")
    
    # Create agent - this will be automatically instrumented
    agent = Agent(
        name="research-assistant",
        instructions="You are a helpful research assistant that provides accurate information.",
        model="gpt-4",
        client=client,
    )
    
    # Run agent - all LLM calls, tool usage, and agent lifecycle will be traced
    result = await agent.run(
        messages=[{"role": "user", "content": "What is the capital of France?"}]
    )
    
    print(f"Agent response: {result}")
    
    # Flush any remaining events before exit
    client = chaukas.get_client()
    if client:
        await client.flush()


if __name__ == "__main__":
    asyncio.run(main())