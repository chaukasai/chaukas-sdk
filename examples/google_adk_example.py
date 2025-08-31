"""
Example usage with Google ADK.
"""

import os
import asyncio
from adk import Agent

# Import and enable Chaukas instrumentation
from chaukas import sdk as chaukas

# Set environment variables
os.environ["CHAUKAS_ENDPOINT"] = "https://api.chaukas.com"
os.environ["CHAUKAS_API_KEY"] = "your-api-key-here"
os.environ["GOOGLE_API_KEY"] = "your-google-api-key"

# Enable instrumentation - this will automatically detect and patch Google ADK
chaukas.enable_chaukas()


async def main():
    # Create agent - this will be automatically instrumented
    agent = Agent(
        name="math-tutor",
        model="gemini-2.0-flash",
        instruction="You are a helpful math tutor that explains concepts clearly.",
        description="A specialized agent for teaching mathematics",
    )
    
    # Run agent - all interactions will be traced
    result = await agent.run("Explain the quadratic formula")
    
    print(f"Agent response: {result}")
    
    # Flush any remaining events before exit
    client = chaukas.get_client()
    if client:
        await client.flush()


if __name__ == "__main__":
    asyncio.run(main())