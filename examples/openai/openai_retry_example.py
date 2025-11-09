"""
OpenAI Agents example with retry simulation.
Demonstrates comprehensive event capture including RETRY events.
"""

import os
import asyncio
import json
import random
from typing import Dict, List, Any
from datetime import datetime

# Set up environment
os.environ["CHAUKAS_TENANT_ID"] = "demo_tenant"
os.environ["CHAUKAS_PROJECT_ID"] = "openai_retry_demo"
os.environ["CHAUKAS_ENDPOINT"] = "http://localhost:8080"
os.environ["CHAUKAS_API_KEY"] = "demo_key"
os.environ["CHAUKAS_EMIT_TO_FILE"] = "true"
os.environ["CHAUKAS_FILE_PATH"] = "openai_retry_demo.jsonl"

# Import Chaukas SDK
from chaukas import sdk as chaukas


class MockOpenAIClient:
    """Mock OpenAI client for testing without real API calls."""
    pass


class MockTool:
    """Mock tool that can simulate failures."""
    
    def __init__(self, name: str, fail_probability: float = 0.5):
        self.name = name
        self.description = f"A tool that {name}"
        self.fail_probability = fail_probability
        self.call_count = 0
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool with simulated failures."""
        self.call_count += 1
        
        if random.random() < self.fail_probability and self.call_count <= 2:
            # Simulate different types of failures
            errors = [
                "Rate limit exceeded",
                "503 Service Unavailable", 
                "Connection timeout",
                "429 Too Many Requests"
            ]
            raise Exception(random.choice(errors))
        
        return {"result": f"Successfully executed {self.name}", "data": kwargs}


class MockMessage:
    """Mock message object."""
    
    def __init__(self, role: str, content: str, tool_calls=None):
        self.role = role
        self.content = content
        self.tool_calls = tool_calls


class MockToolCall:
    """Mock tool call object."""
    
    def __init__(self, tool_name: str, arguments: Dict):
        self.id = f"call_{tool_name}_{datetime.now().timestamp()}"
        self.function = type('obj', (object,), {
            'name': tool_name,
            'arguments': json.dumps(arguments)
        })()


class MockResponse:
    """Mock LLM response."""
    
    def __init__(self, content: str = None, tool_calls: List = None):
        self.content = content
        self.tool_calls = tool_calls
        self.finish_reason = "stop" if content else "tool_calls"
        
        # OpenAI response format
        self.choices = [
            type('obj', (object,), {
                'message': type('obj', (object,), {
                    'content': content,
                    'tool_calls': tool_calls
                })()
            })()
        ]


class MockRunner:
    """Mock Runner that simulates LLM invocations."""
    
    def __init__(self, agent):
        self.agent = agent
        self.run_count = 0
    
    async def run_once(self, messages: List[Dict]) -> MockResponse:
        """Simulate a single LLM invocation."""
        self.run_count += 1
        
        # Simulate rate limit on first attempt
        if self.run_count == 1:
            raise Exception("Rate limit exceeded - please retry")
        
        # Check last message
        last_message = messages[-1] if messages else {}
        content = last_message.get("content", "")
        
        # Simulate tool calls for certain inputs
        if "search" in content.lower():
            tool_calls = [
                MockToolCall("web_search", {"query": content})
            ]
            return MockResponse(tool_calls=tool_calls)
        elif "calculate" in content.lower():
            tool_calls = [
                MockToolCall("calculator", {"expression": content})
            ]
            return MockResponse(tool_calls=tool_calls)
        
        # Regular response
        return MockResponse(content=f"I understand you said: '{content}'. Here's my response based on that input.")


class MockAgent:
    """Mock OpenAI Agent for testing."""
    
    def __init__(self, name: str, instructions: str, model: str, client=None):
        self.name = name
        self.instructions = instructions
        self.model = model
        self.client = client
        self.temperature = 0.7
        self.max_tokens = 1000
        self.tools = [
            MockTool("web_search", fail_probability=0.6),
            MockTool("calculator", fail_probability=0.3),
            MockTool("weather_api", fail_probability=0.4)
        ]
        self.runner = MockRunner(self)
        self.run_attempts = 0
    
    async def run(self, messages: List[Dict]) -> MockResponse:
        """Simulate agent run with potential failures and retries."""
        self.run_attempts += 1
        
        # Simulate occasional agent-level failures
        if self.run_attempts == 1 and random.random() < 0.3:
            raise Exception("503 Service Unavailable - OpenAI service temporarily down")
        
        # Process messages and get response
        try:
            response = await self.runner.run_once(messages)
            
            # If response has tool calls, execute them
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    tool_name = tool_call.function.name
                    tool = next((t for t in self.tools if t.name == tool_name), None)
                    
                    if tool:
                        try:
                            args = json.loads(tool_call.function.arguments)
                            result = tool.execute(**args)
                            print(f"Tool {tool_name} executed successfully: {result}")
                        except Exception as e:
                            print(f"Tool {tool_name} failed: {e}")
                            # Tool failures might trigger retries
                            raise
            
            return response
            
        except Exception as e:
            print(f"Error during agent run: {e}")
            raise


# Monkey patch OpenAI module to use our mocks
import sys
mock_openai = type(sys)('openai')
mock_openai.OpenAI = MockOpenAIClient

mock_agents = type(sys)('agents')
mock_agents.Agent = MockAgent
mock_agents.Runner = MockRunner

mock_openai.agents = mock_agents
sys.modules['openai'] = mock_openai
sys.modules['openai.agents'] = mock_agents


async def main():
    """Run the example with retry simulation."""
    print("=" * 60)
    print("OpenAI Agents Retry Simulation")
    print("=" * 60)
    
    # Enable Chaukas instrumentation
    chaukas.enable_chaukas()
    print("✓ Chaukas instrumentation enabled\n")
    
    # Create mock client and agent
    client = MockOpenAIClient()
    agent = MockAgent(
        name="research-assistant",
        instructions="You are a helpful research assistant that provides accurate information.",
        model="gpt-4",
        client=client
    )
    
    print(f"✓ Created agent: {agent.name}")
    print(f"  Model: {agent.model}")
    print(f"  Tools: {[t.name for t in agent.tools]}\n")
    
    # Test scenarios
    test_messages = [
        {"role": "user", "content": "Hello! Can you help me search for information about Python?"},
        {"role": "user", "content": "Please calculate 42 * 17 for me"},
        {"role": "user", "content": "What's the weather like today?"}
    ]
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n--- Test {i}: {message['content'][:50]}...")
        
        try:
            # Run agent (may fail and retry)
            result = await agent.run([message])
            
            if result.content:
                print(f"✓ Response: {result.content[:100]}...")
            elif result.tool_calls:
                print(f"✓ Tool calls requested: {[tc.function.name for tc in result.tool_calls]}")
                
        except Exception as e:
            print(f"✗ Final failure after retries: {e}")
        
        # Reset for next test
        agent.run_attempts = 0
        agent.runner.run_count = 0
        for tool in agent.tools:
            tool.call_count = 0
        
        await asyncio.sleep(0.5)  # Small delay between tests
    
    # Flush events
    print("\n" + "=" * 60)
    print("Flushing events...")
    
    client = chaukas.get_client()
    if client:
        await client.flush()
        await client.close()
    
    print("✓ Events written to openai_retry_demo.jsonl")
    
    # Analyze the output
    print("\n" + "=" * 60)
    print("Event Analysis:")
    print("=" * 60)
    
    try:
        with open("openai_retry_demo.jsonl", "r") as f:
            events = [json.loads(line) for line in f if line.strip()]
        
        event_types = {}
        retry_events = []
        
        for event in events:
            event_type = event.get("type", "UNKNOWN")
            event_types[event_type] = event_types.get(event_type, 0) + 1
            
            if event_type == "EVENT_TYPE_RETRY":
                retry_events.append(event)
        
        print(f"\nTotal events captured: {len(events)}")
        print("\nEvent type distribution:")
        for event_type, count in sorted(event_types.items()):
            print(f"  {event_type}: {count}")
        
        if retry_events:
            print(f"\n✓ RETRY events captured: {len(retry_events)}")
            for retry in retry_events:
                if "retry" in retry:
                    r = retry["retry"]
                    print(f"  - Attempt {r.get('attempt')}: {r.get('error_message', '')[:50]}...")
                    print(f"    Strategy: {r.get('strategy')}, Backoff: {r.get('backoff_ms')}ms")
        else:
            print("\n✗ No RETRY events captured")
        
        # Check for comprehensive event coverage
        expected_events = [
            "EVENT_TYPE_SESSION_START",
            "EVENT_TYPE_SESSION_END",
            "EVENT_TYPE_AGENT_START",
            "EVENT_TYPE_AGENT_END",
            "EVENT_TYPE_MODEL_INVOCATION_START",
            "EVENT_TYPE_MODEL_INVOCATION_END",
            "EVENT_TYPE_INPUT_RECEIVED",
            "EVENT_TYPE_OUTPUT_EMITTED",
            "EVENT_TYPE_TOOL_CALL_START",
            "EVENT_TYPE_ERROR",
            "EVENT_TYPE_RETRY"
        ]
        
        print("\nEvent coverage check:")
        for event in expected_events:
            if event in event_types:
                print(f"  ✓ {event}")
            else:
                print(f"  ✗ {event} (not captured)")
        
    except FileNotFoundError:
        print("✗ Output file not found")
    except Exception as e:
        print(f"✗ Error analyzing output: {e}")


if __name__ == "__main__":
    asyncio.run(main())
    print("\nExample completed!")