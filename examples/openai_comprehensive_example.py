"""
Comprehensive OpenAI Agents example demonstrating all supported events.

This example showcases:
- SESSION_START/END lifecycle
- AGENT_START/END tracking
- MODEL_INVOCATION_START/END
- TOOL_CALL_START detection
- INPUT_RECEIVED/OUTPUT_EMITTED
- ERROR and RETRY events

Supports running with or without actual OpenAI API.
"""

import os
import sys
import json
import asyncio
import random
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict
import time

# Set up environment for Chaukas
os.environ["CHAUKAS_TENANT_ID"] = "demo_tenant"
os.environ["CHAUKAS_PROJECT_ID"] = "openai_comprehensive_demo"
os.environ["CHAUKAS_OUTPUT_MODE"] = "file"
os.environ["CHAUKAS_OUTPUT_FILE"] = "openai_comprehensive_output.jsonl"
os.environ["CHAUKAS_BATCH_SIZE"] = "1"  # Immediate write for demo

# Import Chaukas SDK
from chaukas import sdk as chaukas


# ============================================================================
# Mock OpenAI SDK Implementation (for testing without API key)
# ============================================================================

class MockTool:
    """Mock tool that can be called by agents."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.call_count = 0
        self.fail_next = False  # For simulating failures
    
    def __call__(self, **kwargs) -> str:
        """Execute the tool."""
        self.call_count += 1
        
        if self.fail_next:
            self.fail_next = False
            raise Exception(f"{self.name} temporarily unavailable")
        
        # Simulate different tools
        if self.name == "web_search":
            query = kwargs.get("query", "")
            return f"Found 10 results for '{query}': Latest news, research papers, and tutorials."
        
        elif self.name == "calculator":
            expression = kwargs.get("expression", "")
            try:
                result = eval(expression)  # In production, use safer methods
                return f"Result: {expression} = {result}"
            except:
                return f"Error: Could not calculate {expression}"
        
        elif self.name == "weather_api":
            location = kwargs.get("location", "")
            return f"Weather in {location}: Sunny, 72°F (22°C), light breeze"
        
        elif self.name == "mcp_context_fetch":
            # Simulate an MCP-like tool (though OpenAI doesn't support MCP)
            context = kwargs.get("context", "")
            return f"Context retrieved: {context} - Full document available"
        
        else:
            return f"{self.name} executed with args: {kwargs}"


class MockToolCall:
    """Mock tool call from LLM."""
    
    def __init__(self, tool_name: str, arguments: Dict):
        self.id = f"call_{tool_name}_{int(time.time() * 1000)}"
        self.function = type('Function', (), {
            'name': tool_name,
            'arguments': json.dumps(arguments)
        })()


class MockMessage:
    """Mock message object."""
    
    def __init__(self, role: str, content: str, tool_calls: List[MockToolCall] = None):
        self.role = role
        self.content = content
        self.tool_calls = tool_calls


class MockChoice:
    """Mock choice in response."""
    
    def __init__(self, message: MockMessage, finish_reason: str = "stop"):
        self.message = message
        self.finish_reason = finish_reason
        self.index = 0


class MockResponse:
    """Mock LLM response."""
    
    def __init__(self, content: str = None, tool_calls: List[MockToolCall] = None):
        self.content = content
        self.tool_calls = tool_calls
        self.finish_reason = "stop" if content else "tool_calls"
        
        # Build response in OpenAI format
        message = MockMessage("assistant", content, tool_calls)
        self.choices = [MockChoice(message, self.finish_reason)]
        
        # Token counts (simulated)
        self.usage = type('Usage', (), {
            'prompt_tokens': random.randint(50, 200),
            'completion_tokens': random.randint(20, 150),
            'total_tokens': 0
        })()
        self.usage.total_tokens = self.usage.prompt_tokens + self.usage.completion_tokens


class MockRunner:
    """Mock Runner for LLM invocations."""
    
    def __init__(self, agent):
        self.agent = agent
        self.invocation_count = 0
        self.simulate_rate_limit = False
    
    async def run_once(self, messages: List[Dict]) -> MockResponse:
        """Simulate a single LLM invocation."""
        self.invocation_count += 1
        
        # Simulate rate limiting on occasion
        if self.simulate_rate_limit or (self.invocation_count == 1 and random.random() < 0.3):
            self.simulate_rate_limit = False
            raise Exception("Rate limit exceeded - please retry after 1 second")
        
        # Simulate service unavailable occasionally
        if random.random() < 0.1:
            raise Exception("503 Service Unavailable")
        
        # Analyze the last message
        if not messages:
            return MockResponse(content="Hello! How can I help you today?")
        
        last_msg = messages[-1]
        content = last_msg.get("content", "").lower()
        
        # Simulate tool calling based on content
        if "search" in content or "find" in content:
            tool_calls = [MockToolCall("web_search", {"query": content})]
            return MockResponse(tool_calls=tool_calls)
        
        elif "calculate" in content or "math" in content or any(op in content for op in ['+', '-', '*', '/']):
            # Extract math expression
            import re
            expr_match = re.search(r'[\d\s\+\-\*/\(\)]+', content)
            if expr_match:
                tool_calls = [MockToolCall("calculator", {"expression": expr_match.group()})]
                return MockResponse(tool_calls=tool_calls)
        
        elif "weather" in content:
            # Extract location
            locations = ["New York", "London", "Tokyo", "Paris", "Sydney"]
            location = next((loc for loc in locations if loc.lower() in content), "New York")
            tool_calls = [MockToolCall("weather_api", {"location": location})]
            return MockResponse(tool_calls=tool_calls)
        
        # Regular response
        responses = [
            f"I understand you're asking about: {content[:50]}",
            f"Based on your question, here's what I found...",
            f"Let me help you with that.",
            f"Here's my analysis of your request.",
            f"I've processed your query about {content[:30]}..."
        ]
        
        return MockResponse(content=random.choice(responses))


class MockAgent:
    """Mock OpenAI Agent."""
    
    def __init__(self, name: str, instructions: str, model: str = "gpt-4", 
                 tools: List[MockTool] = None, temperature: float = 0.7,
                 max_tokens: int = 1000):
        self.name = name
        self.instructions = instructions
        self.model = model
        self.tools = tools or []
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.runner = MockRunner(self)
        self.run_count = 0
    
    async def run(self, messages: List[Dict]) -> MockResponse:
        """Run the agent with messages."""
        self.run_count += 1
        
        # Simulate occasional agent-level failures
        if self.run_count == 1 and random.random() < 0.2:
            raise Exception("Agent initialization failed - retrying")
        
        try:
            # Get LLM response
            response = await self.runner.run_once(messages)
            
            # If we have tool calls, execute them
            if response.tool_calls and self.tools:
                for tool_call in response.tool_calls:
                    tool_name = tool_call.function.name
                    tool = next((t for t in self.tools if t.name == tool_name), None)
                    
                    if tool:
                        try:
                            args = json.loads(tool_call.function.arguments)
                            result = tool(**args)
                            print(f"  🔧 Tool '{tool_name}' executed: {result[:80]}...")
                        except Exception as e:
                            print(f"  ❌ Tool '{tool_name}' failed: {e}")
                            if "unavailable" in str(e).lower():
                                raise  # Propagate for retry
            
            return response
            
        except Exception as e:
            print(f"  ⚠️  Error: {e}")
            raise


class MockOpenAI:
    """Mock OpenAI client."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or "mock-key"


# ============================================================================
# Real OpenAI SDK Support (if available)
# ============================================================================

USE_REAL_OPENAI = False
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if OPENAI_API_KEY and '--use-real-api' in sys.argv:
    try:
        from openai import OpenAI
        from openai.agents import Agent, Runner
        USE_REAL_OPENAI = True
        print("✅ Using real OpenAI API")
    except ImportError:
        print("⚠️  OpenAI SDK not installed, using mock implementation")
else:
    print("ℹ️  Using mock OpenAI implementation (no API key required)")


# ============================================================================
# Monkey-patch for testing
# ============================================================================

if not USE_REAL_OPENAI:
    # Create mock modules
    mock_openai = type(sys)('openai')
    mock_openai.OpenAI = MockOpenAI
    
    mock_agents = type(sys)('agents')
    mock_agents.Agent = MockAgent
    mock_agents.Runner = MockRunner
    
    mock_openai.agents = mock_agents
    sys.modules['openai'] = mock_openai
    sys.modules['openai.agents'] = mock_agents


# ============================================================================
# Agent Scenarios
# ============================================================================

async def research_assistant_scenario():
    """Scenario 1: Research Assistant with web search."""
    print("\n" + "="*60)
    print("Scenario 1: Research Assistant")
    print("="*60)
    
    if USE_REAL_OPENAI:
        from openai import OpenAI
        from openai.agents import Agent
        client = OpenAI(api_key=OPENAI_API_KEY)
        agent = Agent(
            name="research_assistant",
            instructions="You are a helpful research assistant that finds and summarizes information.",
            model="gpt-4",
            client=client
        )
    else:
        tools = [
            MockTool("web_search", "Search the web for information"),
            MockTool("mcp_context_fetch", "Fetch context from MCP server")
        ]
        agent = MockAgent(
            name="research_assistant",
            instructions="You are a helpful research assistant that finds and summarizes information.",
            model="gpt-4",
            tools=tools
        )
    
    conversations = [
        [{"role": "user", "content": "Find information about quantum computing applications"}],
        [{"role": "user", "content": "Search for the latest AI developments in 2024"}],
        [{"role": "user", "content": "What are the best practices for distributed systems?"}]
    ]
    
    for i, messages in enumerate(conversations, 1):
        print(f"\n📝 Query {i}: {messages[0]['content']}")
        
        try:
            response = await agent.run(messages)
            
            if hasattr(response, 'content') and response.content:
                print(f"✅ Response: {response.content[:100]}...")
            elif hasattr(response, 'tool_calls') and response.tool_calls:
                print(f"🔧 Tools requested: {[tc.function.name for tc in response.tool_calls]}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        await asyncio.sleep(0.5)


async def math_tutor_scenario():
    """Scenario 2: Math Tutor with calculator."""
    print("\n" + "="*60)
    print("Scenario 2: Math Tutor")
    print("="*60)
    
    if USE_REAL_OPENAI:
        from openai import OpenAI
        from openai.agents import Agent
        client = OpenAI(api_key=OPENAI_API_KEY)
        agent = Agent(
            name="math_tutor",
            instructions="You are a friendly math tutor. Help students with calculations and explanations.",
            model="gpt-3.5-turbo",
            temperature=0.5,
            client=client
        )
    else:
        tools = [MockTool("calculator", "Perform mathematical calculations")]
        agent = MockAgent(
            name="math_tutor",
            instructions="You are a friendly math tutor. Help students with calculations and explanations.",
            model="gpt-3.5-turbo",
            tools=tools,
            temperature=0.5
        )
    
    conversations = [
        [{"role": "user", "content": "Calculate 42 * 17 + 238"}],
        [{"role": "user", "content": "What is (15 + 8) * 3 - 10?"}],
        [{"role": "user", "content": "Solve: 2x + 5 = 15"}]  # This will just get a response, not calculation
    ]
    
    for i, messages in enumerate(conversations, 1):
        print(f"\n📐 Problem {i}: {messages[0]['content']}")
        
        try:
            response = await agent.run(messages)
            
            if hasattr(response, 'content') and response.content:
                print(f"✅ Solution: {response.content[:100]}...")
            elif hasattr(response, 'tool_calls') and response.tool_calls:
                print(f"🔧 Calculating with: {[tc.function.name for tc in response.tool_calls]}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        await asyncio.sleep(0.5)


async def travel_planner_scenario():
    """Scenario 3: Travel Planner with weather and retry simulation."""
    print("\n" + "="*60)
    print("Scenario 3: Travel Planner (with retry simulation)")
    print("="*60)
    
    if USE_REAL_OPENAI:
        from openai import OpenAI
        from openai.agents import Agent
        client = OpenAI(api_key=OPENAI_API_KEY)
        agent = Agent(
            name="travel_planner",
            instructions="You are a travel planning assistant. Help users plan their trips.",
            model="gpt-4",
            temperature=0.8,
            max_tokens=500,
            client=client
        )
    else:
        tools = [
            MockTool("weather_api", "Get weather information for a location"),
            MockTool("web_search", "Search for travel information")
        ]
        agent = MockAgent(
            name="travel_planner",
            instructions="You are a travel planning assistant. Help users plan their trips.",
            model="gpt-4",
            tools=tools,
            temperature=0.8,
            max_tokens=500
        )
        
        # Simulate rate limiting for retry demonstration
        agent.runner.simulate_rate_limit = True
    
    messages = [
        {"role": "user", "content": "What's the weather like in Paris? I'm planning a trip."},
        {"role": "assistant", "content": "I'll check the weather in Paris for you."},
        {"role": "user", "content": "Also, find the best tourist attractions there."}
    ]
    
    print(f"🗺️ Multi-turn conversation:")
    for msg in messages:
        if msg['role'] == 'user':
            print(f"  👤 User: {msg['content']}")
        else:
            print(f"  🤖 Assistant: {msg['content']}")
    
    # Simulate retries
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"\n  Attempt {attempt + 1}/{max_retries}...")
            response = await agent.run(messages)
            
            if hasattr(response, 'content') and response.content:
                print(f"  ✅ Response: {response.content[:100]}...")
            elif hasattr(response, 'tool_calls') and response.tool_calls:
                print(f"  🔧 Tools used: {[tc.function.name for tc in response.tool_calls]}")
            
            break  # Success
            
        except Exception as e:
            print(f"  ⚠️  Error: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"  ⏳ Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            else:
                print(f"  ❌ Max retries exceeded")


async def error_scenario():
    """Scenario 4: Error and recovery demonstration."""
    print("\n" + "="*60)
    print("Scenario 4: Error Handling and Recovery")
    print("="*60)
    
    if USE_REAL_OPENAI:
        from openai import OpenAI
        from openai.agents import Agent
        client = OpenAI(api_key=OPENAI_API_KEY)
        agent = Agent(
            name="error_test_agent",
            instructions="Test agent for error scenarios.",
            model="gpt-4",
            client=client
        )
    else:
        tools = [MockTool("web_search", "Search tool that might fail")]
        # Make the tool fail initially
        tools[0].fail_next = True
        
        agent = MockAgent(
            name="error_test_agent",
            instructions="Test agent for error scenarios.",
            model="gpt-4",
            tools=tools
        )
    
    test_cases = [
        {"content": "Search for information (will fail first time)", "should_fail": True},
        {"content": "Simple query that should work", "should_fail": False},
        {"content": "Another search query", "should_fail": False}
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n🧪 Test {i}: {test['content']}")
        
        messages = [{"role": "user", "content": test['content']}]
        
        try:
            response = await agent.run(messages)
            print(f"  ✅ Success: Got response")
            
        except Exception as e:
            print(f"  ❌ Error captured: {e}")
            
            # Retry once
            print(f"  ⏳ Retrying...")
            await asyncio.sleep(1)
            
            try:
                response = await agent.run(messages)
                print(f"  ✅ Retry successful!")
            except Exception as e2:
                print(f"  ❌ Retry also failed: {e2}")


# ============================================================================
# Event Analysis
# ============================================================================

def analyze_events(filename: str):
    """Analyze captured events from JSONL file."""
    print("\n" + "="*60)
    print("Event Analysis")
    print("="*60)
    
    try:
        with open(filename, 'r') as f:
            events = [json.loads(line) for line in f if line.strip()]
        
        if not events:
            print("❌ No events captured")
            return
        
        # Count event types
        event_counts = defaultdict(int)
        retry_events = []
        error_events = []
        tool_calls = []
        
        for event in events:
            event_type = event.get('type', 'UNKNOWN')
            event_counts[event_type] += 1
            
            if event_type == 'EVENT_TYPE_RETRY':
                retry_events.append(event)
            elif event_type == 'EVENT_TYPE_ERROR':
                error_events.append(event)
            elif event_type == 'EVENT_TYPE_TOOL_CALL_START':
                tool_calls.append(event)
        
        # Display summary
        print(f"\n📊 Total Events Captured: {len(events)}")
        print("\n📈 Event Distribution:")
        
        # Sort by count
        sorted_events = sorted(event_counts.items(), key=lambda x: x[1], reverse=True)
        for event_type, count in sorted_events:
            event_name = event_type.replace('EVENT_TYPE_', '')
            print(f"  {event_name:25} : {count:3} events")
        
        # Show retry details
        if retry_events:
            print(f"\n🔄 Retry Events ({len(retry_events)} total):")
            for retry in retry_events[:3]:  # Show first 3
                if 'metadata' in retry:
                    meta = retry['metadata']
                    print(f"  - Attempt {meta.get('attempt', 'N/A')}: {meta.get('retry_reason', 'N/A')[:50]}...")
        
        # Show error details
        if error_events:
            print(f"\n❌ Error Events ({len(error_events)} total):")
            for error in error_events[:3]:  # Show first 3
                if 'error' in error:
                    err = error['error']
                    print(f"  - {err.get('error_code', 'N/A')}: {err.get('error_message', 'N/A')[:50]}...")
        
        # Show tool calls
        if tool_calls:
            print(f"\n🔧 Tool Calls ({len(tool_calls)} total):")
            tool_names = defaultdict(int)
            for tc in tool_calls:
                if 'tool_call' in tc:
                    tool_names[tc['tool_call'].get('tool_name', 'unknown')] += 1
            
            for tool_name, count in tool_names.items():
                print(f"  - {tool_name}: {count} calls")
        
        # Check event pairing
        print("\n🔗 Event Pairing Check:")
        start_types = ['SESSION_START', 'AGENT_START', 'MODEL_INVOCATION_START', 'TOOL_CALL_START']
        
        for event_base in start_types:
            start_count = event_counts.get(f'EVENT_TYPE_{event_base}', 0)
            end_count = event_counts.get(f'EVENT_TYPE_{event_base.replace("START", "END")}', 0)
            
            if start_count > 0 or end_count > 0:
                status = "✅" if start_count == end_count else "⚠️"
                print(f"  {status} {event_base.replace('_START', ''):20} : {start_count} START, {end_count} END")
        
        # Coverage report
        print("\n📋 Event Coverage Report:")
        supported_events = [
            'SESSION_START', 'SESSION_END',
            'AGENT_START', 'AGENT_END',
            'MODEL_INVOCATION_START', 'MODEL_INVOCATION_END',
            'TOOL_CALL_START',  # TOOL_CALL_END requires tool execution patching
            'INPUT_RECEIVED', 'OUTPUT_EMITTED',
            'ERROR', 'RETRY'
        ]
        
        captured = 0
        for event in supported_events:
            if event_counts.get(f'EVENT_TYPE_{event}', 0) > 0:
                captured += 1
                print(f"  ✅ {event}")
            else:
                print(f"  ❌ {event} (not captured in this run)")
        
        print(f"\n📊 Coverage: {captured}/{len(supported_events)} supported events captured")
        print(f"   ({captured/len(supported_events)*100:.1f}% of supported events)")
        
    except FileNotFoundError:
        print(f"❌ Output file '{filename}' not found")
    except Exception as e:
        print(f"❌ Error analyzing events: {e}")


# ============================================================================
# Main Program
# ============================================================================

async def main():
    """Main program with interactive menu."""
    print("="*60)
    print("OpenAI Agents Comprehensive Example")
    print("Demonstrating Enhanced Event Capture")
    print("="*60)
    
    # Enable Chaukas instrumentation
    chaukas.enable_chaukas()
    print("✅ Chaukas instrumentation enabled")
    
    if USE_REAL_OPENAI:
        print(f"✅ Using real OpenAI API with key: {OPENAI_API_KEY[:10]}...")
    else:
        print("ℹ️  Using mock implementation (no API key required)")
    
    while True:
        print("\n" + "="*60)
        print("Select a scenario to run:")
        print("="*60)
        print("1. Research Assistant (with web search)")
        print("2. Math Tutor (with calculator)")
        print("3. Travel Planner (with retries)")
        print("4. Error Handling Demo")
        print("5. Run All Scenarios")
        print("6. Analyze Captured Events")
        print("0. Exit")
        print("-"*60)
        
        try:
            choice = input("Enter your choice (0-6): ").strip()
            
            if choice == '0':
                break
            elif choice == '1':
                await research_assistant_scenario()
            elif choice == '2':
                await math_tutor_scenario()
            elif choice == '3':
                await travel_planner_scenario()
            elif choice == '4':
                await error_scenario()
            elif choice == '5':
                print("\n🚀 Running all scenarios...")
                await research_assistant_scenario()
                await math_tutor_scenario()
                await travel_planner_scenario()
                await error_scenario()
            elif choice == '6':
                analyze_events("openai_comprehensive_output.jsonl")
            else:
                print("❌ Invalid choice, please try again")
                
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Cleanup
    print("\n" + "="*60)
    print("Shutting down...")
    
    # Flush events
    client = chaukas.get_client()
    if client:
        await client.flush()
        await client.close()
        print("✅ Events flushed and client closed")
    
    # Final analysis
    print("\n" + "="*60)
    print("Final Event Analysis")
    analyze_events("openai_comprehensive_output.jsonl")
    
    print("\n✅ Example completed successfully!")
    print("📄 Events saved to: openai_comprehensive_output.jsonl")


if __name__ == "__main__":
    # Parse command line arguments
    if '--help' in sys.argv:
        print("""
Usage: python openai_comprehensive_example.py [options]

Options:
  --use-real-api    Use real OpenAI API (requires OPENAI_API_KEY env var)
  --help           Show this help message

Environment Variables:
  OPENAI_API_KEY   Your OpenAI API key (required for --use-real-api)

By default, uses a mock implementation that doesn't require an API key.
        """)
        sys.exit(0)
    
    # Run the main program
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)