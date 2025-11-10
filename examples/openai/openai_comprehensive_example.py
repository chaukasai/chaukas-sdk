"""
Comprehensive OpenAI Agents example demonstrating all supported events with real API calls.

This example showcases:
- SESSION_START/END lifecycle
- AGENT_START/END tracking
- AGENT_HANDOFF for multi-agent collaboration
- MODEL_INVOCATION_START/END
- TOOL_CALL_START/END detection
- MCP_CALL_START/END for Model Context Protocol integration
- INPUT_RECEIVED/OUTPUT_EMITTED
- ERROR and RETRY events
- POLICY_DECISION for content safety
- DATA_ACCESS for data retrieval tracking
- STATE_UPDATE for state management

Requires OPENAI_API_KEY environment variable to be set.
MCP scenario requires MCP server running (see examples/openai/mcp/prompt-server/).
"""

import asyncio
import json
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# Load .env from the same directory as this script
script_dir = Path(__file__).parent
load_dotenv(script_dir / ".env")

# Check for API key before proceeding
if not os.environ.get("OPENAI_API_KEY"):
    print("❌ Error: OPENAI_API_KEY environment variable is not set")
    print("Please set your OpenAI API key:")
    print("  export OPENAI_API_KEY='your-api-key-here'")
    sys.exit(1)

# Set up environment for Chaukas
os.environ["CHAUKAS_TENANT_ID"] = "demo_tenant"
os.environ["CHAUKAS_PROJECT_ID"] = "openai_comprehensive_demo"
os.environ["CHAUKAS_OUTPUT_MODE"] = "file"
os.environ["CHAUKAS_OUTPUT_FILE"] = "openai_comprehensive_output.jsonl"
os.environ["CHAUKAS_BATCH_SIZE"] = "1"  # Immediate write for demo

# Import Chaukas SDK
from chaukas import sdk as chaukas

# Import OpenAI Agents SDK
try:
    from agents import Agent, Runner, function_tool

    print("✅ OpenAI Agents SDK loaded successfully")
except ImportError as e:
    print("❌ Error: Failed to import OpenAI Agents SDK")
    print("Please install it with: pip install openai-agents")
    print(f"Error details: {e}")
    sys.exit(1)


# ============================================================================
# Tool Implementations
# ============================================================================


@function_tool
def web_search(query: str) -> str:
    """
    Search the web for information.

    Args:
        query: The search query string

    Returns:
        Search results as a string
    """
    # Simulate web search results
    results = [
        f"Recent article about {query} from TechCrunch",
        f"Wikipedia entry on {query}",
        f"Academic paper discussing {query}",
        f"Tutorial on {query} from official documentation",
    ]
    return f"Found {len(results)} results for '{query}':\n" + "\n".join(
        f"- {r}" for r in results
    )


@function_tool
def calculator(expression: str) -> str:
    """
    Perform mathematical calculations.

    Args:
        expression: Mathematical expression to evaluate

    Returns:
        Calculation result as a string
    """
    try:
        # In production, use a safer evaluation method
        # For demo purposes, we'll use simple eval with safety checks
        allowed_chars = "0123456789+-*/()., "
        if all(c in allowed_chars for c in expression):
            result = eval(expression)
            return f"Result: {expression} = {result}"
        else:
            return f"Error: Invalid characters in expression '{expression}'"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"


@function_tool
def weather_api(location: str) -> str:
    """
    Get weather information for a location.

    Args:
        location: City or location name

    Returns:
        Weather information as a string
    """
    # Simulate weather API response
    weather_conditions = ["Sunny", "Partly Cloudy", "Cloudy", "Light Rain", "Clear"]
    temp_f = random.randint(60, 85)
    temp_c = round((temp_f - 32) * 5 / 9)
    condition = random.choice(weather_conditions)

    return f"Weather in {location}: {condition}, {temp_f}°F ({temp_c}°C), Humidity: {random.randint(40, 70)}%"


@function_tool
def get_current_time(timezone: str = "UTC") -> str:
    """
    Get the current time in a specific timezone.

    Args:
        timezone: Timezone name (default: UTC)

    Returns:
        Current time as a string
    """
    from datetime import datetime
    from datetime import timezone as tz

    # For simplicity, just return UTC or local time
    if timezone.upper() == "UTC":
        current_time = datetime.now(tz.utc)
        return f"Current time in {timezone}: {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}"
    else:
        current_time = datetime.now()
        return f"Current local time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}"


@function_tool
def mcp_context_fetch(
    context_type: str, query: str, server_url: str = "mcp://context-server"
) -> str:
    """
    Fetch context from MCP (Model Context Protocol) server.
    This simulates an MCP tool for demonstration purposes.

    Args:
        context_type: Type of context to fetch
        query: Query for context retrieval
        server_url: MCP server URL

    Returns:
        Retrieved context as a string
    """
    # Simulate MCP context retrieval
    contexts = {
        "documentation": f"Documentation context for {query}: API reference, examples, and best practices",
        "code": f"Code context for {query}: Implementation details and source code",
        "data": f"Data context for {query}: Relevant datasets and statistics",
    }

    context_result = contexts.get(context_type, f"General context for {query}")
    return f"MCP Context from {server_url}:\n{context_result}"


# ============================================================================
# Agent Scenarios
# ============================================================================


async def research_assistant_scenario():
    """Scenario 1: Research Assistant with web search and tools."""
    print("\n" + "=" * 60)
    print("Scenario 1: Research Assistant")
    print("=" * 60)

    # Create research assistant agent
    agent = Agent(
        name="research_assistant",
        instructions="""You are a helpful research assistant that finds and summarizes information.
        Use the web_search tool to find information on topics.
        Use the mcp_context_fetch tool for technical documentation.
        Be concise but thorough in your responses.""",
        model="gpt-4o-mini",  # Using gpt-4o-mini for cost efficiency
        tools=[web_search, mcp_context_fetch, get_current_time],
    )

    queries = [
        "What are the latest developments in quantum computing?",
        "Find information about distributed systems best practices",
        "What time is it in UTC?",
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n📝 Query {i}: {query}")

        try:
            # Run the agent asynchronously
            result = await Runner.run(agent, query)

            # Display the final output
            if result and result.final_output:
                output = (
                    result.final_output[:200] + "..."
                    if len(result.final_output) > 200
                    else result.final_output
                )
                print(f"✅ Response: {output}")
            else:
                print("✅ Response received (no output)")

        except Exception as e:
            print(f"❌ Error: {e}")
            # Handle specific API errors
            if "rate_limit" in str(e).lower():
                print("  ⏳ Rate limit hit, waiting before retry...")
                await asyncio.sleep(2)
            elif "401" in str(e) or "invalid" in str(e).lower():
                print("  🔑 API key issue detected")
                break

        await asyncio.sleep(0.5)  # Small delay between queries


async def math_tutor_scenario():
    """Scenario 2: Math Tutor with calculator."""
    print("\n" + "=" * 60)
    print("Scenario 2: Math Tutor")
    print("=" * 60)

    # Create math tutor agent
    agent = Agent(
        name="math_tutor",
        instructions="""You are a friendly math tutor. Help students with calculations and explanations.
        Use the calculator tool for mathematical computations.
        Explain your reasoning step by step.""",
        model="gpt-3.5-turbo",  # More cost-effective model
        tools=[calculator],
    )

    problems = [
        "Calculate 42 * 17 + 238",
        "What is (15 + 8) * 3 - 10?",
        "If I have 5 apples and buy 3 more, then give away 2, how many do I have?",
    ]

    for i, problem in enumerate(problems, 1):
        print(f"\n📐 Problem {i}: {problem}")

        try:
            # Run the agent synchronously
            result = await Runner.run(agent, problem)

            # Display the solution
            if result and result.final_output:
                output = (
                    result.final_output[:200] + "..."
                    if len(result.final_output) > 200
                    else result.final_output
                )
                print(f"✅ Solution: {output}")
            else:
                print("✅ Solution calculated")

        except Exception as e:
            print(f"❌ Error: {e}")

        await asyncio.sleep(0.5)


async def travel_planner_scenario():
    """Scenario 3: Travel Planner with weather and search tools."""
    print("\n" + "=" * 60)
    print("Scenario 3: Travel Planner")
    print("=" * 60)

    # Create travel planner agent
    agent = Agent(
        name="travel_planner",
        instructions="""You are a travel planning assistant. Help users plan their trips.
        Use the weather_api to check weather conditions.
        Use web_search to find tourist attractions and travel information.
        Provide helpful and practical advice.""",
        model="gpt-4o-mini",
        tools=[weather_api, web_search, get_current_time],
    )

    # Multi-turn conversation simulation
    queries = [
        "I'm planning a trip to Paris. What's the weather like there?",
        "What are the top tourist attractions I should visit?",
        "What's the best time of year to visit?",
    ]

    print(f"🗺️ Travel planning conversation:")

    for i, query in enumerate(queries, 1):
        print(f"\n👤 User: {query}")

        # Implement retry logic for demonstration
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"  Attempt {attempt + 1}/{max_retries}...")

                # Run the agent
                result = await Runner.run(agent, query)

                if result and result.final_output:
                    output = (
                        result.final_output[:200] + "..."
                        if len(result.final_output) > 200
                        else result.final_output
                    )
                    print(f"  🤖 Assistant: {output}")

                break  # Success, exit retry loop

            except Exception as e:
                error_msg = str(e)
                print(f"  ⚠️  Error: {error_msg}")

                if attempt < max_retries - 1:
                    # Exponential backoff
                    wait_time = 2**attempt
                    print(f"  ⏳ Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"  ❌ Max retries exceeded")

        await asyncio.sleep(0.5)


async def error_handling_scenario():
    """Scenario 4: Error handling and recovery demonstration."""
    print("\n" + "=" * 60)
    print("Scenario 4: Error Handling and Recovery")
    print("=" * 60)

    # Create an agent for error testing
    agent = Agent(
        name="error_test_agent",
        instructions="""You are a test agent for demonstrating error handling.
        Try to use tools when asked.
        Handle errors gracefully.""",
        model="gpt-4o-mini",
        tools=[web_search, calculator],
    )

    test_cases = [
        "Search for information about error handling in distributed systems",
        "Calculate the result of 1/0",  # Will cause a calculation error
        "What happens when an API fails?",
    ]

    for i, test_query in enumerate(test_cases, 1):
        print(f"\n🧪 Test {i}: {test_query}")

        try:
            # First attempt
            result = await Runner.run(agent, test_query)

            if result and result.final_output:
                output = (
                    result.final_output[:150] + "..."
                    if len(result.final_output) > 150
                    else result.final_output
                )
                print(f"  ✅ Success: {output}")

        except Exception as e:
            print(f"  ❌ Error captured: {e}")

            # Retry once
            print(f"  ⏳ Retrying after error...")
            await asyncio.sleep(1)

            try:
                result = await Runner.run(agent, test_query)
                print(f"  ✅ Retry successful!")

            except Exception as e2:
                print(f"  ❌ Retry also failed: {e2}")

        await asyncio.sleep(0.5)


async def multi_agent_handoff_scenario():
    """Scenario 5: Multi-agent handoff demonstration using proper handoff mechanism."""
    print("\n" + "=" * 60)
    print("Scenario 5: Multi-Agent Handoff")
    print("=" * 60)

    from agents import handoff

    # Create specialized agents with proper handoff relationships
    analyst = Agent(
        name="Data Analyst",
        instructions="""You are a data analyst. Perform calculations and analyze numbers using the calculator tool.
        Provide clear, concise answers with the results.""",
        model="gpt-4o-mini",
        tools=[calculator],
        handoff_description="Specialist for calculations and data analysis",
    )

    researcher = Agent(
        name="Research Specialist",
        instructions="""You are a research specialist. Find information using web_search.
        Provide well-researched, accurate information.""",
        model="gpt-4o-mini",
        tools=[web_search],
        handoff_description="Specialist for research and finding information",
    )

    # Main coordinator agent with handoff capabilities
    coordinator = Agent(
        name="Coordinator",
        instructions="""You are a helpful coordinator that delegates tasks to specialists.
        - For calculations and numerical analysis, hand off to the Data Analyst
        - For research and finding information, hand off to the Research Specialist
        Always hand off when appropriate. Be concise.""",
        model="gpt-4o-mini",
        handoffs=[handoff(analyst), handoff(researcher)],
    )

    # Test queries that should trigger handoffs
    queries = [
        "What is the average of these numbers: 45, 67, 89, 23, 56?",
        "Find information about machine learning frameworks",
        "Calculate the sum of 123 + 456 + 789",
    ]

    print(f"🤝 Testing agent handoffs with {len(queries)} queries:")

    for i, query in enumerate(queries, 1):
        print(f"\n👤 Query {i}: {query}")

        try:
            result = await Runner.run(coordinator, query)

            if result and result.final_output:
                output = (
                    result.final_output[:200] + "..."
                    if len(result.final_output) > 200
                    else result.final_output
                )
                print(f"  🤖 Response: {output}")

        except Exception as e:
            print(f"  ❌ Error: {e}")

        await asyncio.sleep(0.5)


async def policy_decision_scenario():
    """Scenario 6: Policy Decision demonstration with content filtering."""
    print("\n" + "=" * 60)
    print("Scenario 6: Policy Decision & Content Safety")
    print("=" * 60)

    from chaukas.sdk.core.event_builder import EventBuilder

    # Create a content moderation agent
    agent = Agent(
        name="content_moderator",
        instructions="""You are a content moderation assistant.
        Review user input for safety and appropriateness.
        Respond with whether the content is safe or needs review.""",
        model="gpt-4o-mini",
    )

    # Test content samples
    test_samples = [
        "Hello, how are you today?",
        "Tell me about Python programming best practices",
        "What's the weather like in New York?",
    ]

    print("🛡️  Running content through policy checks:")

    builder = EventBuilder()
    client = chaukas.get_client()

    for i, content in enumerate(test_samples, 1):
        print(f"\n📝 Sample {i}: {content[:50]}...")

        try:
            # Simulate policy decision (in production, this would be a real policy engine)
            policy_checks = {
                "hate_speech": "PASS",
                "violence": "PASS",
                "pii_detection": "PASS",
                "content_safety": "PASS",
            }

            # Emit policy decision event
            policy_event = builder.create_policy_decision(
                policy_id="content_safety_v1",
                outcome="ALLOWED",
                rule_ids=[
                    "rule_hate_speech",
                    "rule_violence",
                    "rule_pii",
                    "rule_safety",
                ],
                rationale=f"Content passed all safety checks: {', '.join(f'{k}={v}' for k, v in policy_checks.items())}",
            )

            if client:
                await client.send_event(policy_event)

            print(f"  ✅ Policy: ALLOWED - All checks passed")

            # Run the agent with approved content
            result = await Runner.run(agent, content)

            if result and result.final_output:
                output = (
                    result.final_output[:100] + "..."
                    if len(result.final_output) > 100
                    else result.final_output
                )
                print(f"  🤖 Response: {output}")

        except Exception as e:
            print(f"  ❌ Error: {e}")

        await asyncio.sleep(0.3)


async def data_access_scenario():
    """Scenario 7: Data Access demonstration with database and document retrieval."""
    print("\n" + "=" * 60)
    print("Scenario 7: Data Access & Retrieval Tracking")
    print("=" * 60)

    from chaukas.sdk.core.event_builder import EventBuilder

    # Create a data analyst agent
    agent = Agent(
        name="data_analyst",
        instructions="""You are a data analyst that helps users find and analyze information.
        When asked about data, explain what you would retrieve and analyze.""",
        model="gpt-4o-mini",
    )

    # Simulated data queries
    data_queries = [
        {
            "query": "Show me user analytics for Q4 2024",
            "datasource": "analytics_db",
            "documents": ["analytics_2024_q4", "user_metrics_dec"],
            "pii": ["user_id", "email"],
        },
        {
            "query": "Find customer feedback from last month",
            "datasource": "feedback_db",
            "documents": ["feedback_jan_2025", "reviews_jan_2025"],
            "pii": ["customer_name", "email"],
        },
        {
            "query": "Retrieve sales data for products",
            "datasource": "sales_db",
            "documents": ["sales_2025", "product_metrics"],
            "pii": [],
        },
    ]

    print("📊 Tracking data access across multiple sources:")

    builder = EventBuilder()
    client = chaukas.get_client()

    for i, data_query in enumerate(data_queries, 1):
        print(f"\n🔍 Query {i}: {data_query['query']}")

        try:
            # Emit data access event to track what data is being accessed
            data_access_event = builder.create_data_access(
                datasource=data_query["datasource"],
                document_ids=data_query["documents"],
                chunk_ids=[],
                pii_categories=data_query["pii"],
            )

            if client:
                await client.send_event(data_access_event)

            print(f"  📁 Data Source: {data_query['datasource']}")
            print(f"  📄 Documents: {', '.join(data_query['documents'])}")

            if data_query["pii"]:
                print(f"  🔒 PII Detected: {', '.join(data_query['pii'])}")
            else:
                print(f"  ✅ No PII in query")

            # Simulate agent processing
            result = await Runner.run(
                agent, f"Analyze this data query: {data_query['query']}"
            )

            if result and result.final_output:
                output = (
                    result.final_output[:120] + "..."
                    if len(result.final_output) > 120
                    else result.final_output
                )
                print(f"  🤖 Analysis: {output}")

        except Exception as e:
            print(f"  ❌ Error: {e}")

        await asyncio.sleep(0.3)


async def mcp_integration_scenario():
    """Scenario 8: MCP (Model Context Protocol) Integration."""
    print("\n" + "=" * 60)
    print("Scenario 8: MCP Integration")
    print("=" * 60)

    try:
        from agents.mcp import MCPServerStreamableHttp
    except ImportError:
        print(
            "❌ MCP support not available. Install with: pip install 'openai-agents[mcp]'"
        )
        print("\nℹ️  This scenario demonstrates MCP_CALL_START/END events")
        print("   See examples/openai/mcp/prompt-server/ for a working MCP example")
        return

    print("🔌 Connecting to MCP server...")
    print("ℹ️  Note: This requires the MCP server to be running")
    print("   Start it with: python examples/openai/mcp/prompt-server/server.py")
    print()

    try:
        # Try to connect to MCP server
        async with MCPServerStreamableHttp(
            name="Prompt Server",
            params={"url": "http://localhost:8000/mcp"},
        ) as mcp_server:
            print("✅ Connected to MCP server")

            # List available prompts
            print("\n📋 Listing available prompts from MCP server...")
            prompts_result = await mcp_server.list_prompts()

            print(f"   Found {len(prompts_result.prompts)} prompt(s):")
            for i, prompt in enumerate(prompts_result.prompts, 1):
                print(f"   {i}. {prompt.name}")
                if prompt.description:
                    print(f"      {prompt.description}")

            # Get a prompt and use it
            if prompts_result.prompts:
                prompt_name = prompts_result.prompts[0].name
                print(f"\n🎯 Getting prompt: {prompt_name}")

                # Get prompt from MCP server
                prompt_result = await mcp_server.get_prompt(
                    prompt_name, {"focus": "code quality", "language": "python"}
                )

                # Extract instructions from prompt
                content = prompt_result.messages[0].content
                if hasattr(content, "text"):
                    instructions = content.text
                else:
                    instructions = str(content)

                print(f"   ✅ Retrieved instructions ({len(instructions)} chars)")

                # Create agent with MCP-provided instructions
                agent = Agent(
                    name="MCP-Powered Agent",
                    instructions=instructions[:500] + "...",  # Truncate for display
                    model="gpt-4o-mini",
                )

                # Run the agent
                print("\n🤖 Running agent with MCP-provided instructions...")
                test_code = """def calculate(x, y):
    result = x + y
    return result"""

                result = await Runner.run(agent, f"Review this code:\n{test_code}")

                if result and result.final_output:
                    output = (
                        result.final_output[:200] + "..."
                        if len(result.final_output) > 200
                        else result.final_output
                    )
                    print(f"   🤖 Response: {output}")

                print("\n✅ MCP integration complete!")
                print("   MCP_CALL_START and MCP_CALL_END events should be captured")

            else:
                print("   ⚠️  No prompts available on MCP server")

    except ConnectionError as e:
        print(f"\n❌ Could not connect to MCP server: {e}")
        print("\nTo run this scenario:")
        print("1. In a separate terminal, run:")
        print("   cd examples/openai/mcp/prompt-server")
        print("   python server.py")
        print("2. Then run this scenario again")

    except Exception as e:
        print(f"\n❌ MCP Error: {e}")
        print(
            "\nℹ️  See examples/openai/mcp/prompt-server/main.py for a complete MCP example"
        )


# ============================================================================
# Event Analysis
# ============================================================================


def analyze_events(filename: str):
    """Analyze captured events from JSONL file."""
    print("\n" + "=" * 60)
    print("Event Analysis")
    print("=" * 60)

    try:
        with open(filename, "r") as f:
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
            event_type = event.get("type", "UNKNOWN")
            event_counts[event_type] += 1

            if event_type == "EVENT_TYPE_RETRY":
                retry_events.append(event)
            elif event_type == "EVENT_TYPE_ERROR":
                error_events.append(event)
            elif event_type == "EVENT_TYPE_TOOL_CALL_START":
                tool_calls.append(event)

        # Display summary
        print(f"\n📊 Total Events Captured: {len(events)}")
        print("\n📈 Event Distribution:")

        # Sort by count
        sorted_events = sorted(event_counts.items(), key=lambda x: x[1], reverse=True)
        for event_type, count in sorted_events:
            event_name = event_type.replace("EVENT_TYPE_", "")
            print(f"  {event_name:25} : {count:3} events")

        # Show retry details
        if retry_events:
            print(f"\n🔄 Retry Events ({len(retry_events)} total):")
            for retry in retry_events[:3]:  # Show first 3
                if "metadata" in retry:
                    meta = retry["metadata"]
                    print(
                        f"  - Attempt {meta.get('attempt', 'N/A')}: {meta.get('retry_reason', 'N/A')[:50]}..."
                    )

        # Show error details
        if error_events:
            print(f"\n❌ Error Events ({len(error_events)} total):")
            for error in error_events[:3]:  # Show first 3
                if "error" in error:
                    err = error["error"]
                    print(
                        f"  - {err.get('error_code', 'N/A')}: {err.get('error_message', 'N/A')[:50]}..."
                    )

        # Show tool calls
        if tool_calls:
            print(f"\n🔧 Tool Calls ({len(tool_calls)} total):")
            tool_names = defaultdict(int)
            for tc in tool_calls:
                if "tool_call" in tc:
                    tool_names[tc["tool_call"].get("tool_name", "unknown")] += 1

            for tool_name, count in tool_names.items():
                print(f"  - {tool_name}: {count} calls")

        # Check event pairing
        print("\n🔗 Event Pairing Check:")
        start_types = [
            "SESSION_START",
            "AGENT_START",
            "MODEL_INVOCATION_START",
            "TOOL_CALL_START",
        ]

        for event_base in start_types:
            start_count = event_counts.get(f"EVENT_TYPE_{event_base}", 0)
            end_count = event_counts.get(
                f'EVENT_TYPE_{event_base.replace("START", "END")}', 0
            )

            if start_count > 0 or end_count > 0:
                status = "✅" if start_count == end_count else "⚠️"
                print(
                    f"  {status} {event_base.replace('_START', ''):20} : {start_count} START, {end_count} END"
                )

        # Coverage report
        print("\n📋 Event Coverage Report:")
        supported_events = [
            "SESSION_START",
            "SESSION_END",
            "AGENT_START",
            "AGENT_END",
            "AGENT_HANDOFF",
            "MODEL_INVOCATION_START",
            "MODEL_INVOCATION_END",
            "TOOL_CALL_START",
            "TOOL_CALL_END",
            "MCP_CALL_START",
            "MCP_CALL_END",
            "INPUT_RECEIVED",
            "OUTPUT_EMITTED",
            "ERROR",
            "RETRY",
            "POLICY_DECISION",
            "DATA_ACCESS",
            "STATE_UPDATE",
            "SYSTEM",
        ]

        captured = 0
        for event in supported_events:
            if event_counts.get(f"EVENT_TYPE_{event}", 0) > 0:
                captured += 1
                print(f"  ✅ {event}")
            else:
                print(f"  ❌ {event} (not captured in this run)")

        print(
            f"\n📊 Coverage: {captured}/{len(supported_events)} supported events captured"
        )
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
    print("=" * 60)
    print("OpenAI Agents Comprehensive Example")
    print("Using Real API with Enhanced Event Capture")
    print("=" * 60)

    # Verify API key
    api_key = os.environ.get("OPENAI_API_KEY")
    print(f"✅ OpenAI API key detected: {api_key[:10]}...")

    # Enable Chaukas instrumentation
    chaukas.enable_chaukas()
    print("✅ Chaukas instrumentation enabled")

    while True:
        print("\n" + "=" * 60)
        print("Select a scenario to run:")
        print("=" * 60)
        print("1. Research Assistant (web search & tools)")
        print("2. Math Tutor (calculator)")
        print("3. Travel Planner (weather & retries)")
        print("4. Error Handling Demo")
        print("5. Multi-Agent Handoff")
        print("6. Policy Decision & Content Safety")
        print("7. Data Access & Retrieval Tracking")
        print("8. MCP Integration (requires MCP server)")
        print("9. Run All Scenarios")
        print("A. Analyze Captured Events")
        print("0. Exit")
        print("-" * 60)

        try:
            choice = input("Enter your choice (0-9, A): ").strip().upper()

            if choice == "0":
                break
            elif choice == "1":
                await research_assistant_scenario()
            elif choice == "2":
                await math_tutor_scenario()
            elif choice == "3":
                await travel_planner_scenario()
            elif choice == "4":
                await error_handling_scenario()
            elif choice == "5":
                await multi_agent_handoff_scenario()
            elif choice == "6":
                await policy_decision_scenario()
            elif choice == "7":
                await data_access_scenario()
            elif choice == "8":
                await mcp_integration_scenario()
            elif choice == "9":
                print("\n🚀 Running all scenarios...")
                await research_assistant_scenario()
                await math_tutor_scenario()
                await travel_planner_scenario()
                await error_handling_scenario()
                await multi_agent_handoff_scenario()
                await policy_decision_scenario()
                await data_access_scenario()
                print("\nℹ️  Skipping MCP scenario (requires separate server)")
                print("   Run scenario 8 separately if MCP server is available")
            elif choice == "A":
                analyze_events("openai_comprehensive_output.jsonl")
            else:
                print("❌ Invalid choice, please try again")

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback

            traceback.print_exc()

    # Cleanup
    print("\n" + "=" * 60)
    print("Shutting down...")

    # Disable Chaukas (this will close sessions and flush events)
    chaukas.disable_chaukas()
    print("✅ Chaukas disabled, sessions closed, and events flushed")

    # Final analysis
    print("\n" + "=" * 60)
    print("Final Event Analysis")
    analyze_events("openai_comprehensive_output.jsonl")

    print("\n✅ Example completed successfully!")
    print("📄 Events saved to: openai_comprehensive_output.jsonl")


if __name__ == "__main__":
    # Parse command line arguments
    if "--help" in sys.argv:
        print(
            """
Usage: python openai_comprehensive_example.py [options]

Options:
  --help           Show this help message

Environment Variables:
  OPENAI_API_KEY   Your OpenAI API key (REQUIRED)

The example uses the real OpenAI API to demonstrate:
- Agent creation with tools
- Asynchronous execution with Runner.run()
- Tool calling and execution
- Error handling and retries
- Multi-agent scenarios
- Comprehensive event capture with Chaukas SDK

Make sure to set your OPENAI_API_KEY before running:
  export OPENAI_API_KEY='your-api-key-here'
        """
        )
        sys.exit(0)

    # Run the main program
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
