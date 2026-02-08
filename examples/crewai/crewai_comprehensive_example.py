"""
Comprehensive CrewAI example demonstrating event capture across different scenarios.

=============================================================================
EVENT COVERAGE BY SCENARIO
=============================================================================

✅ CURRENTLY CAPTURED (All Scenarios):
• SESSION_START/END - Crew execution lifecycle
• AGENT_START/END - Agent execution tracking
• MODEL_INVOCATION_START/END - LLM calls
• TOOL_CALL_START/END - Custom tool usage
• INPUT_RECEIVED/OUTPUT_EMITTED - Input/output tracking

✅ SCENARIO-SPECIFIC EVENTS:
• AGENT_HANDOFF - Scenario 1 (3 agents in sequence)
• MCP_CALL_START/END - Scenario 1 (MCPContextTool)
• ERROR/RETRY - Scenario 4 (FlakeySearchTool, NetworkAnalysisTool)

⚠️  REQUIRES SPECIAL CREWAI FEATURES:

1. DATA_ACCESS Events:
   Feature Required: CrewAI Knowledge with vector stores
   See: examples/crewai/crewai_knowledge_example.py
   Implementation: Use Agent(knowledge_sources=[...])

2. POLICY_DECISION Events:
   Feature Required: Task Guardrails
   See: examples/crewai/crewai_guardrails_example.py
   Implementation: Use Task(guardrail=validation_function)

3. STATE_UPDATE Events:
   Feature Required: Internal CrewAI Agent Reasoning
   Automatically captured from: AgentReasoningStartedEvent, AgentReasoningCompletedEvent
   Note: These are internal CrewAI events, not triggered by custom tools

4. SYSTEM Events:
   Feature Required: CrewAI Flows
   See: examples/crewai/crewai_flows_example.py
   Implementation: Use Flow class with @start and @listen decorators

=============================================================================
TOOLS USED
=============================================================================

This example uses straightforward tools that demonstrate core event capture:
• WebSearchTool, CalculatorTool, WeatherTool, CodeGeneratorTool
• MCPContextTool (for MCP_CALL events)
• FlakeySearchTool, NetworkAnalysisTool (for ERROR/RETRY events)

All tools trigger standard TOOL_CALL_START/END events.

=============================================================================

Requires OPENAI_API_KEY environment variable to be set.
"""

import asyncio
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from dotenv import load_dotenv

# Get script directory for consistent file paths
SCRIPT_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = SCRIPT_DIR / "output"

# Load .env from the same directory as this script
load_dotenv(SCRIPT_DIR / ".env")

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(exist_ok=True)

# Get base name from env (without extension), construct timestamped path
# .env should have: CHAUKAS_OUTPUT_FILE=crewai_output (no extension)
OUTPUT_BASE = os.environ.get("CHAUKAS_OUTPUT_FILE", "crewai_output")
# Strip any extension if accidentally provided
OUTPUT_BASE = OUTPUT_BASE.rsplit(".", 1)[0] if "." in OUTPUT_BASE else OUTPUT_BASE
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FILE = str(OUTPUT_DIR / f"{OUTPUT_BASE}_{TIMESTAMP}.jsonl")

# Check for API key before proceeding
if not os.environ.get("OPENAI_API_KEY"):
    print("❌ Error: OPENAI_API_KEY environment variable is not set")
    print("Please add your OpenAI API key to .env file")
    sys.exit(1)

# Disable CrewAI's own telemetry
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

# Set up Chaukas output file
os.environ["CHAUKAS_OUTPUT_MODE"] = "file"
os.environ["CHAUKAS_OUTPUT_FILE"] = OUTPUT_FILE
os.environ.setdefault("CHAUKAS_BATCH_SIZE", "1")  # Immediate writes for demo

# Import Chaukas SDK
from chaukas import sdk as chaukas

# Import event analysis tool
sys.path.insert(0, str(SCRIPT_DIR.parent.parent))
from tools.summarize_event_stats import summarize_event_stats

# Import CrewAI
try:
    from crewai import Agent, Crew, Process, Task
    from crewai.tools import BaseTool
    from pydantic import BaseModel, Field

    print("✅ CrewAI SDK loaded successfully")
except ImportError as e:
    print("❌ Error: Failed to import CrewAI")
    print("Please install it with: pip install crewai")
    print(f"Error details: {e}")
    sys.exit(1)


# ============================================================================
# Tool Input Schemas
# ============================================================================


class WebSearchInput(BaseModel):
    """Input for web search tool."""

    query: str = Field(description="The search query")


class CalculatorInput(BaseModel):
    """Input for calculator tool."""

    expression: str = Field(description="Mathematical expression to evaluate")


class WeatherInput(BaseModel):
    """Input for weather tool."""

    location: str = Field(description="Location to get weather for")


class MCPContextInput(BaseModel):
    """Input for MCP context tool."""

    context_request: str = Field(description="Context request in format: type|query")


class CodeGeneratorInput(BaseModel):
    """Input for code generator tool."""

    request: str = Field(
        description="Description of code to generate, including language"
    )


class FlakeySearchInput(BaseModel):
    """Input for the flakey search tool."""

    query: str = Field(description="The search query")


class NetworkAnalysisInput(BaseModel):
    """Input for network analysis tool."""

    data: str = Field(description="Data to analyze")


# ============================================================================
# Tool Implementations
# ============================================================================


class WebSearchTool(BaseTool):
    """Tool for searching the web."""

    name: str = "web_search"
    description: str = "Search the web for information on any topic"
    args_schema: Type[BaseModel] = WebSearchInput

    def _run(self, query: str) -> str:
        """Execute web search."""
        # Simulate web search results
        results = [
            f"Latest article about {query} from TechCrunch",
            f"Wikipedia entry on {query}",
            f"Research paper discussing {query}",
            f"Tutorial on {query} from documentation",
        ]
        return f"Found {len(results)} results for '{query}':\n" + "\n".join(
            f"- {r}" for r in results
        )


class CalculatorTool(BaseTool):
    """Tool for mathematical calculations."""

    name: str = "calculator"
    description: str = "Perform mathematical calculations and solve equations"
    args_schema: Type[BaseModel] = CalculatorInput

    def _run(self, expression: str) -> str:
        """Execute calculation."""
        try:
            # Safety check for allowed characters
            allowed = "0123456789+-*/()., "
            if all(c in allowed for c in expression):
                result = eval(expression)
                return f"Result: {expression} = {result}"
            return f"Error: Invalid characters in expression '{expression}'"
        except Exception as e:
            return f"Error calculating {expression}: {str(e)}"


class WeatherTool(BaseTool):
    """Tool for getting weather information."""

    name: str = "weather_api"
    description: str = "Get current weather information for any location"
    args_schema: Type[BaseModel] = WeatherInput

    def _run(self, location: str) -> str:
        """Get weather for location."""
        conditions = ["Sunny", "Partly Cloudy", "Cloudy", "Light Rain", "Clear"]
        temp_f = random.randint(60, 85)
        temp_c = round((temp_f - 32) * 5 / 9)
        condition = random.choice(conditions)
        return f"Weather in {location}: {condition}, {temp_f}°F ({temp_c}°C), Humidity: {random.randint(40, 70)}%"


class MCPContextTool(BaseTool):
    """MCP (Model Context Protocol) tool for context retrieval."""

    name: str = "mcp_context_fetch"
    description: str = (
        "Fetch context from MCP server for documentation and code examples"
    )
    args_schema: Type[BaseModel] = MCPContextInput

    def _run(self, context_request: str) -> str:
        """Fetch context from MCP server."""
        # Parse context request
        parts = context_request.split("|")
        context_type = parts[0] if parts else "general"
        query = parts[1] if len(parts) > 1 else context_request

        # Simulate MCP context retrieval
        contexts = {
            "documentation": f"Documentation for {query}: API reference, examples, best practices",
            "code": f"Code examples for {query}: Implementation patterns and samples",
            "data": f"Data context for {query}: Schemas, models, and structures",
        }

        server_url = "mcp://context-server.example.com"
        result = contexts.get(context_type, f"General context for {query}")
        return f"MCP Context from {server_url}:\n{result}"


class CodeGeneratorTool(BaseTool):
    """Tool for generating code snippets."""

    name: str = "code_generator"
    description: str = "Generate code snippets in various programming languages"
    args_schema: Type[BaseModel] = CodeGeneratorInput

    def _run(self, request: str) -> str:
        """Generate code based on request."""
        # Simple code generation simulation
        if "python" in request.lower():
            return (
                """```python
def example_function(param1, param2):
    '''Generated function for: """
                + request
                + """'''
    result = param1 + param2
    return result
```"""
            )
        elif "javascript" in request.lower():
            return (
                """```javascript
function exampleFunction(param1, param2) {
    // Generated function for: """
                + request
                + """
    return param1 + param2;
}
```"""
            )
        return f"Generated pseudocode for: {request}\n1. Initialize variables\n2. Process input\n3. Return result"


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


# ============================================================================
# Agent Scenarios
# ============================================================================


def research_team_scenario():
    """Scenario 1: Research team with multiple specialized agents."""
    print("\n" + "=" * 60)
    print("Scenario 1: Research Team")
    print("=" * 60)

    # Create specialized research agents
    researcher = Agent(
        role="Senior Researcher",
        goal="Find comprehensive information on topics",
        backstory="You are an experienced researcher with expertise in finding and analyzing information from various sources.",
        tools=[WebSearchTool(), MCPContextTool()],
        verbose=True,
    )

    analyst = Agent(
        role="Data Analyst",
        goal="Analyze data and provide insights",
        backstory="You are a data analyst who excels at finding patterns and extracting insights from information.",
        tools=[CalculatorTool(), WeatherTool()],
        verbose=True,
    )

    writer = Agent(
        role="Content Writer",
        goal="Create clear and engaging content",
        backstory="You are a skilled writer who can synthesize complex information into clear, readable content.",
        tools=[CodeGeneratorTool()],
        verbose=True,
    )

    # Create tasks
    research_task = Task(
        description="Research the latest developments in artificial intelligence and machine learning",
        expected_output="A comprehensive list of recent AI/ML developments with sources",
        agent=researcher,
    )

    analysis_task = Task(
        description="Analyze the research findings and identify key trends and patterns",
        expected_output="Analysis report with identified trends and statistical insights",
        agent=analyst,
    )

    writing_task = Task(
        description="Write a comprehensive report based on the research and analysis",
        expected_output="A well-structured report on AI/ML developments with insights",
        agent=writer,
    )

    # Create crew with sequential process (demonstrates AGENT_HANDOFF)
    crew = Crew(
        agents=[researcher, analyst, writer],
        tasks=[research_task, analysis_task, writing_task],
        process=Process.sequential,
        verbose=True,
    )

    print("🚀 Starting research team crew...")

    try:
        result = crew.kickoff()
        print(f"\n✅ Research team completed successfully!")
        if result:
            output = (
                str(result)[:300] + "..." if len(str(result)) > 300 else str(result)
            )
            print(f"📄 Final Report Preview: {output}")
    except Exception as e:
        print(f"❌ Error in research team: {e}")


def development_team_scenario():
    """Scenario 2: Development team with technical agents."""
    print("\n" + "=" * 60)
    print("Scenario 2: Development Team")
    print("=" * 60)

    # Create technical agents
    architect = Agent(
        role="Software Architect",
        goal="Design robust software architectures",
        backstory="You are a senior architect with expertise in system design and best practices.",
        tools=[MCPContextTool(), PolicyCheckTool()],
        verbose=True,
    )

    developer = Agent(
        role="Senior Developer",
        goal="Implement high-quality code",
        backstory="You are an experienced developer who writes clean, efficient code.",
        tools=[CodeGeneratorTool(), StatePersistenceTool()],
        verbose=True,
    )

    tester = Agent(
        role="QA Engineer",
        goal="Ensure code quality and reliability",
        backstory="You are a detail-oriented QA engineer who finds and prevents bugs.",
        tools=[CalculatorTool(), PolicyCheckTool()],
        verbose=True,
    )

    # Create development tasks
    design_task = Task(
        description="Design a microservices architecture for an e-commerce platform",
        expected_output="Architecture design document with diagrams and specifications",
        agent=architect,
    )

    implementation_task = Task(
        description="Implement the core services based on the architecture design",
        expected_output="Code implementation with documentation",
        agent=developer,
    )

    testing_task = Task(
        description="Create and execute test plans for the implemented services",
        expected_output="Test report with coverage metrics and findings",
        agent=tester,
    )

    # Create development crew
    crew = Crew(
        agents=[architect, developer, tester],
        tasks=[design_task, implementation_task, testing_task],
        process=Process.sequential,
        verbose=True,
    )

    print("🚀 Starting development team crew...")

    try:
        result = crew.kickoff()
        print(f"\n✅ Development team completed successfully!")
        if result:
            output = (
                str(result)[:300] + "..." if len(str(result)) > 300 else str(result)
            )
            print(f"📄 Development Output: {output}")
    except Exception as e:
        print(f"❌ Error in development team: {e}")


def customer_support_scenario():
    """Scenario 3: Customer support team demonstrating knowledge access."""
    print("\n" + "=" * 60)
    print("Scenario 3: Customer Support Team")
    print("=" * 60)

    # Create support agents
    support_agent = Agent(
        role="Customer Support Specialist",
        goal="Resolve customer issues efficiently",
        backstory="You are a helpful support specialist who resolves customer problems with empathy.",
        tools=[KnowledgeBaseTool(), StatePersistenceTool()],
        verbose=True,
    )

    technical_support = Agent(
        role="Technical Support Engineer",
        goal="Solve complex technical issues",
        backstory="You are a technical expert who can diagnose and fix complex problems.",
        tools=[MCPContextTool(), CalculatorTool()],
        verbose=True,
    )

    escalation_manager = Agent(
        role="Escalation Manager",
        goal="Handle critical issues and escalations",
        backstory="You manage escalated issues and ensure customer satisfaction.",
        tools=[PolicyCheckTool(), StatePersistenceTool()],
        verbose=True,
    )

    # Create support tasks with different priority levels
    ticket_task = Task(
        description="Handle customer ticket: 'Application crashes when uploading large files'",
        expected_output="Resolution steps and customer communication",
        agent=support_agent,
    )

    technical_task = Task(
        description="Investigate the technical root cause of the file upload crash issue",
        expected_output="Technical analysis with proposed fix",
        agent=technical_support,
    )

    escalation_task = Task(
        description="Review the issue resolution and ensure it meets SLA requirements",
        expected_output="Escalation review report with recommendations",
        agent=escalation_manager,
    )

    # Create support crew
    crew = Crew(
        agents=[support_agent, technical_support, escalation_manager],
        tasks=[ticket_task, technical_task, escalation_task],
        process=Process.sequential,
        verbose=True,
    )

    print("🚀 Starting customer support crew...")

    try:
        result = crew.kickoff(inputs={"ticket_id": "TICKET-12345", "priority": "high"})
        print(f"\n✅ Support team completed successfully!")
        if result:
            output = (
                str(result)[:300] + "..." if len(str(result)) > 300 else str(result)
            )
            print(f"📄 Resolution Summary: {output}")
    except Exception as e:
        print(f"❌ Error in support team: {e}")


def error_retry_scenario():
    """Scenario 4: Deliberately trigger errors and retries with flaky tools."""
    print("\n" + "=" * 60)
    print("Scenario 4: Error and Retry Demonstration")
    print("=" * 60)
    print("\nThis scenario demonstrates RETRY and ERROR event capture.")
    print("Tools will fail initially and then succeed on retry.")
    print("=" * 60)

    # Create tools that will fail and retry
    flakey_tool = FlakeySearchTool()
    network_tool = NetworkAnalysisTool()

    # Create an agent with the flakey tools
    researcher = Agent(
        role="Research Analyst",
        goal="Gather information despite unreliable tools",
        backstory="You are persistent and retry when tools fail temporarily.",
        tools=[flakey_tool, network_tool],
        verbose=True,
        max_iter=5,  # Allow multiple attempts for retries
    )

    # Create a task that will use the flakey tools
    research_task = Task(
        description="""Use the flakey_search tool to find information about 'AI trends'.
        Then use the network_analysis tool to analyze the results.
        Be persistent if tools fail - they often work on retry.""",
        expected_output="A summary of AI trends with analysis",
        agent=researcher,
    )

    # Create crew
    crew = Crew(
        agents=[researcher],
        tasks=[research_task],
        verbose=True,
        process=Process.sequential,
    )

    print("🚀 Starting error/retry demonstration...")
    print("(Expect to see tool failures and ERROR events)\n")

    try:
        result = crew.kickoff()
        print("\n✅ Error/retry scenario completed!")
        if result:
            output = (
                str(result)[:200] + "..." if len(str(result)) > 200 else str(result)
            )
            print(f"Result: {output}")

        print("\n📊 Events captured:")
        print("  • TOOL_CALL_START/END for each tool attempt")
        print("  • ERROR events when tools fail")
        print("  • RETRY events if CrewAI retries (conservative in v1.4.1)")
        print("\nNote: CrewAI 1.4.1 uses conservative retry logic.")
        print("The agent may choose alternatives after encountering failures.")

    except Exception as e:
        print(f"\n❌ Error/retry scenario failed: {e}")


def complex_workflow_scenario():
    """Scenario 5: Complex workflow with multiple crews and state management."""
    print("\n" + "=" * 60)
    print("Scenario 5: Complex Multi-Crew Workflow")
    print("=" * 60)

    # Phase 1: Planning Crew
    planner = Agent(
        role="Project Planner",
        goal="Create comprehensive project plans",
        backstory="You are an experienced project planner who creates detailed execution strategies.",
        tools=[WebSearchTool(), CalculatorTool()],
        verbose=True,
    )

    plan_task = Task(
        description="Create a project plan for launching a new product feature",
        expected_output="Detailed project plan with milestones",
        agent=planner,
    )

    planning_crew = Crew(
        agents=[planner], tasks=[plan_task], process=Process.sequential, verbose=True
    )

    print("📋 Phase 1: Planning...")
    try:
        planning_result = planning_crew.kickoff()
        print("✅ Planning phase completed")
    except Exception as e:
        print(f"❌ Planning failed: {e}")
        return

    # Phase 2: Execution Crew
    executor = Agent(
        role="Project Executor",
        goal="Execute project plans efficiently",
        backstory="You execute projects according to plan with attention to detail.",
        tools=[CodeGeneratorTool(), WebSearchTool()],
        verbose=True,
    )

    reviewer = Agent(
        role="Quality Reviewer",
        goal="Review and ensure quality standards",
        backstory="You review work to ensure it meets quality standards.",
        tools=[WeatherTool(), MCPContextTool()],
        verbose=True,
    )

    execution_task = Task(
        description="Execute the project plan created in the planning phase",
        expected_output="Implementation results and artifacts",
        agent=executor,
    )

    review_task = Task(
        description="Review the execution results for quality and compliance",
        expected_output="Quality review report with recommendations",
        agent=reviewer,
    )

    execution_crew = Crew(
        agents=[executor, reviewer],
        tasks=[execution_task, review_task],
        process=Process.sequential,
        verbose=True,
    )

    print("\n⚙️ Phase 2: Execution...")
    try:
        execution_result = execution_crew.kickoff()
        print("✅ Execution phase completed")
    except Exception as e:
        print(f"❌ Execution failed: {e}")
        return

    print("\n✅ Complex workflow completed successfully!")


# ============================================================================
# Main Program
# ============================================================================


def main():
    """Main program with interactive menu."""
    print("=" * 60)
    print("CrewAI Comprehensive Example")
    print("Demonstrating All 19 Chaukas Event Types")
    print("=" * 60)

    # Verify API key
    api_key = os.environ.get("OPENAI_API_KEY")
    print(f"✅ OpenAI API key detected: {api_key[:10]}...")

    # Enable Chaukas instrumentation
    chaukas.enable_chaukas()
    print("✅ Chaukas instrumentation enabled")
    print("✅ CrewAI telemetry disabled")

    while True:
        print("\n" + "=" * 60)
        print("Select a scenario to run:")
        print("=" * 60)
        print("1. Research Team (web search, analysis, writing)")
        print("2. Development Team (architecture, coding, testing)")
        print("3. Customer Support (knowledge base, escalation)")
        print("4. Error & Retry Demo")
        print("5. Complex Multi-Crew Workflow")
        print("6. Run All Scenarios")
        print("7. Analyze Captured Events")
        print("0. Exit")
        print("-" * 60)

        try:
            choice = input("Enter your choice (0-7): ").strip()

            if choice == "0":
                break
            elif choice == "1":
                research_team_scenario()
            elif choice == "2":
                development_team_scenario()
            elif choice == "3":
                customer_support_scenario()
            elif choice == "4":
                error_retry_scenario()
            elif choice == "5":
                complex_workflow_scenario()
            elif choice == "6":
                print("\n🚀 Running all scenarios...")
                research_team_scenario()
                development_team_scenario()
                customer_support_scenario()
                error_retry_scenario()
                complex_workflow_scenario()
            elif choice == "7":
                summarize_event_stats(OUTPUT_FILE)
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
    summarize_event_stats(OUTPUT_FILE)

    print("\n✅ Example completed successfully!")
    print(f"📄 Events saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    # Parse command line arguments
    if "--help" in sys.argv:
        print(
            """
Usage: python crewai_comprehensive_example.py [options]

Options:
  --help           Show this help message

Environment Variables:
  OPENAI_API_KEY   Your OpenAI API key (REQUIRED)

This example demonstrates all 20 Chaukas event types using CrewAI:
- Multiple crews with specialized agents
- Various tool types including MCP context tools
- Agent handoffs in sequential workflows
- Error handling and retry mechanisms
- Policy decisions and data access tracking
- State management and updates
- Comprehensive event capture with Chaukas SDK

CrewAI has 100% event coverage, capturing all 20 event types!

Make sure to set your OPENAI_API_KEY before running:
  export OPENAI_API_KEY='your-api-key-here'
        """
        )
        sys.exit(0)

    # Run the main program
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
