"""
Example usage with CrewAI including tools to test event capture.
This example demonstrates tool usage and event capture.
"""

import os

# Disable CrewAI telemetry to prevent "Service Unavailable" errors
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

from crewai import Agent, Task, Crew, Process
from crewai.tools import tool

# Import and enable Chaukas instrumentation
from chaukas import sdk as chaukas

# Set environment variables
os.environ["CHAUKAS_TENANT_ID"] = "test_tenant_1"
os.environ["CHAUKAS_PROJECT_ID"] = "test_project_1"
os.environ["CHAUKAS_OUTPUT_MODE"] = "file"
os.environ["CHAUKAS_OUTPUT_FILE"] = "crewai_tools_output.jsonl"
os.environ["CHAUKAS_BATCH_SIZE"] = "1"
#os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# Enable instrumentation - this will automatically detect and patch CrewAI
chaukas.enable_chaukas()


# Define custom tools
@tool
def search_tool(query: str) -> str:
    """
    Simulates a search tool for finding information.
    This tool will trigger TOOL_CALL_START and TOOL_CALL_END events.
    """
    return f"Search results for '{query}': Found 5 relevant articles about the topic."


@tool
def calculator_tool(expression: str) -> str:
    """
    Simulates a calculator tool for mathematical operations.
    This tool will trigger TOOL_CALL_START and TOOL_CALL_END events.
    """
    try:
        # Simple evaluation (in production, use safer methods)
        result = eval(expression)
        return f"Calculation result: {expression} = {result}"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"


@tool
def mcp_data_fetch(endpoint: str) -> str:
    """
    Simulates an MCP tool for fetching data from a model context.
    This should trigger MCP_CALL_START and MCP_CALL_END events if detected.
    """
    return f"MCP data from {endpoint}: Sample context data retrieved successfully."


def main():
    # Define agents with tools
    researcher = Agent(
        role='Senior Research Analyst',
        goal='Research and analyze information using various tools',
        backstory="""You are an expert researcher who knows how to use various tools
        to gather and analyze information effectively.""",
        tools=[search_tool, calculator_tool],
        verbose=True
    )
    
    data_analyst = Agent(
        role='Data Analysis Expert',
        goal='Analyze data and provide insights using specialized tools',
        backstory="""You are a data analysis expert who can work with various data sources
        and tools including MCP endpoints for context data.""",
        tools=[calculator_tool, mcp_data_fetch],
        verbose=True
    )
    
    # Define tasks that will use tools with interpolated inputs
    research_task = Task(
        description="""Research the topic of '{research_topic}'.
        Use the search tool to find relevant information.
        Then calculate the percentage growth if AI investment grew from ${base_investment}B to ${current_investment}B.""",
        expected_output="A report with search results and calculated growth percentage.",
        agent=researcher
    )
    
    analysis_task = Task(
        description="""Analyze the research findings for {research_topic}.
        Use the MCP data fetch tool to get additional context from endpoint '{mcp_endpoint}'.
        Calculate the compound annual growth rate based on the investment data provided.""",
        expected_output="An analytical summary with MCP context data included.",
        agent=data_analyst
    )
    
    # Create and run crew
    crew = Crew(
        agents=[researcher, data_analyst],
        tasks=[research_task, analysis_task],
        verbose=True,
        process=Process.sequential
    )
    
    # Define inputs for the crew
    inputs = {
        "research_topic": "AI advancements in 2024",
        "base_investment": 100,
        "current_investment": 150,
        "mcp_endpoint": "/ai/trends",
        "investment_years": {"2023": 100, "2024": 150}
    }
    
    print("Starting crew execution with tools...")
    print(f"Inputs: {inputs}")
    print("This will generate various events including:")
    print("- SESSION_START/END")
    print("- AGENT_START/END")
    print("- TOOL_CALL_START/END (for search and calculator tools)")
    print("- MCP_CALL_START/END (for mcp_data_fetch tool)")
    print("- INPUT_RECEIVED (with crew inputs)")
    print("- OUTPUT_EMITTED")
    print("-" * 50)
    
    # Execute crew with inputs - all tool usage and INPUT_RECEIVED event will be traced
    result = crew.kickoff(inputs=inputs)
    
    print("-" * 50)
    print(f"Crew execution completed!")
    print(f"Result: {result}")
    print(f"\nCheck {os.environ['CHAUKAS_OUTPUT_FILE']} for captured events.")


if __name__ == "__main__":
    main()