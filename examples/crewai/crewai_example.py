"""
Example usage with CrewAI.
"""

import os

# Disable CrewAI telemetry to prevent "Service Unavailable" errors
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

import asyncio
from crewai import Agent, Task, Crew, Process

# Import and enable Chaukas instrumentation
from chaukas import sdk as chaukas

# Set environment variables
os.environ["CHAUKAS_TENANT_ID"] = "test_tenant_1"
os.environ["CHAUKAS_PROJECT_ID"] = "test_project_1"
os.environ["CHAUKAS_OUTPUT_MODE"] = "file"
os.environ["CHAUKAS_OUTPUT_FILE"] = "crewai_output.jsonl"
os.environ["CHAUKAS_BATCH_SIZE"] = "1"
#os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# Enable instrumentation - this will automatically detect and patch CrewAI
chaukas.enable_chaukas()


def main():
    # Define agents
    researcher = Agent(
        role='Senior Research Analyst',
        goal='Uncover cutting-edge developments in AI and data science',
        backstory="""You work at a leading tech think tank.
        Your expertise lies in identifying emerging trends.
        You have a knack for dissecting complex data and presenting actionable insights.""",
        verbose=True
    )
    
    writer = Agent(
        role='Tech Content Strategist',
        goal='Craft compelling content on tech advancements',
        backstory="""You are a renowned Content Strategist, known for your insightful
        and engaging articles. You transform complex concepts into compelling narratives.""",
        verbose=True
    )
    
    # Define tasks with interpolated inputs
    task1 = Task(
        description="""Conduct a comprehensive analysis of the latest advancements in {topic} in {year}.
        Focus particularly on {focus_area}.
        Identify key trends, breakthrough technologies, and potential industry impacts.""",
        expected_output="A comprehensive 3 paragraphs long report on the latest {topic} advancements in {year}.",
        agent=researcher
    )
    
    task2 = Task(
        description="""Using the insights provided, develop an engaging blog post
        that highlights the most significant {topic} advancements in {focus_area}.""",
        expected_output="A compelling 4 paragraph blog post formatted as markdown.",
        agent=writer
    )
    
    # Create and run crew - this will be automatically instrumented
    crew = Crew(
        agents=[researcher, writer],
        tasks=[task1, task2],
        verbose=True,
        process=Process.sequential
    )
    
    # Define inputs for the crew
    inputs = {
        "topic": "AI",
        "year": "2024",
        "focus_area": "healthcare and finance applications"
    }
    
    # Execute crew with inputs - all agent interactions and INPUT_RECEIVED event will be traced
    print(f"Starting crew with inputs: {inputs}")
    result = crew.kickoff(inputs=inputs)
    
    print(f"Crew result: {result}")


if __name__ == "__main__":
    main()