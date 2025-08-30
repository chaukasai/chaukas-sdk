"""
Example usage with CrewAI.
"""

import os
import asyncio
from crewai import Agent, Task, Crew, Process

# Import and enable Chaukas instrumentation
import chaukas

# Set environment variables
os.environ["CHAUKAS_ENDPOINT"] = "https://api.chaukas.com"
os.environ["CHAUKAS_API_KEY"] = "your-api-key-here"
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

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
    
    # Define tasks
    task1 = Task(
        description="""Conduct a comprehensive analysis of the latest advancements in AI in 2024.
        Identify key trends, breakthrough technologies, and potential industry impacts.""",
        expected_output="A comprehensive 3 paragraphs long report on the latest AI advancements in 2024.",
        agent=researcher
    )
    
    task2 = Task(
        description="""Using the insights provided, develop an engaging blog post
        that highlights the most significant AI advancements.""",
        expected_output="A compelling 4 paragraph blog post formatted as markdown.",
        agent=writer
    )
    
    # Create and run crew - this will be automatically instrumented
    crew = Crew(
        agents=[researcher, writer],
        tasks=[task1, task2],
        verbose=2,
        process=Process.sequential
    )
    
    # Execute crew - all agent interactions and task execution will be traced
    result = crew.kickoff()
    
    print(f"Crew result: {result}")


if __name__ == "__main__":
    main()