"""
Demonstration of SYSTEM event capture with CrewAI Flows.

This example uses CrewAI's Flows feature to create a structured workflow
that triggers SYSTEM events for flow lifecycle tracking.

Requires:
- pip install crewai
- OPENAI_API_KEY environment variable
"""

import os
from pathlib import Path

from crewai import Agent, Crew, Process, Task
from crewai.flow.flow import Flow, listen, start
from dotenv import load_dotenv

# Load .env from the same directory as this script
script_dir = Path(__file__).parent
load_dotenv(script_dir / ".env")

# Disable CrewAI telemetry
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

# Import and enable Chaukas instrumentation
from chaukas import sdk as chaukas

# Enable instrumentation - this will automatically detect and patch CrewAI
chaukas.enable_chaukas()


class ContentCreationFlow(Flow):
    """
    A Flow that orchestrates content creation through multiple stages.
    Each stage transition will trigger SYSTEM events.
    """

    @start()
    def research_phase(self):
        """Phase 1: Research content topics."""
        print("\n📚 Phase 1: Research Phase")
        print("=" * 60)

        researcher = Agent(
            role="Research Analyst",
            goal="Research and gather information on given topics",
            backstory="Expert at finding and synthesizing information from various sources.",
            verbose=True,
        )

        research_task = Task(
            description="Research the latest trends in AI and machine learning for 2024",
            expected_output="A summary of key AI/ML trends",
            agent=researcher,
        )

        crew = Crew(agents=[researcher], tasks=[research_task], verbose=True)

        result = crew.kickoff()
        return {"research_results": str(result)}

    @listen(research_phase)
    def writing_phase(self, research_results):
        """Phase 2: Write content based on research."""
        print("\n✍️  Phase 2: Writing Phase")
        print("=" * 60)

        writer = Agent(
            role="Content Writer",
            goal="Create engaging content based on research",
            backstory="Skilled writer who transforms research into compelling narratives.",
            verbose=True,
        )

        writing_task = Task(
            description=f"""Using the research findings below, write a blog post about AI trends.

            Research: {research_results}

            Make it engaging and informative.""",
            expected_output="A well-written blog post draft",
            agent=writer,
        )

        crew = Crew(agents=[writer], tasks=[writing_task], verbose=True)

        result = crew.kickoff()
        return {"draft_content": str(result)}

    @listen(writing_phase)
    def review_phase(self, draft_content):
        """Phase 3: Review and finalize content."""
        print("\n🔍 Phase 3: Review Phase")
        print("=" * 60)

        editor = Agent(
            role="Content Editor",
            goal="Review and improve written content",
            backstory="Experienced editor with an eye for detail and quality.",
            verbose=True,
        )

        review_task = Task(
            description=f"""Review the following blog post draft and provide feedback:

            Draft: {draft_content}

            Check for clarity, engagement, and accuracy.""",
            expected_output="Final reviewed content with improvements",
            agent=editor,
        )

        crew = Crew(agents=[editor], tasks=[review_task], verbose=True)

        result = crew.kickoff()
        return {"final_content": str(result)}

    @listen(review_phase)
    def publish_phase(self, final_content):
        """Phase 4: Prepare content for publishing."""
        print("\n📤 Phase 4: Publishing Phase")
        print("=" * 60)
        print("Content ready for publishing:")
        print("-" * 60)
        output = (
            final_content[:500] + "..." if len(final_content) > 500 else final_content
        )
        print(output)
        print("-" * 60)
        return {"status": "published", "content": final_content}


def main():
    print("=" * 60)
    print("CrewAI Flows - SYSTEM Event Demonstration")
    print("=" * 60)
    print("\nThis example demonstrates SYSTEM event capture when CrewAI Flows")
    print("orchestrate multi-stage workflows.")
    print("=" * 60 + "\n")

    # Create and run the flow
    print("🚀 Starting Content Creation Flow...")
    print("(Flow lifecycle events will trigger SYSTEM events)\n")

    try:
        flow = ContentCreationFlow()
        result = flow.kickoff()

        print("\n" + "=" * 60)
        print("✅ Flow completed successfully!")
        print("=" * 60)
        print(f"\nFlow Result: {result}")

        print("\n" + "=" * 60)
        print("📊 Events Captured:")
        print("=" * 60)
        print("The following events were captured in the output file:")
        print("• SYSTEM (FlowStartedEvent) - When flow begins")
        print("• SYSTEM (MethodExecutionStartedEvent) - For each phase start")
        print("• SYSTEM (MethodExecutionFinishedEvent) - For each phase end")
        print("• SYSTEM (FlowFinishedEvent) - When flow completes")
        print("• SESSION_START/END - For each crew in the flow")
        print("• AGENT_START/END - For each agent execution")
        print("• MODEL_INVOCATION_START/END - For LLM calls")
        print("\nNote: SYSTEM events are emitted by CrewAI Flows to track")
        print("workflow orchestration and execution flow between stages.")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error during flow execution: {e}")
        import traceback

        traceback.print_exc()

    # Flush events
    print("\n🔄 Flushing events to file...")
    client = chaukas.get_client()
    if client:
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(client.flush())
            else:
                loop.run_until_complete(client.flush())
        except:
            asyncio.run(client.flush())
        print("✅ Events written to output file")


if __name__ == "__main__":
    main()
