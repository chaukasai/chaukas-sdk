"""
Demonstration of POLICY_DECISION event capture with CrewAI Task Guardrails.

This example uses CrewAI's built-in Task Guardrail feature to validate outputs
and trigger POLICY_DECISION events when guardrails are checked.

Requires:
- pip install crewai
- OPENAI_API_KEY environment variable
"""

import os
from pathlib import Path

from crewai import Agent, Crew, Process, Task
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


# Define guardrail validation functions
def validate_content_safety(output):
    """
    Guardrail to ensure content doesn't include sensitive information.
    This will trigger POLICY_DECISION events.
    """
    # Check for sensitive patterns
    sensitive_keywords = [
        "password",
        "secret",
        "api_key",
        "private_key",
        "ssn",
        "credit card",
        "confidential",
    ]

    output_lower = output.lower()
    for keyword in sensitive_keywords:
        if keyword in output_lower:
            print(
                f"⚠️  Guardrail FAILED: Content contains sensitive keyword '{keyword}'"
            )
            return False

    print("✅ Content Safety Guardrail PASSED")
    return True


def validate_output_length(output):
    """
    Guardrail to ensure output meets minimum quality standards.
    This will trigger POLICY_DECISION events.
    """
    min_length = 100
    max_length = 2000

    if len(output) < min_length:
        print(
            f"⚠️  Guardrail FAILED: Output too short ({len(output)} chars, min {min_length})"
        )
        return False

    if len(output) > max_length:
        print(
            f"⚠️  Guardrail FAILED: Output too long ({len(output)} chars, max {max_length})"
        )
        return False

    print(f"✅ Length Guardrail PASSED ({len(output)} chars)")
    return True


def validate_professional_tone(output):
    """
    Guardrail to ensure output maintains professional language.
    This will trigger POLICY_DECISION events.
    """
    # Check for unprofessional patterns
    unprofessional_words = [
        "stupid",
        "dumb",
        "idiotic",
        "sucks",
        "terrible",
        "worst",
        "garbage",
        "trash",
        "awful",
    ]

    output_lower = output.lower()
    for word in unprofessional_words:
        if word in output_lower:
            print(f"⚠️  Guardrail FAILED: Unprofessional language detected '{word}'")
            return False

    print("✅ Professional Tone Guardrail PASSED")
    return True


def validate_competitor_mentions(output):
    """
    Guardrail to check for competitor mentions in marketing content.
    This will trigger POLICY_DECISION events.
    """
    competitors = ["competitorA", "competitorB", "rival corp"]

    output_lower = output.lower()
    for competitor in competitors:
        if competitor in output_lower:
            print(f"⚠️  Guardrail FAILED: Competitor mention detected '{competitor}'")
            return False

    print("✅ Competitor Mention Guardrail PASSED")
    return True


def main():
    print("=" * 60)
    print("CrewAI Task Guardrails - POLICY_DECISION Event Demonstration")
    print("=" * 60)
    print("\nThis example demonstrates POLICY_DECISION event capture when")
    print("task guardrails validate agent outputs.")
    print("=" * 60 + "\n")

    # Create agents
    content_writer = Agent(
        role="Content Writer",
        goal="Write professional marketing content",
        backstory="""You are a professional content writer who creates engaging
        marketing materials while maintaining professional standards.""",
        verbose=True,
    )

    data_analyst = Agent(
        role="Data Analyst",
        goal="Analyze data and provide insights",
        backstory="""You are a data analyst who provides clear, concise insights
        while protecting sensitive information.""",
        verbose=True,
    )

    # Create tasks with guardrails
    print("📋 Task 1: Content Writing with Multiple Guardrails\n")

    content_task = Task(
        description="""Write a professional product description for our new AI-powered
        analytics platform. Highlight key features and benefits in 150-300 words.
        Focus on innovation and customer value.""",
        expected_output="A professional product description between 150-300 words",
        agent=content_writer,
        # Add multiple guardrails that will be checked
        guardrail=lambda output: (
            validate_content_safety(output)
            and validate_output_length(output)
            and validate_professional_tone(output)
            and validate_competitor_mentions(output)
        ),
    )

    print("📋 Task 2: Data Analysis with Sensitivity Check\n")

    analysis_task = Task(
        description="""Analyze the Q4 sales performance and provide a summary report.
        Focus on trends, key metrics, and actionable insights. Keep it concise.""",
        expected_output="A concise Q4 sales analysis report",
        agent=data_analyst,
        guardrail=lambda output: (
            validate_content_safety(output) and validate_output_length(output)
        ),
    )

    # Create crew
    crew = Crew(
        agents=[content_writer, data_analyst],
        tasks=[content_task, analysis_task],
        verbose=True,
        process=Process.sequential,
    )

    print("🚀 Starting crew with guardrail validation...")
    print("(Guardrail checks will trigger POLICY_DECISION events)\n")

    try:
        result = crew.kickoff()

        print("\n" + "=" * 60)
        print("✅ Crew execution completed!")
        print("=" * 60)
        if result:
            output = (
                str(result)[:500] + "..." if len(str(result)) > 500 else str(result)
            )
            print(f"\nFinal Output:\n{output}")

        print("\n" + "=" * 60)
        print("📊 Events Captured:")
        print("=" * 60)
        print("The following events were captured in the output file:")
        print("• SESSION_START/END - Crew execution lifecycle")
        print("• AGENT_START/END - Agent execution")
        print("• POLICY_DECISION - Guardrail validations")
        print("  - Decision: 'allow' or 'block'")
        print("  - Rule: Validation function name")
        print("  - Context: Task and agent information")
        print("  - Outcome: Pass/fail status")
        print("• MODEL_INVOCATION_START/END - LLM calls")
        print("\nNote: POLICY_DECISION events are emitted when CrewAI's Task")
        print("Guardrails validate agent outputs against defined rules.")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error during crew execution: {e}")
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
