"""
Demonstration of DATA_ACCESS event capture with CrewAI Knowledge feature.

This example uses CrewAI's built-in Knowledge feature with vector store integration
to trigger DATA_ACCESS events when agents retrieve information from knowledge sources.

Requires:
- pip install crewai chromadb (or qdrant-client for Qdrant)
- OPENAI_API_KEY environment variable
"""

import os
from pathlib import Path

from crewai import Agent, Crew, Process, Task
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
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


def main():
    print("=" * 60)
    print("CrewAI Knowledge Feature - DATA_ACCESS Event Demonstration")
    print("=" * 60)
    print("\nThis example demonstrates DATA_ACCESS event capture when agents")
    print("retrieve information from CrewAI's Knowledge sources.")
    print("=" * 60 + "\n")

    # Create knowledge sources with domain information
    ai_knowledge = StringKnowledgeSource(
        content="""
        Artificial Intelligence (AI) in 2024 has seen major advancements:
        - Large Language Models have improved context windows to 200K+ tokens
        - Multimodal AI now seamlessly integrates text, image, and audio
        - AI agents can now perform complex multi-step reasoning
        - Edge AI deployment has become more efficient with quantization
        - Retrieval-Augmented Generation (RAG) is now standard practice
        """,
        metadata={"source": "ai_trends_2024"},
    )

    healthcare_knowledge = StringKnowledgeSource(
        content="""
        AI in Healthcare 2024 breakthroughs:
        - AI-powered diagnostic tools achieve 95% accuracy in imaging
        - Personalized treatment plans using genetic and lifestyle data
        - Real-time patient monitoring with predictive alerts
        - Drug discovery accelerated by 10x using AI simulations
        - Mental health chatbots provide 24/7 support
        """,
        metadata={"source": "healthcare_ai_2024"},
    )

    finance_knowledge = StringKnowledgeSource(
        content="""
        AI in Finance 2024 applications:
        - Fraud detection systems process millions of transactions in real-time
        - Algorithmic trading strategies adapt to market conditions instantly
        - Credit scoring models incorporate alternative data sources
        - Regulatory compliance automated with NLP document analysis
        - Personal finance advisors use AI for portfolio optimization
        """,
        metadata={"source": "finance_ai_2024"},
    )

    # Create an agent with access to knowledge sources
    research_agent = Agent(
        role="AI Research Analyst",
        goal="Research and analyze AI developments across different sectors",
        backstory="""You are an expert AI researcher with deep knowledge of artificial
        intelligence applications across healthcare and finance. You use comprehensive
        knowledge sources to provide accurate and detailed insights.""",
        verbose=True,
        knowledge_sources=[ai_knowledge, healthcare_knowledge, finance_knowledge],
    )

    # Create a task that will query the knowledge sources
    research_task = Task(
        description="""Research the latest AI developments in 2024, focusing on:
        1. General AI advancements
        2. Healthcare applications
        3. Finance applications

        Use the available knowledge sources to gather comprehensive information.""",
        expected_output="""A detailed report covering:
        - Key AI advancements in 2024
        - Specific healthcare AI applications
        - Finance sector AI implementations
        - Analysis of trends across sectors""",
        agent=research_agent,
    )

    # Create crew
    crew = Crew(
        agents=[research_agent],
        tasks=[research_task],
        verbose=True,
        process=Process.sequential,
        # Configure embedder (optional - uses default if not specified)
        embedder={"provider": "openai", "config": {"model": "text-embedding-3-small"}},
    )

    print("🚀 Starting knowledge-powered research...")
    print("(Knowledge queries will trigger DATA_ACCESS events)\n")

    try:
        result = crew.kickoff()

        print("\n" + "=" * 60)
        print("✅ Research completed!")
        print("=" * 60)
        if result:
            output = (
                str(result)[:500] + "..." if len(str(result)) > 500 else str(result)
            )
            print(f"\nResearch Report:\n{output}")

        print("\n" + "=" * 60)
        print("📊 Events Captured:")
        print("=" * 60)
        print("The following events were captured in the output file:")
        print("• SESSION_START/END - Crew execution lifecycle")
        print("• AGENT_START/END - Research agent execution")
        print("• DATA_ACCESS - Knowledge retrieval from vector store")
        print("  - Source type: 'knowledge_source'")
        print("  - Query text and metadata included")
        print("  - Retrieved document chunks")
        print("• MODEL_INVOCATION_START/END - LLM calls")
        print("• TOOL_CALL_START/END - If any tools used")
        print("\nNote: DATA_ACCESS events are emitted when CrewAI's Knowledge")
        print("feature retrieves information from vector stores using RAG.")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error during research: {e}")
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
