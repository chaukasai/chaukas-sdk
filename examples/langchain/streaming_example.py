"""
Streaming Example with Chaukas Instrumentation

This example demonstrates how Chaukas captures streaming output from LangChain.
The on_text callback emits OUTPUT_EMITTED events for each text chunk during streaming.

Key Events Captured:
- SESSION_START/END - Chain lifecycle
- MODEL_INVOCATION_START/END - LLM call tracking
- OUTPUT_EMITTED - Multiple events, one per text chunk via on_text callback

Requirements:
    pip install chaukas-sdk langchain langchain-openai

Environment Variables:
    OPENAI_API_KEY - Your OpenAI API key
    CHAUKAS_OUTPUT_MODE - Set to "file" for local testing
    CHAUKAS_OUTPUT_FILE - Path to output file (e.g., "streaming_events.jsonl")
"""

import os

# For local testing, output events to a file
os.environ.setdefault("CHAUKAS_OUTPUT_MODE", "file")
os.environ.setdefault("CHAUKAS_OUTPUT_FILE", "streaming_events.jsonl")
os.environ.setdefault("CHAUKAS_TENANT_ID", "demo-tenant")
os.environ.setdefault("CHAUKAS_PROJECT_ID", "langchain-streaming-demo")

from chaukas import sdk as chaukas
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Enable Chaukas instrumentation - one line setup!
chaukas.enable_chaukas()


def main():
    """Demonstrate streaming output capture with Chaukas."""
    print("=" * 60)
    print("LangChain Streaming Example with Chaukas Instrumentation")
    print("=" * 60)
    print()

    # Create a simple chain with streaming enabled
    prompt = ChatPromptTemplate.from_template(
        "Write a short haiku (3 lines) about {topic}. Be creative!"
    )
    llm = ChatOpenAI(model="gpt-4o-mini", streaming=True, temperature=0.7)
    chain = prompt | llm | StrOutputParser()

    print("Streaming response (each chunk triggers OUTPUT_EMITTED):")
    print("-" * 40)

    # Stream the output - each chunk triggers on_text callback
    # which emits OUTPUT_EMITTED events with is_streaming=True metadata
    for chunk in chain.stream({"topic": "the ocean at sunset"}):
        print(chunk, end="", flush=True)

    print()
    print("-" * 40)
    print()

    # Try another streaming example
    print("Streaming a longer response:")
    print("-" * 40)

    prompt2 = ChatPromptTemplate.from_template(
        "List 3 interesting facts about {subject}. Keep each fact to one sentence."
    )
    chain2 = prompt2 | llm | StrOutputParser()

    for chunk in chain2.stream({"subject": "dolphins"}):
        print(chunk, end="", flush=True)

    print()
    print("-" * 40)
    print()
    print("Check streaming_events.jsonl for captured events.")
    print("Look for OUTPUT_EMITTED events with 'is_streaming': true")


if __name__ == "__main__":
    main()
