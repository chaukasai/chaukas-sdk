"""
Custom Events Example with Chaukas Instrumentation

This example demonstrates how to emit custom events from within LangChain workflows.
Custom events are captured as SYSTEM events via the on_custom_event callback.

IMPORTANT: dispatch_custom_event must be called from within a Runnable context.
It cannot be called from arbitrary code outside of a chain execution.

Key Events Captured:
- SESSION_START/END - Chain lifecycle
- MODEL_INVOCATION_START/END - LLM call tracking
- SYSTEM - Custom events with name, data, and severity levels

Requirements:
    pip install chaukas-sdk langchain langchain-openai

Environment Variables:
    OPENAI_API_KEY - Your OpenAI API key
    CHAUKAS_OUTPUT_MODE - Set to "file" for local testing
    CHAUKAS_OUTPUT_FILE - Path to output file (e.g., "custom_events.jsonl")
"""

import os

# For local testing, output events to a file
os.environ.setdefault("CHAUKAS_OUTPUT_MODE", "file")
os.environ.setdefault("CHAUKAS_OUTPUT_FILE", "custom_events.jsonl")
os.environ.setdefault("CHAUKAS_TENANT_ID", "demo-tenant")
os.environ.setdefault("CHAUKAS_PROJECT_ID", "langchain-custom-events-demo")

from chaukas import sdk as chaukas
from langchain_core.runnables import RunnableLambda
from langchain_core.callbacks.manager import dispatch_custom_event
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Enable Chaukas instrumentation - one line setup!
chaukas.enable_chaukas()


def preprocessing_step(input_data: dict) -> dict:
    """
    A preprocessing step that emits custom events.

    Custom events are useful for:
    - Tracking workflow progress
    - Logging checkpoints
    - Recording validation results
    - Monitoring custom metrics
    """
    text = input_data.get("text", "")

    # Emit event at start of processing
    dispatch_custom_event(
        "preprocessing_started",
        {"input_length": len(text), "has_content": bool(text.strip())},
    )

    # Validate input and emit appropriate event
    if len(text) < 10:
        dispatch_custom_event(
            "validation_warning",
            {"issue": "input_too_short", "length": len(text), "min_required": 10},
            metadata={"severity": "warn"},
        )
    else:
        dispatch_custom_event(
            "validation_passed",
            {"length": len(text)},
            metadata={"severity": "info"},
        )

    # Emit checkpoint event
    dispatch_custom_event(
        "checkpoint",
        {"stage": "preprocessing_complete", "ready_for_llm": True},
        metadata={"severity": "debug"},
    )

    return input_data


def postprocessing_step(llm_output: str) -> str:
    """Post-process LLM output and emit custom events."""
    # Emit event with output analysis
    dispatch_custom_event(
        "postprocessing_started",
        {"output_length": len(llm_output), "word_count": len(llm_output.split())},
    )

    # Check for potential issues
    if len(llm_output) > 1000:
        dispatch_custom_event(
            "output_warning",
            {"issue": "output_too_long", "length": len(llm_output)},
            metadata={"severity": "warn"},
        )

    dispatch_custom_event(
        "workflow_complete",
        {"status": "success", "final_length": len(llm_output)},
        metadata={"severity": "info"},
    )

    return llm_output


def main():
    """Demonstrate custom event emission with Chaukas."""
    print("=" * 60)
    print("LangChain Custom Events Example with Chaukas Instrumentation")
    print("=" * 60)
    print()

    # Create runnables from our custom functions
    preprocess = RunnableLambda(preprocessing_step)
    postprocess = RunnableLambda(postprocessing_step)

    # Create the LLM chain
    prompt = ChatPromptTemplate.from_template(
        "Summarize the following text in 2-3 sentences:\n\n{text}"
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    parser = StrOutputParser()

    # Build the full chain with pre and post processing
    chain = preprocess | prompt | llm | parser | postprocess

    # Test with a valid input
    print("Example 1: Normal text processing")
    print("-" * 40)
    result = chain.invoke(
        {
            "text": "Chaukas SDK is an open-source observability solution for AI agents. "
            "It provides comprehensive instrumentation for popular frameworks like "
            "LangChain, CrewAI, and OpenAI Agents. With just one line of code, "
            "developers can capture detailed events about their AI agent's behavior."
        }
    )
    print(f"Result: {result}")
    print()

    # Test with a short input to trigger warning
    print("Example 2: Short input (triggers warning)")
    print("-" * 40)
    result = chain.invoke({"text": "Hello!"})
    print(f"Result: {result}")
    print()

    print("Check custom_events.jsonl for captured events.")
    print("Look for SYSTEM events with 'custom_event_name' in metadata.")
    print()
    print("Severity levels used:")
    print("  - debug: checkpoint events")
    print("  - info: validation_passed, workflow_complete")
    print("  - warn: validation_warning, output_warning")


if __name__ == "__main__":
    main()
