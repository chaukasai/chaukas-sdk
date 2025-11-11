"""
LangChain multi-chain workflow example with Chaukas instrumentation.

This example demonstrates complex workflows with multiple chains. It captures:
- Multiple SESSION events
- Multiple AGENT events (if agent chains are involved)
- All MODEL_INVOCATION events
- Complete event hierarchy
"""

import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from chaukas import sdk as chaukas

# Enable Chaukas instrumentation (one-line setup!)
chaukas.enable_chaukas()


def main():
    """Run multiple chains in a workflow with Chaukas instrumentation."""

    llm = ChatOpenAI(model="gpt-4", temperature=0.7)

    # Chain 1: Generate a topic
    topic_prompt = ChatPromptTemplate.from_template(
        "Generate a random topic about {subject}"
    )
    topic_chain = topic_prompt | llm | StrOutputParser()

    # Chain 2: Write about the topic
    writing_prompt = ChatPromptTemplate.from_template(
        "Write a short paragraph about: {topic}"
    )
    writing_chain = writing_prompt | llm | StrOutputParser()

    # Chain 3: Summarize the paragraph
    summary_prompt = ChatPromptTemplate.from_template(
        "Summarize this paragraph in one sentence: {paragraph}"
    )
    summary_chain = summary_prompt | llm | StrOutputParser()

    # All chains are automatically tracked by Chaukas!
    print("Step 1: Generating topic...")
    topic = topic_chain.invoke({"subject": "artificial intelligence"})
    print(f"Topic: {topic}\n")

    print("Step 2: Writing about topic...")
    paragraph = writing_chain.invoke({"topic": topic})
    print(f"Paragraph: {paragraph}\n")

    print("Step 3: Summarizing paragraph...")
    summary = summary_chain.invoke({"paragraph": paragraph})
    print(f"Summary: {summary}")

    print("\n✅ Events captured by Chaukas")


if __name__ == "__main__":
    import time

    main()

    # Give async operations time to complete
    time.sleep(0.5)

    # Explicitly disable Chaukas to flush events to file
    chaukas.disable_chaukas()

    print("✅ Events written to file")
