"""
Simple LangChain chain example with Chaukas instrumentation.

This example demonstrates basic LangChain chain usage with Chaukas callback handler.
It captures:
- SESSION_START/END events
- MODEL_INVOCATION_START/END events
- INPUT_RECEIVED/OUTPUT_EMITTED events
"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import chaukas

# Enable Chaukas instrumentation
chaukas.enable_chaukas()

# Get the callback handler
callback = chaukas.get_langchain_callback()


def main():
    """Run a simple LangChain chain with Chaukas instrumentation."""

    # Create a simple chain: prompt | llm | output_parser
    prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
    llm = ChatOpenAI(model="gpt-4", temperature=0.7)
    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser

    # Invoke the chain with Chaukas callback
    result = chain.invoke(
        {"topic": "programming"},
        config={"callbacks": [callback]}
    )

    print("Result:", result)


if __name__ == "__main__":
    main()
