"""
Simple LangChain chain example with Chaukas instrumentation.

This example demonstrates basic LangChain chain usage with automatic Chaukas tracking.
It captures:
- SESSION_START/END events
- MODEL_INVOCATION_START/END events
- INPUT_RECEIVED/OUTPUT_EMITTED events
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from chaukas import sdk as chaukas

# Enable Chaukas instrumentation (one-line setup - automatically tracks all LangChain operations!)
chaukas.enable_chaukas()


def main():
    """Run a simple LangChain chain with automatic Chaukas instrumentation."""

    # Create a simple chain: prompt | llm | output_parser
    prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
    llm = ChatOpenAI(model="gpt-4", temperature=0.7)
    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser

    # Invoke the chain - Chaukas automatically tracks everything!
    result = chain.invoke({"topic": "programming"})

    print("Result:", result)
    print("\n✅ Events captured by Chaukas")


if __name__ == "__main__":
    import time

    main()
