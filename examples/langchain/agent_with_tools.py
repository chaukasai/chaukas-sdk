"""
LangChain agent with tools example using Chaukas instrumentation.

This example demonstrates LangChain agent usage with tools. It captures:
- AGENT_START/END events
- TOOL_CALL_START/END events
- MODEL_INVOCATION events
- All standard chain events
"""

import os
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

import chaukas

# Enable Chaukas instrumentation
chaukas.enable_chaukas()

# Get the callback handler
callback = chaukas.get_langchain_callback()


# Define custom tools
@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Input should be a valid Python expression."""
    try:
        result = eval(expression)
        return f"The result is: {result}"
    except Exception as e:
        return f"Error evaluating expression: {e}"


@tool
def word_counter(text: str) -> str:
    """Count the number of words in a given text."""
    words = text.split()
    return f"The text contains {len(words)} words"


def main():
    """Run a LangChain agent with tools and Chaukas instrumentation."""

    # Create the LLM
    llm = ChatOpenAI(model="gpt-4", temperature=0)

    # Create tools list
    tools = [calculator, word_counter]

    # Create the prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that can do calculations and count words."),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Create the agent
    agent = create_openai_tools_agent(llm, tools, prompt)

    # Create agent executor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # Run the agent with Chaukas callback
    result = agent_executor.invoke(
        {"input": "What is 25 * 4? Also, how many words are in this sentence?"},
        config={"callbacks": [callback]}
    )

    print("\nFinal Result:", result["output"])


if __name__ == "__main__":
    main()
