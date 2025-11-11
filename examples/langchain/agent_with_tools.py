"""
LangChain tool calling example using Chaukas instrumentation.

This example demonstrates LangChain tool usage with OpenAI function calling. It captures:
- TOOL_CALL_START/END events
- MODEL_INVOCATION events
- SESSION_START/END events
- INPUT_RECEIVED/OUTPUT_EMITTED events
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

from chaukas import sdk as chaukas

# Enable Chaukas instrumentation (one-line setup!)
chaukas.enable_chaukas()


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
    """Run a LangChain model with tools and Chaukas instrumentation."""

    # Create the LLM with tools bound
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    tools = [calculator, word_counter]
    llm_with_tools = llm.bind_tools(tools)

    # Create a prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that can do calculations and count words.",
            ),
            ("human", "{input}"),
        ]
    )

    # Create the chain
    chain = prompt | llm_with_tools

    # Invoke the chain - Chaukas automatically tracks everything!
    result = chain.invoke({"input": "What is 25 * 4?"})

    print("\nResult:", result)
    print(
        "\nTool calls:", result.tool_calls if hasattr(result, "tool_calls") else "None"
    )

    # If there are tool calls, execute them
    if hasattr(result, "tool_calls") and result.tool_calls:
        print("\nExecuting tools:")
        for tool_call in result.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            print(f"  - {tool_name}({tool_args})")

            # Execute the tool
            if tool_name == "calculator":
                tool_result = calculator.invoke(tool_args)
                print(f"    Result: {tool_result}")
            elif tool_name == "word_counter":
                tool_result = word_counter.invoke(tool_args)
                print(f"    Result: {tool_result}")

    print("\n✅ Events captured by Chaukas")


if __name__ == "__main__":
    import time

    main()

    # Give async operations time to complete
    time.sleep(0.5)

    # Explicitly disable Chaukas to flush events to file
    chaukas.disable_chaukas()

    print("✅ Events written to file")
