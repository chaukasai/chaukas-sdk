"""
MCP Tools Example with Chaukas Instrumentation

This example demonstrates how Chaukas captures MCP (Model Context Protocol) tool calls.
When tools are invoked with `mcp_server` metadata, Chaukas emits MCP_CALL_START/END
events instead of regular TOOL_CALL events.

MCP Detection:
- Chaukas detects MCP tools by looking for `mcp_server` or `mcp_server_name` in metadata
- When present, events are classified as MCP_CALL_START/END
- MCP events include protocol version, server name, and server URL

Key Events Captured:
- SESSION_START/END - Chain lifecycle
- MODEL_INVOCATION_START/END - LLM call tracking
- MCP_CALL_START/END - MCP protocol tool calls with server metadata

Requirements:
    pip install chaukas-sdk langchain langchain-openai

Environment Variables:
    OPENAI_API_KEY - Your OpenAI API key
    CHAUKAS_OUTPUT_MODE - Set to "file" for local testing
    CHAUKAS_OUTPUT_FILE - Path to output file (e.g., "mcp_events.jsonl")
"""

import os
import sys

# For local testing, output events to a file
os.environ.setdefault("CHAUKAS_OUTPUT_MODE", "file")
os.environ.setdefault("CHAUKAS_OUTPUT_FILE", "mcp_events.jsonl")
os.environ.setdefault("CHAUKAS_TENANT_ID", "demo-tenant")
os.environ.setdefault("CHAUKAS_PROJECT_ID", "langchain-mcp-demo")

from chaukas import sdk as chaukas
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Try new langgraph-based agents first (langchain >= 1.0), fallback to legacy
USE_LANGGRAPH = False
try:
    from langgraph.prebuilt import create_react_agent
    USE_LANGGRAPH = True
except ImportError:
    try:
        from langchain.agents import AgentExecutor, create_tool_calling_agent
    except ImportError:
        print("Error: Agent functionality not available.")
        print("For langchain >= 1.0: pip install langgraph")
        print("For langchain < 1.0: pip install langchain==0.2.16")
        sys.exit(1)

# Enable Chaukas instrumentation - one line setup!
chaukas.enable_chaukas()


# Define tools that simulate MCP server interactions
@tool
def fetch_documentation(query: str) -> str:
    """Fetch documentation from the documentation MCP server."""
    # Simulated MCP server response
    docs = {
        "chaukas": "Chaukas SDK provides one-line observability for AI agents.",
        "langchain": "LangChain is a framework for building LLM applications.",
        "mcp": "Model Context Protocol enables standardized model-tool communication.",
    }
    query_lower = query.lower()
    for key, value in docs.items():
        if key in query_lower:
            return value
    return f"No documentation found for: {query}"


@tool
def search_knowledge_base(topic: str) -> str:
    """Search the knowledge base MCP server for information."""
    # Simulated knowledge base response
    knowledge = {
        "observability": "Observability includes metrics, logs, and traces for system monitoring.",
        "agents": "AI agents are autonomous systems that can perceive, reason, and act.",
        "instrumentation": "Instrumentation adds monitoring capabilities to applications.",
    }
    topic_lower = topic.lower()
    for key, value in knowledge.items():
        if key in topic_lower:
            return value
    return f"No knowledge found about: {topic}"


def _extract_output(result):
    """Extract output from agent result (handles both old and new API)."""
    if isinstance(result, dict):
        if "output" in result:
            return result["output"]
        if "messages" in result and result["messages"]:
            last_msg = result["messages"][-1]
            return last_msg.content if hasattr(last_msg, "content") else str(last_msg)
    return str(result)


def main():
    """Demonstrate MCP tool call capture with Chaukas."""
    print("=" * 60)
    print("LangChain MCP Tools Example with Chaukas Instrumentation")
    print("=" * 60)
    print()

    # Create tools list
    tools = [fetch_documentation, search_knowledge_base]

    # Create the agent
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    if USE_LANGGRAPH:
        # New langgraph-based agent (langchain >= 1.0)
        agent = create_react_agent(llm, tools)

        # Example 1: Invoke with MCP metadata - triggers MCP_CALL events
        print("Example 1: Tool call WITH MCP metadata (MCP_CALL events)")
        print("-" * 50)
        print("Using mcp_server='docs-server' in config metadata...")
        print()

        result = agent.invoke(
            {"messages": [("human", "What is Chaukas SDK?")]},
            config={
                "metadata": {
                    "mcp_server": "docs-server",
                    "mcp_server_url": "mcp://docs.example.com",
                }
            },
        )
        print(f"Result: {_extract_output(result)}")
        print()

        # Example 2: Invoke without MCP metadata - triggers regular TOOL_CALL events
        print("Example 2: Tool call WITHOUT MCP metadata (TOOL_CALL events)")
        print("-" * 50)
        print("No mcp_server in config - standard tool call events...")
        print()

        result = agent.invoke(
            {"messages": [("human", "Tell me about observability in AI systems.")]},
        )
        print(f"Result: {_extract_output(result)}")
        print()

        # Example 3: Multiple tools with MCP metadata
        print("Example 3: Multiple tool calls with MCP metadata")
        print("-" * 50)

        result = agent.invoke(
            {"messages": [("human", "What is LangChain and how does it relate to AI agents?")]},
            config={
                "metadata": {
                    "mcp_server": "knowledge-server",
                    "mcp_server_url": "mcp://knowledge.example.com",
                }
            },
        )
        print(f"Result: {_extract_output(result)}")
        print()

    else:
        # Legacy AgentExecutor (langchain < 1.0)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant that uses tools to find information. "
                    "Always use the available tools to answer questions.",
                ),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        # Example 1: Invoke with MCP metadata - triggers MCP_CALL events
        print("Example 1: Tool call WITH MCP metadata (MCP_CALL events)")
        print("-" * 50)
        print("Using mcp_server='docs-server' in config metadata...")
        print()

        result = agent_executor.invoke(
            {"input": "What is Chaukas SDK?"},
            config={
                "metadata": {
                    "mcp_server": "docs-server",
                    "mcp_server_url": "mcp://docs.example.com",
                }
            },
        )
        print(f"Result: {result['output']}")
        print()

        # Example 2: Invoke without MCP metadata - triggers regular TOOL_CALL events
        print("Example 2: Tool call WITHOUT MCP metadata (TOOL_CALL events)")
        print("-" * 50)
        print("No mcp_server in config - standard tool call events...")
        print()

        result = agent_executor.invoke(
            {"input": "Tell me about observability in AI systems."},
        )
        print(f"Result: {result['output']}")
        print()

        # Example 3: Multiple tools with MCP metadata
        print("Example 3: Multiple tool calls with MCP metadata")
        print("-" * 50)

        result = agent_executor.invoke(
            {"input": "What is LangChain and how does it relate to AI agents?"},
            config={
                "metadata": {
                    "mcp_server": "knowledge-server",
                    "mcp_server_url": "mcp://knowledge.example.com",
                }
            },
        )
        print(f"Result: {result['output']}")
        print()

    print("Check mcp_events.jsonl for captured events.")
    print()
    print("MCP events include:")
    print("  - server_name: from mcp_server metadata")
    print("  - server_url: from mcp_server_url metadata")
    print("  - protocol_version: '1.0'")
    print("  - operation: 'call_tool'")
    print()
    print("Compare MCP_CALL_START/END events (with metadata)")
    print("vs TOOL_CALL_START/END events (without metadata)")


if __name__ == "__main__":
    main()
