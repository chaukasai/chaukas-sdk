"""
Comprehensive LangChain example demonstrating all supported events with real API calls.

This example showcases Chaukas instrumentation for LangChain with 17/19 event coverage:

Session & Agent Events:
- SESSION_START/END - Root chain lifecycle management
- AGENT_START/END - Agent execution tracking
- AGENT_HANDOFF - Multi-agent collaboration (via context passing)

Model Operations:
- MODEL_INVOCATION_START/END - LLM/chat model calls with token tracking

Tool & MCP Events:
- TOOL_CALL_START/END - Tool execution monitoring
- MCP_CALL_START/END - Model Context Protocol calls (when mcp_server metadata present)

I/O & Data Events:
- INPUT_RECEIVED - User input capture
- OUTPUT_EMITTED - Agent responses and streaming output (via on_text)
- DATA_ACCESS - Retriever and vector store operations

Error & Operational Events:
- ERROR - Error tracking with context
- RETRY - Automatic retry detection
- SYSTEM - Custom events (via on_custom_event)

Not captured (require application-level instrumentation):
- POLICY_DECISION
- STATE_UPDATE

Requires:
- OPENAI_API_KEY environment variable
- pip install chaukas-sdk langchain langchain-openai langchain-community faiss-cpu

Usage:
    python langchain_comprehensive_example.py          # Interactive menu
    python langchain_comprehensive_example.py --all    # Run all scenarios
    python langchain_comprehensive_example.py --help   # Show help
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# Get script directory for consistent file paths
SCRIPT_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = SCRIPT_DIR / "output"

# Load .env from the script directory first (to get CHAUKAS_OUTPUT_FILE base name)
try:
    from dotenv import load_dotenv

    load_dotenv(SCRIPT_DIR / ".env")
except ImportError:
    pass

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(exist_ok=True)

# Get base name from env (without extension), construct timestamped path
# .env should have: CHAUKAS_OUTPUT_FILE=langchain_output (no extension)
OUTPUT_BASE = os.environ.get("CHAUKAS_OUTPUT_FILE", "langchain_output")
# Strip any extension if accidentally provided
OUTPUT_BASE = OUTPUT_BASE.rsplit(".", 1)[0] if "." in OUTPUT_BASE else OUTPUT_BASE
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FILE = str(OUTPUT_DIR / f"{OUTPUT_BASE}_{TIMESTAMP}.jsonl")

# Validate OpenAI API key
if not os.environ.get("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY environment variable is required")
    print("Set it with: export OPENAI_API_KEY='your-key'")
    sys.exit(1)

# Configure Chaukas for file output (for demo visibility)
os.environ["CHAUKAS_OUTPUT_MODE"] = "file"
os.environ["CHAUKAS_OUTPUT_FILE"] = OUTPUT_FILE
os.environ.setdefault("CHAUKAS_TENANT_ID", "demo-tenant")
os.environ.setdefault("CHAUKAS_PROJECT_ID", "langchain-comprehensive-demo")
os.environ.setdefault("CHAUKAS_BATCH_SIZE", "1")  # Immediate writes for demo

# Import and enable Chaukas
from chaukas import sdk as chaukas

chaukas.enable_chaukas()

# Import event analysis tool (shared with OpenAI and CrewAI examples)
sys.path.insert(0, str(SCRIPT_DIR.parent.parent))
from tools.summarize_event_stats import summarize_event_stats

# Import LangChain components with helpful error messages
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except ImportError:
    print("Error: langchain-openai is not installed")
    print("Install with: pip install langchain-openai")
    sys.exit(1)

try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.tools import tool
    from langchain_core.documents import Document
    from langchain_core.runnables import RunnableLambda, RunnablePassthrough
except ImportError:
    print("Error: langchain-core is not installed")
    print("Install with: pip install langchain langchain-core")
    sys.exit(1)

# Try new langgraph-based agents first (langchain >= 1.0), fallback to legacy
AGENTS_AVAILABLE = False
USE_LANGGRAPH = False

try:
    from langgraph.prebuilt import create_react_agent
    AGENTS_AVAILABLE = True
    USE_LANGGRAPH = True
except ImportError:
    pass

if not AGENTS_AVAILABLE:
    try:
        from langchain.agents import AgentExecutor, create_tool_calling_agent
        AGENTS_AVAILABLE = True
        USE_LANGGRAPH = False
    except ImportError:
        pass

if not AGENTS_AVAILABLE:
    print("Warning: Agent functionality not available.")
    print("For langchain >= 1.0: pip install langgraph")
    print("For langchain < 1.0: pip install langchain==0.2.16")
    print("Agent scenarios will be skipped.")

try:
    from langchain_community.vectorstores import FAISS
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: langchain-community or faiss-cpu not installed")
    print("RAG scenario will be skipped. Install with: pip install langchain-community faiss-cpu")

# Optional imports for custom events
try:
    from langchain_core.callbacks.manager import dispatch_custom_event

    CUSTOM_EVENTS_AVAILABLE = True
except ImportError:
    CUSTOM_EVENTS_AVAILABLE = False


# =============================================================================
# Tool Definitions
# =============================================================================


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Supports basic operations (+, -, *, /, **)."""
    try:
        # Safe evaluation of math expressions
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def word_counter(text: str) -> str:
    """Count words, characters, and sentences in text."""
    words = len(text.split())
    chars = len(text)
    sentences = text.count(".") + text.count("!") + text.count("?")
    return f"Words: {words}, Characters: {chars}, Sentences: {sentences}"


@tool
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for information about AI and observability."""
    knowledge = {
        "chaukas": "Chaukas SDK provides one-line observability for AI agents with 19 event types.",
        "langchain": "LangChain is a framework for building LLM-powered applications with chains and agents.",
        "observability": "Observability includes metrics, logs, and distributed traces for system monitoring.",
        "mcp": "Model Context Protocol (MCP) enables standardized communication between models and tools.",
    }
    query_lower = query.lower()
    for key, value in knowledge.items():
        if key in query_lower:
            return value
    return f"No specific knowledge found for: {query}. Try searching for: chaukas, langchain, observability, or mcp."


# =============================================================================
# Scenario 1: Simple Chain
# =============================================================================


def scenario_simple_chain():
    """
    Scenario 1: Basic Chain

    Demonstrates:
    - SESSION_START/END
    - MODEL_INVOCATION_START/END
    - INPUT_RECEIVED/OUTPUT_EMITTED
    """
    print("\n" + "=" * 60)
    print("Scenario 1: Simple Chain")
    print("=" * 60)
    print("Events: SESSION_START/END, MODEL_INVOCATION_START/END, I/O")
    print("-" * 60)

    prompt = ChatPromptTemplate.from_template(
        "Tell me one interesting fact about {topic} in 2 sentences."
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    chain = prompt | llm | StrOutputParser()

    topics = ["quantum computing", "octopuses"]

    for topic in topics:
        print(f"\nTopic: {topic}")
        try:
            result = chain.invoke({"topic": topic})
            print(f"Result: {result[:150]}..." if len(result) > 150 else f"Result: {result}")
        except Exception as e:
            print(f"Error: {e}")


# =============================================================================
# Scenario 2: Agent with Tools
# =============================================================================


def scenario_agent_tools():
    """
    Scenario 2: Agent with Tools

    Demonstrates:
    - AGENT_START/END
    - TOOL_CALL_START/END
    - MODEL_INVOCATION_START/END
    """
    print("\n" + "=" * 60)
    print("Scenario 2: Agent with Tools")
    print("=" * 60)
    print("Events: AGENT_START/END, TOOL_CALL_START/END, MODEL_INVOCATION")
    print("-" * 60)

    if not AGENTS_AVAILABLE:
        print("\nSkipping: Agent functionality not available.")
        print("Install with: pip install langgraph")
        return

    tools = [calculator, word_counter]
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    queries = [
        "What is 25 * 17 + 83?",
        "Count the words in: 'The quick brown fox jumps over the lazy dog.'",
    ]

    if USE_LANGGRAPH:
        # New langgraph-based agent (langchain >= 1.0)
        agent = create_react_agent(llm, tools)
        for query in queries:
            print(f"\nQuery: {query}")
            try:
                result = agent.invoke({"messages": [("human", query)]})
                # Extract the last message content
                if "messages" in result and result["messages"]:
                    last_msg = result["messages"][-1]
                    output = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
                else:
                    output = str(result)
                print(f"Result: {output[:150]}..." if len(output) > 150 else f"Result: {output}")
            except Exception as e:
                print(f"Error: {e}")
    else:
        # Legacy AgentExecutor (langchain < 1.0)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant. Use tools when needed."),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )
        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

        for query in queries:
            print(f"\nQuery: {query}")
            try:
                result = agent_executor.invoke({"input": query})
                output = result.get("output", str(result))
                print(f"Result: {output[:150]}..." if len(output) > 150 else f"Result: {output}")
            except Exception as e:
                print(f"Error: {e}")


# =============================================================================
# Scenario 3: RAG with Retriever
# =============================================================================


def scenario_rag_retriever():
    """
    Scenario 3: RAG with Retriever

    Demonstrates:
    - DATA_ACCESS events
    - Retriever tracking
    - Document context
    """
    print("\n" + "=" * 60)
    print("Scenario 3: RAG with Retriever")
    print("=" * 60)
    print("Events: DATA_ACCESS, SESSION, MODEL_INVOCATION")
    print("-" * 60)

    if not FAISS_AVAILABLE:
        print("\nSkipping: FAISS not available.")
        print("Install with: pip install langchain-community faiss-cpu")
        return

    # Create sample documents about Chaukas
    documents = [
        Document(
            page_content="Chaukas SDK is an open-source observability solution for AI agents. "
            "It provides comprehensive instrumentation with just one line of code.",
            metadata={"source": "docs", "topic": "overview"},
        ),
        Document(
            page_content="Chaukas supports LangChain, CrewAI, and OpenAI Agents frameworks. "
            "It captures 17-19 event types depending on the framework.",
            metadata={"source": "docs", "topic": "frameworks"},
        ),
        Document(
            page_content="Events captured include SESSION_START, AGENT_START, MODEL_INVOCATION, "
            "TOOL_CALL, DATA_ACCESS, ERROR, RETRY, and more.",
            metadata={"source": "docs", "topic": "events"},
        ),
    ]

    # Create vector store
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    splits = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(splits, OpenAIEmbeddings())
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # Create RAG chain
    prompt = ChatPromptTemplate.from_template(
        """Answer based on the context below:

Context: {context}

Question: {question}

Answer:"""
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    questions = [
        "What is Chaukas SDK?",
        "What event types does Chaukas capture?",
    ]

    for question in questions:
        print(f"\nQuestion: {question}")
        try:
            result = rag_chain.invoke(question)
            print(f"Answer: {result[:150]}..." if len(result) > 150 else f"Answer: {result}")
        except Exception as e:
            print(f"Error: {e}")


# =============================================================================
# Scenario 4: Streaming Output
# =============================================================================


def scenario_streaming():
    """
    Scenario 4: Streaming Output

    Demonstrates:
    - OUTPUT_EMITTED via on_text callback
    - Real-time streaming capture
    - Streaming metadata
    """
    print("\n" + "=" * 60)
    print("Scenario 4: Streaming Output")
    print("=" * 60)
    print("Events: OUTPUT_EMITTED (streaming), MODEL_INVOCATION, SESSION")
    print("-" * 60)

    prompt = ChatPromptTemplate.from_template(
        "Write a haiku about {topic}. Just the haiku, no explanation."
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.8, streaming=True)
    chain = prompt | llm | StrOutputParser()

    topics = ["coding", "coffee"]

    for topic in topics:
        print(f"\nStreaming haiku about '{topic}':")
        try:
            for chunk in chain.stream({"topic": topic}):
                print(chunk, end="", flush=True)
            print()
        except Exception as e:
            print(f"Error: {e}")


# =============================================================================
# Scenario 5: Custom Events
# =============================================================================


def scenario_custom_events():
    """
    Scenario 5: Custom Events

    Demonstrates:
    - SYSTEM events via on_custom_event
    - Custom severity levels
    - Application-specific tracking

    Note: dispatch_custom_event must be called from within a Runnable context.
    """
    print("\n" + "=" * 60)
    print("Scenario 5: Custom Events")
    print("=" * 60)
    print("Events: SYSTEM (custom events), MODEL_INVOCATION, SESSION")
    print("-" * 60)

    if not CUSTOM_EVENTS_AVAILABLE:
        print("Custom events require langchain-core >= 0.2.15")
        print("Skipping this scenario.")
        return

    def processing_with_events(input_data: dict) -> dict:
        """Process input and emit custom events."""
        text = input_data.get("text", "")

        # Emit start event
        dispatch_custom_event(
            "processing_started",
            {"input_length": len(text)},
        )

        # Emit validation event
        if len(text) < 10:
            dispatch_custom_event(
                "validation_warning",
                {"issue": "short_input", "length": len(text)},
                metadata={"severity": "warn"},
            )
        else:
            dispatch_custom_event(
                "validation_passed",
                {"length": len(text)},
                metadata={"severity": "info"},
            )

        # Emit completion event
        dispatch_custom_event(
            "processing_complete",
            {"status": "success"},
            metadata={"severity": "debug"},
        )

        return input_data

    process_runnable = RunnableLambda(processing_with_events)
    prompt = ChatPromptTemplate.from_template("Summarize in one sentence: {text}")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = process_runnable | prompt | llm | StrOutputParser()

    inputs = [
        {"text": "Short"},  # Will trigger warning
        {
            "text": "Chaukas SDK provides comprehensive observability for AI agents with just one line of code."
        },  # Normal
    ]

    for input_data in inputs:
        print(f"\nInput: {input_data['text'][:50]}...")
        try:
            result = chain.invoke(input_data)
            print(f"Result: {result}")
        except Exception as e:
            print(f"Error: {e}")


# =============================================================================
# Scenario 6: MCP Tools
# =============================================================================


def scenario_mcp_tools():
    """
    Scenario 6: MCP Tools

    Demonstrates:
    - MCP_CALL_START/END when mcp_server metadata is present
    - Protocol version tracking
    - Server metadata

    Note: MCP events are triggered by including 'mcp_server' in config metadata.
    """
    print("\n" + "=" * 60)
    print("Scenario 6: MCP Tools")
    print("=" * 60)
    print("Events: MCP_CALL_START/END, TOOL_CALL, MODEL_INVOCATION")
    print("-" * 60)

    if not AGENTS_AVAILABLE:
        print("\nSkipping: Agent functionality not available.")
        print("Install with: pip install langgraph")
        return

    tools = [search_knowledge_base]
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    if USE_LANGGRAPH:
        # New langgraph-based agent (langchain >= 1.0)
        agent = create_react_agent(llm, tools)

        # With MCP metadata - triggers MCP_CALL events
        print("\nWith MCP metadata (MCP_CALL events):")
        try:
            result = agent.invoke(
                {"messages": [("human", "What is Chaukas?")]},
                config={
                    "metadata": {
                        "mcp_server": "knowledge-server",
                        "mcp_server_url": "mcp://knowledge.local",
                    }
                },
            )
            if "messages" in result and result["messages"]:
                last_msg = result["messages"][-1]
                output = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
            else:
                output = str(result)
            print(f"Result: {output[:150]}..." if len(output) > 150 else f"Result: {output}")
        except Exception as e:
            print(f"Error: {e}")

        # Without MCP metadata - triggers regular TOOL_CALL events
        print("\nWithout MCP metadata (TOOL_CALL events):")
        try:
            result = agent.invoke({"messages": [("human", "Tell me about MCP protocol.")]})
            if "messages" in result and result["messages"]:
                last_msg = result["messages"][-1]
                output = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
            else:
                output = str(result)
            print(f"Result: {output[:150]}..." if len(output) > 150 else f"Result: {output}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        # Legacy AgentExecutor (langchain < 1.0)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant. Use the search_knowledge_base tool to find information.",
                ),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )
        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

        # With MCP metadata - triggers MCP_CALL events
        print("\nWith MCP metadata (MCP_CALL events):")
        try:
            result = agent_executor.invoke(
                {"input": "What is Chaukas?"},
                config={
                    "metadata": {
                        "mcp_server": "knowledge-server",
                        "mcp_server_url": "mcp://knowledge.local",
                    }
                },
            )
            output = result.get("output", str(result))
            print(f"Result: {output[:150]}..." if len(output) > 150 else f"Result: {output}")
        except Exception as e:
            print(f"Error: {e}")

        # Without MCP metadata - triggers regular TOOL_CALL events
        print("\nWithout MCP metadata (TOOL_CALL events):")
        try:
            result = agent_executor.invoke({"input": "Tell me about MCP protocol."})
            output = result.get("output", str(result))
            print(f"Result: {output[:150]}..." if len(output) > 150 else f"Result: {output}")
        except Exception as e:
            print(f"Error: {e}")


# =============================================================================
# Scenario 7: Error Handling
# =============================================================================


def scenario_error_handling():
    """
    Scenario 7: Error Handling

    Demonstrates:
    - ERROR events
    - RETRY events (when applicable)
    - Error context and recovery
    """
    print("\n" + "=" * 60)
    print("Scenario 7: Error Handling")
    print("=" * 60)
    print("Events: ERROR, RETRY (if applicable)")
    print("-" * 60)

    if not AGENTS_AVAILABLE:
        print("\nSkipping: Agent functionality not available.")
        print("Install with: pip install langgraph")
        return

    @tool
    def failing_tool(query: str) -> str:
        """A tool that always fails for demonstration."""
        raise ValueError(f"Simulated error for query: {query}")

    tools = [failing_tool, calculator]
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    print("\nAttempting to use failing tool:")

    if USE_LANGGRAPH:
        # New langgraph-based agent (langchain >= 1.0)
        agent = create_react_agent(llm, tools)
        try:
            result = agent.invoke({"messages": [("human", "Use the failing_tool with query 'test'")]})
            if "messages" in result and result["messages"]:
                last_msg = result["messages"][-1]
                output = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
            else:
                output = str(result)
            print(f"Result: {output[:150]}..." if len(output) > 150 else f"Result: {output}")
        except Exception as e:
            print(f"Caught error (expected): {type(e).__name__}: {str(e)[:100]}")
    else:
        # Legacy AgentExecutor (langchain < 1.0)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant. Try using the failing_tool first."),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )
        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=3,
        )

        try:
            result = agent_executor.invoke({"input": "Use the failing_tool with query 'test'"})
            output = result.get("output", str(result))
            print(f"Result: {output[:150]}..." if len(output) > 150 else f"Result: {output}")
        except Exception as e:
            print(f"Caught error (expected): {type(e).__name__}: {str(e)[:100]}")


# =============================================================================
# Event Analysis
# =============================================================================


def analyze_events():
    """Analyze captured events using the shared summarize_event_stats tool."""
    if not os.path.exists(OUTPUT_FILE):
        print(f"\nNo events file found at {OUTPUT_FILE}")
        print("Run some scenarios first to capture events.")
        return

    try:
        summarize_event_stats(OUTPUT_FILE, title="LangChain Event Analysis")
    except FileNotFoundError:
        print(f"\nNo events file found at {OUTPUT_FILE}")
        print("Run some scenarios first to capture events.")


# =============================================================================
# Main Program
# =============================================================================


def show_menu():
    """Display the interactive menu."""
    print("\n" + "=" * 60)
    print("LangChain Comprehensive Example with Chaukas")
    print("=" * 60)
    print()
    print("Select a scenario to run:")
    print()
    print("  1. Simple Chain        (SESSION, MODEL_INVOCATION, I/O)")
    print("  2. Agent with Tools    (TOOL_CALL, AGENT)")
    print("  3. RAG Retriever       (DATA_ACCESS)")
    print("  4. Streaming Output    (OUTPUT_EMITTED)")
    print("  5. Custom Events       (SYSTEM)")
    print("  6. MCP Tools           (MCP_CALL)")
    print("  7. Error Handling      (ERROR, RETRY)")
    print()
    print("  8. Run ALL scenarios")
    print("  9. Analyze captured events")
    print("  0. Exit")
    print()


def run_all_scenarios():
    """Run all scenarios in sequence."""
    scenarios = [
        scenario_simple_chain,
        scenario_agent_tools,
        scenario_rag_retriever,
        scenario_streaming,
        scenario_custom_events,
        scenario_mcp_tools,
        scenario_error_handling,
    ]

    for scenario in scenarios:
        try:
            scenario()
        except KeyboardInterrupt:
            print("\nInterrupted. Moving to next scenario...")
        except Exception as e:
            print(f"Error in scenario: {e}")

    print("\n" + "=" * 60)
    print("All scenarios completed!")
    print(f"Events saved to: {OUTPUT_FILE}")
    print("=" * 60)


def main():
    """Main entry point with interactive menu."""
    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] in ["--help", "-h"]:
            print(__doc__)
            print("\nOptions:")
            print("  --all     Run all scenarios")
            print("  --help    Show this help message")
            return
        elif sys.argv[1] == "--all":
            run_all_scenarios()
            analyze_events()
            return

    # Interactive menu
    scenarios = {
        "1": scenario_simple_chain,
        "2": scenario_agent_tools,
        "3": scenario_rag_retriever,
        "4": scenario_streaming,
        "5": scenario_custom_events,
        "6": scenario_mcp_tools,
        "7": scenario_error_handling,
    }

    while True:
        show_menu()
        try:
            choice = input("Enter choice (0-9): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if choice == "0":
            print("\nGoodbye!")
            break
        elif choice == "8":
            run_all_scenarios()
        elif choice == "9":
            analyze_events()
        elif choice in scenarios:
            try:
                scenarios[choice]()
            except KeyboardInterrupt:
                print("\nScenario interrupted.")
            except Exception as e:
                print(f"Error: {e}")
        else:
            print("Invalid choice. Please enter 0-9.")

    # Cleanup
    print(f"\nEvents saved to: {OUTPUT_FILE}")
    chaukas.disable_chaukas()


if __name__ == "__main__":
    main()
