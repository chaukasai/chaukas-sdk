# LangChain / LangGraph Examples with Chaukas Instrumentation

This directory contains examples demonstrating how to use Chaukas SDK with LangChain and LangGraph for comprehensive observability.

> **Note:** LangGraph uses the same integration as LangChain - no separate setup needed. LangGraph is built on LangChain's `Runnable` interface, so when you call `enable_chaukas()`, both LangChain chains and LangGraph graphs are automatically instrumented.

## Event Coverage

LangChain integration with Chaukas provides **17/19 event coverage (89%)**, capturing:

- **SESSION_START/END** - Root chain lifecycle
- **AGENT_START/END** - Agent chain execution
- **MODEL_INVOCATION_START/END** - LLM/chat model calls
- **TOOL_CALL_START/END** - Tool execution
- **MCP_CALL_START/END** - MCP protocol tool calls (when `mcp_server` metadata is present)
- **DATA_ACCESS** - Retriever/vector store queries
- **INPUT_RECEIVED/OUTPUT_EMITTED** - I/O tracking (including streaming via `on_text`)
- **ERROR** - Error handling
- **RETRY** - Retry detection
- **AGENT_HANDOFF** - Multi-agent workflows
- **SYSTEM** - Custom events (via `on_custom_event`)

## Supported Versions

| Package | Supported Versions |
|---------|-------------------|
| `langchain` | `>=0.1.0,<2.0` |
| `langchain-core` | `>=0.1.0,<2.0` |
| `langchain-openai` | `>=0.0.1,<2.0` |
| `langchain-community` | `>=0.0.1,<2.0` |

**Python Version Note:** LangChain 1.x requires Python ≥3.10. LangChain 0.x supports Python ≥3.9.

## Setup

1. Install required packages:
```bash
pip install chaukas-sdk langchain langchain-openai langchain-community faiss-cpu
```

2. Set environment variables:
```bash
export OPENAI_API_KEY="your-openai-key"
export CHAUKAS_ENDPOINT="https://api.chaukas.ai"
export CHAUKAS_API_KEY="your-chaukas-key"
export CHAUKAS_TENANT_ID="your-tenant-id"
export CHAUKAS_PROJECT_ID="your-project-id"
```

## Examples

### Comprehensive Example (Recommended)

**`langchain_comprehensive_example.py`** - Interactive example demonstrating all 17 event types.

This is the best way to explore Chaukas + LangChain integration. It includes:
- 7 scenarios covering all supported event types
- Interactive menu to run individual scenarios
- Built-in event analysis to see what was captured
- `--all` flag to run all scenarios at once

**Usage:**
```bash
# Interactive menu
python langchain_comprehensive_example.py

# Run all scenarios
python langchain_comprehensive_example.py --all

# Show help
python langchain_comprehensive_example.py --help
```

**Scenarios:**
1. Simple Chain (SESSION, MODEL_INVOCATION, I/O)
2. Agent with Tools (TOOL_CALL, AGENT)
3. RAG Retriever (DATA_ACCESS)
4. Streaming Output (OUTPUT_EMITTED)
5. Custom Events (SYSTEM)
6. MCP Tools (MCP_CALL)
7. Error Handling (ERROR, RETRY)

---

### Focused Examples

The following examples focus on specific features. Use these for quick reference on individual topics.

### 1. Simple Chain (`simple_chain.py`)
Basic LangChain chain with prompt → LLM → parser pattern.

**Key Events Captured:**
- SESSION_START/END
- MODEL_INVOCATION_START/END
- INPUT_RECEIVED/OUTPUT_EMITTED

**Usage:**
```bash
python simple_chain.py
```

### 2. Agent with Tools (`agent_with_tools.py`)
LangChain agent that can use custom tools (calculator, word counter).

**Key Events Captured:**
- AGENT_START/END
- TOOL_CALL_START/END
- MODEL_INVOCATION_START/END
- All standard chain events

**Usage:**
```bash
python agent_with_tools.py
```

### 3. RAG with Retriever (`rag_with_retriever.py`)
Retrieval-Augmented Generation using vector store and retriever.

**Key Events Captured:**
- DATA_ACCESS (retriever calls)
- MODEL_INVOCATION_START/END
- SESSION_START/END
- Context and document tracking

**Usage:**
```bash
python rag_with_retriever.py
```

### 4. Multi-Chain Workflow (`multi_chain_workflow.py`)
Complex workflow with multiple sequential chains.

**Key Events Captured:**
- Multiple SESSION events
- Complete event hierarchy
- Inter-chain dependencies

**Usage:**
```bash
python multi_chain_workflow.py
```

### 5. Streaming Output (`streaming_example.py`)
Real-time streaming with text chunk capture via `on_text` callback.

**Key Events Captured:**
- OUTPUT_EMITTED (multiple events, one per text chunk)
- MODEL_INVOCATION_START/END
- SESSION_START/END

**Usage:**
```bash
python streaming_example.py
```

### 6. Custom Events (`custom_events_example.py`)
Application-specific custom event tracking using `dispatch_custom_event`.

**Key Events Captured:**
- SYSTEM events (via `on_custom_event` callback)
- Custom severity levels (debug, info, warn, error)
- Application-specific metadata

**Usage:**
```bash
python custom_events_example.py
```

**Note:** `dispatch_custom_event` must be called from within a Runnable context (e.g., inside a `RunnableLambda` function).

### 7. MCP Tools (`mcp_tools_example.py`)
Model Context Protocol tool integration with server metadata.

**Key Events Captured:**
- MCP_CALL_START/END (when `mcp_server` metadata is present)
- Protocol version and server tracking
- Server name and URL metadata

**Usage:**
```bash
python mcp_tools_example.py
```

**Note:** To trigger MCP events, include `mcp_server` in the config metadata:
```python
result = chain.invoke(
    {"input": "..."},
    config={"metadata": {"mcp_server": "server-name", "mcp_server_url": "mcp://..."}}
)
```

## Usage Pattern

All examples follow the same simple pattern:

```python
from chaukas import sdk as chaukas

# Enable Chaukas instrumentation (one-line setup!)
chaukas.enable_chaukas()

# That's it! All LangChain operations are now automatically tracked
result = chain.invoke(input_data)
```

Chaukas automatically patches LangChain's Runnable methods to inject callbacks, so you don't need to manually pass callbacks anymore!

## Key Differences from Other Integrations

All three major frameworks now have the same one-line setup:

- **OpenAI Agents**: Automatic via method patching - `chaukas.enable_chaukas()` is enough
- **CrewAI**: Automatic via event bus hooking - `chaukas.enable_chaukas()` is enough
- **LangChain**: Automatic via Runnable method patching - `chaukas.enable_chaukas()` is enough ✨

Chaukas automatically patches LangChain's Runnable methods to inject callbacks, providing the same seamless one-line setup as other frameworks.

### Manual Callback Usage (Optional)

If you need more control or want to use callbacks for specific operations only, you can still get the callback manually:

```python
import chaukas

chaukas.enable_chaukas()
callback = chaukas.get_langchain_callback()

# Use only for specific operations
result = chain.invoke(input, config={"callbacks": [callback]})
```

## Viewing Events

Events are sent to your Chaukas dashboard in real-time. You can view:

1. **Trace View**: Complete execution trace with nested spans
2. **Timeline**: Chronological event flow
3. **Metrics**: Token usage, latency, error rates
4. **Agent View**: Agent-specific insights

## Troubleshooting

### No Events Appearing
**Problem**: No events appearing in dashboard

**Solution**: Verify that:
1. `enable_chaukas()` was called before any LangChain operations
2. LangChain is properly installed (`pip install langchain langchain-core`)
3. Your environment variables are set correctly (CHAUKAS_API_KEY, etc.)

### LangChain Version Issues
**Problem**: Auto-instrumentation not working

**Solution**: Chaukas supports LangChain >= 0.1.0. If you have an older version, either:
1. Upgrade LangChain: `pip install --upgrade langchain langchain-core`
2. Use manual callback passing:
   ```python
   callback = chaukas.get_langchain_callback()
   chain.invoke(input, config={"callbacks": [callback]})
   ```

### Python Version Requirements
**Problem**: Installation fails or import errors with LangChain 1.x

**Solution**: LangChain 1.x requires Python ≥3.10. Check your Python version:
```bash
python --version
```

| LangChain Version | Python Requirement |
|-------------------|-------------------|
| 0.1.x - 0.3.x | Python ≥3.9 |
| 1.x (1.0.0+) | Python ≥3.10 |

If you're on Python 3.9, you'll need to use LangChain 0.x versions:
```bash
pip install "langchain>=0.1.0,<1.0"
```

### Events Only for Some Operations
**Problem**: Some LangChain operations aren't tracked

**Solution**: This is expected - Chaukas tracks the main event types (chains, LLMs, tools, retrievers). If you need custom event tracking, you can extend the callback handler.

### Import Errors
**Problem**: `ImportError: No module named 'langchain'`

**Solution**: Install LangChain packages:
```bash
pip install langchain langchain-openai langchain-community
```

## Advanced Usage

### Custom Tools with Data Access Tracking
```python
from langchain_core.tools import tool

@tool
def search_database(query: str) -> str:
    """Search the database for information."""
    # Your database logic here
    return results

# The callback automatically tracks tool execution
# Including DATA_ACCESS events if your tool accesses data stores
```

### Streaming Support
```python
# Streaming is supported - events are captured during streaming
# The on_text callback emits OUTPUT_EMITTED events for streamed text chunks
for chunk in chain.stream(input, config={"callbacks": [callback]}):
    print(chunk, end="", flush=True)
```

### Async Support
```python
import asyncio

async def main():
    result = await chain.ainvoke(
        input,
        config={"callbacks": [callback]}
    )

asyncio.run(main())
```

## Next Steps

- Check out the [Chaukas Dashboard](https://app.chaukas.ai) to view your traces
- Explore [LangChain documentation](https://python.langchain.com/docs/get_started/introduction) for more patterns
- Read the [Chaukas SDK documentation](https://docs.chaukas.ai) for advanced features

## Support

For issues or questions:
- GitHub: https://github.com/chaukasai/chaukas-sdk/issues
- Documentation: https://docs.chaukas.ai
- Email: support@chaukas.ai
