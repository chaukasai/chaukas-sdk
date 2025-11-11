# LangChain Examples with Chaukas Instrumentation

This directory contains examples demonstrating how to use Chaukas SDK with LangChain for comprehensive observability.

## Event Coverage

LangChain integration with Chaukas provides ~95% event coverage, capturing:

- **SESSION_START/END** - Root chain lifecycle
- **AGENT_START/END** - Agent chain execution
- **MODEL_INVOCATION_START/END** - LLM/chat model calls
- **TOOL_CALL_START/END** - Tool execution
- **DATA_ACCESS** - Retriever/vector store queries
- **INPUT_RECEIVED/OUTPUT_EMITTED** - I/O tracking
- **ERROR** - Error handling
- **RETRY** - Retry detection
- **AGENT_HANDOFF** - Multi-agent workflows

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

## Usage Pattern

All examples follow the same pattern:

```python
import chaukas

# 1. Enable Chaukas instrumentation
chaukas.enable_chaukas()

# 2. Get the callback handler
callback = chaukas.get_langchain_callback()

# 3. Use callback in your chains/agents
result = chain.invoke(
    input_data,
    config={"callbacks": [callback]}
)
```

## Key Differences from Other Integrations

Unlike OpenAI Agents and CrewAI which use automatic patching, LangChain requires **explicit callback passing**:

- **OpenAI Agents**: Automatic via hooks - `chaukas.enable_chaukas()` is enough
- **CrewAI**: Automatic via event bus - `chaukas.enable_chaukas()` is enough
- **LangChain**: Requires callback - must pass `callbacks=[callback]` to invoke/run methods

This is because LangChain uses a callback-based architecture for extensibility.

## Viewing Events

Events are sent to your Chaukas dashboard in real-time. You can view:

1. **Trace View**: Complete execution trace with nested spans
2. **Timeline**: Chronological event flow
3. **Metrics**: Token usage, latency, error rates
4. **Agent View**: Agent-specific insights

## Troubleshooting

### Callback Not Working
**Problem**: No events appearing in dashboard

**Solution**: Make sure you're passing the callback to **all** chain/agent invocations:
```python
# Correct
chain.invoke(input, config={"callbacks": [callback]})

# Incorrect - no events will be captured
chain.invoke(input)
```

### Missing Events
**Problem**: Some events not captured

**Solution**: Ensure callback is passed to **nested** chains too:
```python
# For RAG chains, pass callback to the main chain
rag_chain = {...}
result = rag_chain.invoke(query, config={"callbacks": [callback]})

# The callback will automatically propagate to sub-chains and retrievers
```

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
- GitHub: https://github.com/anthropics/chaukas-sdk/issues
- Documentation: https://docs.chaukas.ai
- Email: support@chaukas.ai
