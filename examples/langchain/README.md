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

All examples follow the same simple pattern:

```python
from chaukas import sdk as chaukas

# Enable Chaukas instrumentation (one-line setup!)
chaukas.enable_chaukas()

# That's it! All LangChain operations are now automatically tracked
result = chain.invoke(input_data)
```

Chaukas automatically injects its callback handler into LangChain's global callback system, so you don't need to manually pass callbacks anymore!

## Key Differences from Other Integrations

All three major frameworks now have the same one-line setup:

- **OpenAI Agents**: Automatic via method patching - `chaukas.enable_chaukas()` is enough
- **CrewAI**: Automatic via event bus hooking - `chaukas.enable_chaukas()` is enough
- **LangChain**: Automatic via global callback injection - `chaukas.enable_chaukas()` is enough ✨

Chaukas automatically injects its callback handler into LangChain's global default callbacks, providing the same seamless one-line setup as other frameworks.

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
