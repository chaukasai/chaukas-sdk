# LangChain / LangGraph Integration

This document provides a comprehensive explanation of how the Chaukas SDK integrates with LangChain and LangGraph, including how each event type is detected and ingested.

## Overview

**LangGraph uses the same integration as LangChain** - no separate setup needed. This is because LangGraph is built on top of LangChain's `Runnable` interface and callback system. When you call `enable_chaukas()`, both LangChain chains and LangGraph graphs are automatically instrumented.

The integration achieves **~95% event coverage** (17/19 event types) with a single line of code:

```python
from chaukas import sdk as chaukas
chaukas.enable_chaukas()
```

---

## Supported Versions

| Package | Supported Versions | Python Requirement |
|---------|-------------------|-------------------|
| `langchain` | `>=0.1.0,<2.0` | See below |
| `langchain-core` | `>=0.1.0,<2.0` | See below |
| `langchain-openai` | `>=0.0.1,<2.0` | |
| `langchain-community` | `>=0.0.1,<2.0` | |

### Python Version Compatibility

| LangChain Version | Python Requirement |
|-------------------|-------------------|
| 0.1.x - 0.3.x | `>=3.9` |
| 1.x (1.0.0+) | `>=3.10.0` |

**Note:** LangChain 1.x dropped support for Python 3.9. If you're using Python 3.9, you'll be limited to LangChain 0.x versions.

The SDK handles API differences between versions gracefully with import fallbacks:

```python
# Automatic version compatibility
try:
    from langchain_core.runnables import Runnable  # langchain >= 0.1
except ImportError:
    from langchain.schema.runnable import Runnable  # older versions

# Agent API compatibility (langchain 1.x vs 0.x)
try:
    from langgraph.prebuilt import create_react_agent  # langchain >= 1.0
except ImportError:
    from langchain.agents import AgentExecutor  # langchain < 1.0
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  User Code: chain.invoke() / agent.invoke()                     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  LangChainWrapper.auto_instrument()                             │
│  - Patches Runnable.invoke() and Runnable.ainvoke()             │
│  - Patches RunnableSequence.invoke() and RunnableSequence.ainvoke()│
│  - Auto-injects ChaukasCallbackHandler into config['callbacks'] │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  ChaukasCallbackHandler (extends BaseCallbackHandler)           │
│  - Receives all LangChain/LangGraph lifecycle callbacks         │
│  - Maps callbacks → Chaukas event types                         │
│  - Tracks span_ids for START/END pairing                        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  EventBuilder.create_*() → Proto Event                          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  ChaukasClient.send_event() → HTTP/File output                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Auto-Instrumentation (One-Line Setup)

When you call `chaukas.enable_chaukas()`, the SDK automatically patches LangChain's core classes:

```python
# src/chaukas/sdk/integrations/langchain.py:61-171
def auto_instrument(self):
    # 1. Import LangChain's Runnable classes
    from langchain_core.runnables import Runnable
    from langchain_core.runnables.base import RunnableSequence

    # 2. Store original methods for later restoration
    self._original_runnable_invoke = Runnable.invoke
    self._original_runnable_ainvoke = Runnable.ainvoke

    # 3. Create wrapped methods that inject the callback
    def wrapped_invoke(self_runnable, input, config=None, **kwargs):
        if config is None:
            config = {}
        if "callbacks" not in config:
            config["callbacks"] = []
        if chaukas_callback not in config["callbacks"]:
            config["callbacks"].append(chaukas_callback)  # ← Auto-inject
        return original_runnable_invoke(self_runnable, input, config=config, **kwargs)

    # 4. Monkey-patch the classes
    Runnable.invoke = wrapped_invoke
    Runnable.ainvoke = wrapped_ainvoke
```

This means **every** `Runnable.invoke()` call (chains, agents, LangGraph graphs) automatically gets the Chaukas callback injected.

---

## Step 2: Event Detection via Callback Methods

The `ChaukasCallbackHandler` implements LangChain's `BaseCallbackHandler` interface. Here's how each event type is detected:

### Session Events

| Callback | Condition | Chaukas Event |
|----------|-----------|---------------|
| `on_chain_start` | `parent_run_id is None` (root chain) | `SESSION_START` |
| `on_chain_end` | `run_id == self._root_chain_id` | `SESSION_END` |

```python
# langchain.py:304-361
def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id, ...):
    is_root_chain = parent_run_id is None

    if is_root_chain and not self._session_started:
        self._session_started = True
        self._root_chain_id = str(run_id)

        # Emit SESSION_START
        session_start = self.event_builder.create_session_start(
            metadata={"framework": "langchain", "chain_type": chain_type, ...}
        )
        self.wrapper._send_event_sync(session_start)

        # Also emit INPUT_RECEIVED for the user's input
        input_event = self.event_builder.create_input_received(content=str(inputs))
        self.wrapper._send_event_sync(input_event)
```

### Agent Events

| Callback | Condition | Chaukas Event |
|----------|-----------|---------------|
| `on_chain_start` | `"agent" in chain_type.lower()` | `AGENT_START` |
| `on_chain_end` | `run_id in self._agent_spans` | `AGENT_END` |
| Agent type change | `last_name != agent_name` | `AGENT_HANDOFF` |

```python
# langchain.py:363-403
def on_chain_start(...):
    is_agent_chain = "agent" in chain_type.lower()

    if is_agent_chain:
        # Check for agent handoff
        if self._last_active_agent:
            last_id, last_name = self._last_active_agent
            if last_name != agent_name:  # Different agent type
                handoff_event = self.event_builder.create_agent_handoff(
                    from_agent_id=last_id, to_agent_id=agent_id, ...
                )
                self.wrapper._send_event_sync(handoff_event)

        self._last_active_agent = (agent_id, agent_name)
        agent_start = self.event_builder.create_agent_start(agent_id, agent_name, ...)
        self.wrapper._send_event_sync(agent_start)
```

### Model Invocation Events

| Callback | Chaukas Event |
|----------|---------------|
| `on_llm_start` | `MODEL_INVOCATION_START` |
| `on_chat_model_start` | `MODEL_INVOCATION_START` |
| `on_llm_end` | `MODEL_INVOCATION_END` |
| `on_llm_error` | `ERROR` + `MODEL_INVOCATION_END` |

#### Difference between `on_llm_start` and `on_chat_model_start`

| Callback | LangChain Model Type | Input Format |
|----------|---------------------|--------------|
| `on_llm_start` | Legacy completion models (`OpenAI`, `Cohere`) | `prompts: List[str]` - raw text prompts |
| `on_chat_model_start` | Chat models (`ChatOpenAI`, `ChatAnthropic`) | `messages: List[List[BaseMessage]]` - structured messages with roles |

Both map to `MODEL_INVOCATION_START` because semantically they represent the same thing - the start of an LLM call. The handler normalizes the input format before creating the event.

```python
# langchain.py:565-612
def on_chat_model_start(self, serialized, messages, *, run_id, ...):
    self._start_times[str(run_id)] = time.time()  # Track start time

    model = serialized.get("name", "unknown")
    provider = self._extract_provider(model)  # "openai", "anthropic", etc.

    # Convert LangChain messages to standard format
    formatted_messages = []
    for message_list in messages:
        for msg in message_list:
            formatted_messages.append({"role": msg.type, "content": str(msg.content)})

    llm_start = self.event_builder.create_model_invocation_start(
        provider=provider,
        model=model,
        messages=formatted_messages,
        agent_id=agent_id,
        ...
    )
    self.wrapper._send_event_sync(llm_start)
    self._llm_spans[str(run_id)] = llm_start.span_id  # Store for END pairing
```

```python
# langchain.py:614-688
def on_llm_end(self, response, *, run_id, ...):
    span_id = self._llm_spans.pop(run_id_str, None)  # Retrieve paired span_id
    duration_ms = (time.time() - self._start_times.pop(run_id_str)) * 1000

    # Extract token usage from LLMResult
    if hasattr(response, "llm_output") and response.llm_output:
        token_usage = response.llm_output.get("token_usage", {})
        prompt_tokens = token_usage.get("prompt_tokens")
        completion_tokens = token_usage.get("completion_tokens")

    llm_end = self.event_builder.create_model_invocation_end(
        provider=provider,
        model=model_name,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        duration_ms=duration_ms,
        span_id=span_id,  # Same span_id as START
        ...
    )
    self.wrapper._send_event_sync(llm_end)
```

### Tool Events

| Callback | Condition | Chaukas Event |
|----------|-----------|---------------|
| `on_tool_start` | No MCP metadata | `TOOL_CALL_START` |
| `on_tool_start` | `metadata.get("mcp_server")` present | `MCP_CALL_START` |
| `on_tool_end` | `run_id in self._mcp_runs` | `MCP_CALL_END` |
| `on_tool_end` | Otherwise | `TOOL_CALL_END` |
| `on_agent_action` | Always | `TOOL_CALL_START` (agent's tool selection) |

```python
# langchain.py:756-833
def on_tool_start(self, serialized, input_str, *, run_id, metadata, ...):
    tool_name = serialized.get("name", "unknown")

    # Parse arguments
    if inputs:
        arguments = inputs
    elif input_str.startswith("{"):
        arguments = json.loads(input_str)
    else:
        arguments = {"input": input_str}

    # Check for MCP tool
    is_mcp = metadata and metadata.get("mcp_server") is not None

    if is_mcp:
        self._mcp_runs.add(str(run_id))  # Track for END event
        mcp_start = self.event_builder.create_mcp_call_start(
            server_name=metadata.get("mcp_server"),
            operation="call_tool",
            method=tool_name,
            request=arguments,
            ...
        )
        self.wrapper._send_event_sync(mcp_start)
    else:
        tool_start = self.event_builder.create_tool_call_start(
            tool_name=tool_name,
            arguments=arguments,
            call_id=str(run_id),
            ...
        )
        self.wrapper._send_event_sync(tool_start)
```

### Data Access Events (RAG/Retriever)

| Callback | Chaukas Event |
|----------|---------------|
| `on_retriever_start` | `DATA_ACCESS` |
| `on_retriever_end` | `DATA_ACCESS` (with document IDs) |

```python
# langchain.py:1172-1247
def on_retriever_start(self, serialized, query, *, run_id, ...):
    retriever_name = serialized.get("name", "unknown")

    data_event = self.event_builder.create_data_access(
        datasource=retriever_name,
        document_ids=None,  # Not known at start
        ...
    )
    self.wrapper._send_event_sync(data_event)

def on_retriever_end(self, documents, *, run_id, ...):
    # Extract document IDs from retrieved documents
    doc_ids = []
    for doc in documents:
        if hasattr(doc, "metadata") and "id" in doc.metadata:
            doc_ids.append(doc.metadata["id"])

    data_event = self.event_builder.create_data_access(
        datasource="retriever",
        document_ids=doc_ids,
        ...
    )
    self.wrapper._send_event_sync(data_event)
```

### Streaming Output Events

| Callback | Chaukas Event |
|----------|---------------|
| `on_text` | `OUTPUT_EMITTED` (with `is_streaming: true`) |

```python
# langchain.py:1065-1100
def on_text(self, text, *, run_id, parent_run_id, ...):
    if not text or not text.strip():
        return

    output_event = self.event_builder.create_output_emitted(
        content=text[:1000],
        target="stream",
        metadata={
            "run_id": str(run_id),
            "is_streaming": True,
        },
    )
    self.wrapper._send_event_sync(output_event)
```

### Custom Events

| Callback | Chaukas Event |
|----------|---------------|
| `on_custom_event` | `SYSTEM` |

```python
# langchain.py:1104-1168
def on_custom_event(self, name, data, *, run_id, tags, metadata, ...):
    # Map severity from metadata
    severity_str = metadata.get("severity", "info").lower()
    severity_map = {"debug": SEVERITY_DEBUG, "info": SEVERITY_INFO, ...}

    system_event = self.event_builder.create_system_event(
        message=f"Custom event: {name}",
        severity=severity_map.get(severity_str),
        metadata={"custom_event_name": name, "data": data, ...},
    )
    self.wrapper._send_event_sync(system_event)
```

### Error & Retry Events

| Callback | Condition | Chaukas Event |
|----------|-----------|---------------|
| `on_chain_error` | Retryable pattern detected | `RETRY` + `ERROR` |
| `on_llm_error` | Always | `ERROR` + `MODEL_INVOCATION_END` |
| `on_tool_error` | Always | `ERROR` + `TOOL_CALL_END` |
| `on_retry` | Always | `RETRY` |

```python
# langchain.py:1314-1327
def _is_retryable_error(self, error_msg: str) -> bool:
    retryable_patterns = [
        "rate limit", "timeout", "connection", "temporary",
        "503", "429", "network", "unavailable",
    ]
    return any(pattern in error_msg.lower() for pattern in retryable_patterns)
```

---

## Step 3: Span ID Tracking for START/END Pairing

The handler maintains dictionaries to correlate START and END events:

```python
# langchain.py:277-294
self._chain_spans = {}      # Map chain run_id → span_id
self._llm_spans = {}        # Map LLM run_id → span_id
self._tool_spans = {}       # Map tool run_id → span_id
self._agent_spans = {}      # Map agent run_id → span_id
self._retriever_spans = {}  # Map retriever run_id → span_id
self._mcp_runs = set()      # Track which run_ids are MCP calls
```

When a START event is created, its `span_id` is stored in the appropriate dictionary keyed by `run_id`. When the corresponding END callback is invoked, the handler retrieves the `span_id` to ensure both events share the same span.

---

## LangGraph-Specific Behavior

LangGraph uses the same callback system as LangChain. The SDK captures LangGraph metadata automatically:

```python
# LangGraph events include this metadata:
{
    "langgraph_step": 1,           # Super-step number
    "langgraph_node": "agent",     # Current node name
    "langgraph_triggers": ("branch:to:agent",),
    "langgraph_path": ("__pregel_pull", "agent"),
    "langgraph_checkpoint_ns": "agent:uuid"
}
```

| LangGraph Concept | Chaukas Event |
|-------------------|---------------|
| Graph invocation start | `SESSION_START` |
| Node execution start | `AGENT_START` |
| Node execution end | `AGENT_END` |
| Conditional edge routing | `AGENT_HANDOFF` |
| Subgraph entry | `AGENT_START` (nested) |

---

## Complete Event Flow Diagram

```
User: agent.invoke({"messages": [("human", "Calculate 25*17")]})
                           │
┌──────────────────────────┴─────────────────────────────────────┐
│ on_chain_start (parent_run_id=None) ──▶ SESSION_START          │
│                                    ──▶ INPUT_RECEIVED          │
└──────────────────────────┬─────────────────────────────────────┘
                           │
┌──────────────────────────┴─────────────────────────────────────┐
│ on_chat_model_start ──▶ MODEL_INVOCATION_START                 │
│   (LLM decides to use calculator tool)                         │
│ on_llm_end ──▶ MODEL_INVOCATION_END                            │
└──────────────────────────┬─────────────────────────────────────┘
                           │
┌──────────────────────────┴─────────────────────────────────────┐
│ on_agent_action (tool="calculator") ──▶ TOOL_CALL_START        │
│ on_tool_start ──▶ TOOL_CALL_START                              │
│   (tool executes: 25*17 = 425)                                 │
│ on_tool_end ──▶ TOOL_CALL_END                                  │
└──────────────────────────┬─────────────────────────────────────┘
                           │
┌──────────────────────────┴─────────────────────────────────────┐
│ on_chat_model_start ──▶ MODEL_INVOCATION_START                 │
│   (LLM generates final response)                               │
│ on_llm_end ──▶ MODEL_INVOCATION_END                            │
└──────────────────────────┬─────────────────────────────────────┘
                           │
┌──────────────────────────┴─────────────────────────────────────┐
│ on_agent_finish ──▶ OUTPUT_EMITTED                             │
│ on_chain_end (root) ──▶ SESSION_END                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Event Coverage Summary

| Event Type | Detection Method | Coverage |
|------------|------------------|----------|
| `SESSION_START` | Root chain start | ✅ |
| `SESSION_END` | Root chain end | ✅ |
| `AGENT_START` | Agent chain start | ✅ |
| `AGENT_END` | Agent chain end | ✅ |
| `AGENT_HANDOFF` | Agent type change | ✅ |
| `MODEL_INVOCATION_START` | LLM/Chat model start | ✅ |
| `MODEL_INVOCATION_END` | LLM end | ✅ |
| `TOOL_CALL_START` | Tool start / Agent action | ✅ |
| `TOOL_CALL_END` | Tool end | ✅ |
| `MCP_CALL_START` | Tool with `mcp_server` metadata | ✅ |
| `MCP_CALL_END` | MCP tool end | ✅ |
| `INPUT_RECEIVED` | Root chain input | ✅ |
| `OUTPUT_EMITTED` | `on_text` / Agent finish | ✅ |
| `DATA_ACCESS` | Retriever start/end | ✅ |
| `ERROR` | Any `*_error` callback | ✅ |
| `RETRY` | `on_retry` / Retryable errors | ✅ |
| `SYSTEM` | `on_custom_event` | ✅ |
| `POLICY_DECISION` | — | ❌ (requires app-level instrumentation) |
| `STATE_UPDATE` | — | ❌ (requires app-level instrumentation) |

**Total: 17/19 events captured automatically** (~95% coverage)

### Note: `on_llm_start` vs `on_chat_model_start`

Both callbacks map to `MODEL_INVOCATION_START`, but they are triggered by different LLM types:

| Callback | LangChain Model Type | Input Format | Example Models |
|----------|---------------------|--------------|----------------|
| `on_llm_start` | Legacy completion models | `prompts: List[str]` - raw text | `OpenAI`, `Cohere` |
| `on_chat_model_start` | Chat models | `messages: List[List[BaseMessage]]` - structured messages with roles | `ChatOpenAI`, `ChatAnthropic`, `ChatGoogleGenerativeAI` |

```python
# on_llm_start - Legacy completion API (rarely used today)
from langchain.llms import OpenAI
llm = OpenAI()
llm.invoke("Tell me a joke")  # Triggers on_llm_start

# on_chat_model_start - Modern chat API (most common)
from langchain_openai import ChatOpenAI
chat = ChatOpenAI()
chat.invoke([HumanMessage("Tell me a joke")])  # Triggers on_chat_model_start
```

Both are normalized to the same `MODEL_INVOCATION_START` event because they semantically represent the same operation - the start of an LLM call. The handler converts the different input formats to a unified message structure before creating the event.

**In practice today**, `on_chat_model_start` is used almost exclusively since modern LLMs (GPT-4, Claude, Gemini) all use the chat/messages API.

---

## Usage Examples

### Basic Chain

```python
from chaukas import sdk as chaukas
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

chaukas.enable_chaukas()

prompt = ChatPromptTemplate.from_template("Tell me about {topic}")
llm = ChatOpenAI(model="gpt-4o-mini")
chain = prompt | llm

result = chain.invoke({"topic": "quantum computing"})
# Events captured: SESSION_START, INPUT_RECEIVED, MODEL_INVOCATION_START/END, OUTPUT_EMITTED, SESSION_END
```

### Agent with Tools

```python
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

agent = create_react_agent(ChatOpenAI(), [calculator])
result = agent.invoke({"messages": [("human", "What is 25 * 17?")]})
# Events captured: SESSION_START, MODEL_INVOCATION_START/END, TOOL_CALL_START/END, SESSION_END
```

### RAG with Retriever

```python
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever()

rag_chain = retriever | prompt | llm
result = rag_chain.invoke("What is Chaukas?")
# Events captured: SESSION_START, DATA_ACCESS, MODEL_INVOCATION_START/END, SESSION_END
```

### MCP Tools

```python
# Include mcp_server in config metadata to trigger MCP_CALL events
result = agent.invoke(
    {"messages": [("human", "Search for AI news")]},
    config={
        "metadata": {
            "mcp_server": "search-server",
            "mcp_server_url": "mcp://search.local",
        }
    },
)
# Events captured: MCP_CALL_START/END instead of TOOL_CALL_START/END
```

---

## Disabling Instrumentation

To disable instrumentation and restore original methods:

```python
chaukas.disable_chaukas()
```

This will:
1. Restore original `Runnable.invoke()` and `Runnable.ainvoke()` methods
2. Flush any remaining events in the queue
3. Clean up resources
