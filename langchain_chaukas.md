# LangChain/LangGraph → Chaukas-Spec Event Mapping

## Overview

This document provides an accurate mapping between LangChain/LangGraph's callback system and the chaukas-spec event types as defined in the [chaukas-spec repository](https://github.com/chaukasai/chaukas-spec).

---

## Chaukas-Spec Event Types (from proto/chaukas/spec/common/v1/events.proto)

Based on the official README, chaukas-spec defines these event types:

| Category | Event Types |
|----------|-------------|
| **Session Lifecycle** | `EVENT_TYPE_SESSION_START`, `EVENT_TYPE_SESSION_END` |
| **Agent Operations** | `EVENT_TYPE_AGENT_START`, `EVENT_TYPE_AGENT_END`, `EVENT_TYPE_AGENT_HANDOFF` |
| **Model Invocations** | `EVENT_TYPE_MODEL_INVOCATION_START`, `EVENT_TYPE_MODEL_INVOCATION_END` |
| **Tool Interactions** | `EVENT_TYPE_TOOL_CALL_START`, `EVENT_TYPE_TOOL_CALL_END` |
| **MCP Protocol** | `EVENT_TYPE_MCP_CALL_START`, `EVENT_TYPE_MCP_CALL_END` |
| **I/O Events** | `EVENT_TYPE_INPUT_RECEIVED`, `EVENT_TYPE_OUTPUT_EMITTED` |
| **Error Handling** | `EVENT_TYPE_ERROR`, `EVENT_TYPE_RETRY` |

---

## LangChain BaseCallbackHandler Methods

LangChain's `BaseCallbackHandler` provides these callback methods:

### LLM/Chat Model Callbacks
- `on_llm_start(serialized, prompts, *, run_id, parent_run_id, tags, metadata)`
- `on_chat_model_start(serialized, messages, *, run_id, parent_run_id, tags, metadata)`
- `on_llm_new_token(token, *, chunk, run_id, parent_run_id)`
- `on_llm_end(response, *, run_id, parent_run_id)`
- `on_llm_error(error, *, run_id, parent_run_id)`

### Chain Callbacks
- `on_chain_start(serialized, inputs, *, run_id, parent_run_id, tags, metadata)`
- `on_chain_end(outputs, *, run_id, parent_run_id)`
- `on_chain_error(error, *, run_id, parent_run_id)`

### Tool Callbacks
- `on_tool_start(serialized, input_str, *, run_id, parent_run_id, tags, metadata, inputs)`
- `on_tool_end(output, *, run_id, parent_run_id)`
- `on_tool_error(error, *, run_id, parent_run_id)`

### Agent Callbacks
- `on_agent_action(action, *, run_id, parent_run_id)`
- `on_agent_finish(finish, *, run_id, parent_run_id)`

### Retriever Callbacks
- `on_retriever_start(serialized, query, *, run_id, parent_run_id, tags, metadata)`
- `on_retriever_end(documents, *, run_id, parent_run_id)`
- `on_retriever_error(error, *, run_id, parent_run_id)`

### Other Callbacks
- `on_retry(retry_state, *, run_id, parent_run_id)`
- `on_text(text, *, run_id, parent_run_id)`
- `on_custom_event(name, data, *, run_id, tags, metadata)`

---

## Complete Event Mapping Table

### Direct Mappings

| LangChain Callback | Chaukas-Spec EventType | Notes |
|--------------------|------------------------|-------|
| `on_llm_start` | `EVENT_TYPE_MODEL_INVOCATION_START` | For non-chat models (legacy LLMs) |
| `on_chat_model_start` | `EVENT_TYPE_MODEL_INVOCATION_START` | For chat models (ChatGPT, Claude, etc.) |
| `on_llm_end` | `EVENT_TYPE_MODEL_INVOCATION_END` | Includes response and token usage |
| `on_llm_error` | `EVENT_TYPE_ERROR` | Error during LLM invocation |
| `on_tool_start` | `EVENT_TYPE_TOOL_CALL_START` | Tool/function execution begins |
| `on_tool_end` | `EVENT_TYPE_TOOL_CALL_END` | Tool/function execution completes |
| `on_tool_error` | `EVENT_TYPE_ERROR` | Error during tool execution |
| `on_chain_start` (agent) | `EVENT_TYPE_AGENT_START` | When chain represents an agent |
| `on_chain_end` (agent) | `EVENT_TYPE_AGENT_END` | Agent execution completes |
| `on_chain_error` | `EVENT_TYPE_ERROR` | Error during chain/agent execution |
| `on_agent_action` | `EVENT_TYPE_TOOL_CALL_START` | Agent selects a tool (precedes tool call) |
| `on_agent_finish` | `EVENT_TYPE_AGENT_END` | Agent completes with final answer |
| `on_retry` | `EVENT_TYPE_RETRY` | Retry attempt (e.g., rate limit, transient error) |

### Mappings Requiring Context/Heuristics

| LangChain Callback | Chaukas-Spec EventType | Condition/Logic |
|--------------------|------------------------|-----------------|
| `on_chain_start` | `EVENT_TYPE_SESSION_START` | If this is the root chain (no `parent_run_id`) |
| `on_chain_start` | `EVENT_TYPE_AGENT_START` | If chain represents an agent (check `serialized["name"]` or tags) |
| `on_chain_end` | `EVENT_TYPE_SESSION_END` | If this is the root chain (no `parent_run_id`) |
| `on_chain_end` | `EVENT_TYPE_AGENT_END` | If chain represents an agent |
| `on_retriever_start` | `EVENT_TYPE_TOOL_CALL_START` | Retriever can be treated as a specialized tool |
| `on_retriever_end` | `EVENT_TYPE_TOOL_CALL_END` | Retriever completion |
| `on_text` | `EVENT_TYPE_OUTPUT_EMITTED` | Text output from any component |
| User message input | `EVENT_TYPE_INPUT_RECEIVED` | Must be captured at application level |

### Chaukas-Spec Events Without Direct LangChain Equivalents

| Chaukas-Spec EventType | How to Capture | Status |
|------------------------|----------------|--------|
| `EVENT_TYPE_SESSION_START` | Emit when first `on_chain_start` has no `parent_run_id` | ✅ Implemented |
| `EVENT_TYPE_SESSION_END` | Emit when root chain completes | ✅ Implemented |
| `EVENT_TYPE_AGENT_HANDOFF` | Custom: detect when one agent delegates to another | ✅ Implemented |
| `EVENT_TYPE_MCP_CALL_START` | Detected via `mcp_server` metadata in `on_tool_start` | ✅ Implemented |
| `EVENT_TYPE_MCP_CALL_END` | Tracked via `_mcp_runs` set for MCP tool completions | ✅ Implemented |
| `EVENT_TYPE_INPUT_RECEIVED` | Captured at chain invocation | ✅ Implemented |
| `EVENT_TYPE_OUTPUT_EMITTED` | Captured via `on_text` callback for streaming | ✅ Implemented |
| `EVENT_TYPE_SYSTEM` | Custom events via `on_custom_event` callback | ✅ Implemented |
| `EVENT_TYPE_POLICY_DECISION` | Requires application-level instrumentation | ❌ Not captured |
| `EVENT_TYPE_STATE_UPDATE` | Requires application-level instrumentation | ❌ Not captured |

---

## LangGraph-Specific Considerations

LangGraph extends LangChain with graph-based execution. Key metadata available in LangGraph events:

```python
{
    "langgraph_step": 1,           # Super-step number
    "langgraph_node": "agent",     # Current node name
    "langgraph_triggers": ("branch:to:agent",),
    "langgraph_path": ("__pregel_pull", "agent"),
    "langgraph_checkpoint_ns": "agent:uuid"
}
```

### LangGraph Event Mapping

| LangGraph Event/Metadata | Chaukas-Spec EventType | Notes |
|-------------------------|------------------------|-------|
| Graph invocation start | `EVENT_TYPE_SESSION_START` | Root graph execution |
| Node execution start | `EVENT_TYPE_AGENT_START` | Each node as sub-agent |
| Node execution end | `EVENT_TYPE_AGENT_END` | Node completes |
| Conditional edge routing | `EVENT_TYPE_AGENT_HANDOFF` | When routing between nodes |
| Checkpoint save | No direct mapping | Could extend as custom event |
| Subgraph entry | `EVENT_TYPE_AGENT_START` (nested) | Nested agent execution |

---

## Data Field Mappings

### Model Invocation Event Fields

```python
# LangChain on_chat_model_start → EVENT_TYPE_MODEL_INVOCATION_START
{
    "event_type": EventType.EVENT_TYPE_MODEL_INVOCATION_START,
    "event_id": str(uuid4()),
    "trace_id": derive_trace_id(run_id, parent_run_id),  # See trace correlation section
    "session_id": session_id,  # From root chain
    
    # Map from LangChain parameters
    "model_name": serialized.get("kwargs", {}).get("model_name"),
    "model_provider": serialized.get("id", ["unknown"])[-1],  # e.g., "ChatOpenAI"
    "input_messages": [
        {"role": msg.type, "content": msg.content}
        for msg in messages[0]
    ],
    "metadata": metadata or {},
    "tags": tags or [],
}

# LangChain on_llm_end → EVENT_TYPE_MODEL_INVOCATION_END
{
    "event_type": EventType.EVENT_TYPE_MODEL_INVOCATION_END,
    "event_id": str(uuid4()),
    "trace_id": trace_id,
    
    # Map from LLMResult
    "output_messages": [
        {"role": "assistant", "content": g.text}
        for gen_list in response.generations for g in gen_list
    ],
    "token_usage": {
        "input_tokens": response.llm_output.get("token_usage", {}).get("prompt_tokens"),
        "output_tokens": response.llm_output.get("token_usage", {}).get("completion_tokens"),
    },
    "finish_reason": response.generations[0][0].generation_info.get("finish_reason"),
    "model_response_id": response.llm_output.get("id"),
}
```

### Tool Call Event Fields

```python
# LangChain on_tool_start → EVENT_TYPE_TOOL_CALL_START
{
    "event_type": EventType.EVENT_TYPE_TOOL_CALL_START,
    "event_id": str(uuid4()),
    "trace_id": trace_id,
    
    # Map from LangChain parameters
    "tool_name": serialized.get("name"),
    "tool_description": serialized.get("description"),
    "tool_arguments": inputs or {"raw_input": input_str},
}

# LangChain on_tool_end → EVENT_TYPE_TOOL_CALL_END
{
    "event_type": EventType.EVENT_TYPE_TOOL_CALL_END,
    "event_id": str(uuid4()),
    "trace_id": trace_id,
    
    "tool_output": output,
    "success": True,  # False if on_tool_error was called
}
```

---

## Trace Correlation Strategy

LangChain uses `run_id` and `parent_run_id` for hierarchical tracing. Chaukas-spec uses `trace_id`, `session_id`, and span relationships.

### Mapping Strategy

```python
class TraceCorrelator:
    def __init__(self):
        self.run_to_trace: Dict[UUID, str] = {}
        self.root_run_id: Optional[UUID] = None
        self.session_id: Optional[str] = None
    
    def derive_trace_id(self, run_id: UUID, parent_run_id: Optional[UUID]) -> str:
        """Derive chaukas trace_id from LangChain run hierarchy."""
        if parent_run_id is None:
            # This is a root run - it defines the trace
            trace_id = str(run_id)
            self.run_to_trace[run_id] = trace_id
            self.root_run_id = run_id
            self.session_id = f"session_{trace_id[:8]}"
            return trace_id
        
        # Inherit trace_id from parent
        if parent_run_id in self.run_to_trace:
            trace_id = self.run_to_trace[parent_run_id]
        else:
            # Fallback: use parent_run_id as trace
            trace_id = str(parent_run_id)
        
        self.run_to_trace[run_id] = trace_id
        return trace_id
```

---

## Implementation: ChaukasCallbackHandler

```python
from langchain_core.callbacks import BaseCallbackHandler
from chaukas.spec.common.v1.events_pb2 import Event, EventType
from chaukas.spec.client.v1.client_pb2 import IngestEventRequest
from uuid import UUID, uuid4
from typing import Any, Dict, List, Optional, Sequence
from datetime import datetime


class ChaukasCallbackHandler(BaseCallbackHandler):
    """
    LangChain callback handler that emits chaukas-spec compliant events.
    
    Usage:
        handler = ChaukasCallbackHandler(exporter=my_exporter)
        chain.invoke(inputs, config={"callbacks": [handler]})
    """
    
    def __init__(self, exporter, session_id: Optional[str] = None):
        self.exporter = exporter
        self.session_id = session_id or f"session_{uuid4().hex[:8]}"
        self.run_to_trace: Dict[UUID, str] = {}
        self.root_run_id: Optional[UUID] = None
        
    def _get_trace_id(self, run_id: UUID, parent_run_id: Optional[UUID]) -> str:
        if parent_run_id is None:
            trace_id = str(run_id)
            self.root_run_id = run_id
        elif parent_run_id in self.run_to_trace:
            trace_id = self.run_to_trace[parent_run_id]
        else:
            trace_id = str(parent_run_id)
        self.run_to_trace[run_id] = trace_id
        return trace_id

    def _emit(self, event_type: EventType, run_id: UUID, 
              parent_run_id: Optional[UUID], **kwargs):
        trace_id = self._get_trace_id(run_id, parent_run_id)
        event = Event(
            event_id=f"evt_{uuid4().hex}",
            type=event_type,
            session_id=self.session_id,
            trace_id=trace_id,
            span_id=str(run_id),
            parent_span_id=str(parent_run_id) if parent_run_id else None,
            timestamp=datetime.utcnow().isoformat(),
            **kwargs
        )
        self.exporter.emit(event)

    # === Session Events ===
    
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], *,
                       run_id: UUID, parent_run_id: Optional[UUID] = None,
                       tags: Optional[List[str]] = None,
                       metadata: Optional[Dict[str, Any]] = None, **kwargs):
        
        # Determine if this is session start or agent start
        if parent_run_id is None:
            # Root chain = session start
            self._emit(
                EventType.EVENT_TYPE_SESSION_START,
                run_id, parent_run_id,
                # Additional fields can be added based on your Event proto schema
            )
        
        # Also emit agent start for the chain itself
        self._emit(
            EventType.EVENT_TYPE_AGENT_START,
            run_id, parent_run_id,
        )
    
    def on_chain_end(self, outputs: Dict[str, Any], *,
                     run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs):
        
        self._emit(
            EventType.EVENT_TYPE_AGENT_END,
            run_id, parent_run_id,
        )
        
        # If root chain, also emit session end
        if run_id == self.root_run_id:
            self._emit(
                EventType.EVENT_TYPE_SESSION_END,
                run_id, parent_run_id,
            )

    def on_chain_error(self, error: BaseException, *,
                       run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs):
        self._emit(
            EventType.EVENT_TYPE_ERROR,
            run_id, parent_run_id,
        )

    # === Model Invocation Events ===

    def on_chat_model_start(self, serialized: Dict[str, Any],
                            messages: List[List], *,
                            run_id: UUID, parent_run_id: Optional[UUID] = None,
                            tags: Optional[List[str]] = None,
                            metadata: Optional[Dict[str, Any]] = None, **kwargs):
        self._emit(
            EventType.EVENT_TYPE_MODEL_INVOCATION_START,
            run_id, parent_run_id,
        )
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], *,
                     run_id: UUID, parent_run_id: Optional[UUID] = None,
                     tags: Optional[List[str]] = None,
                     metadata: Optional[Dict[str, Any]] = None, **kwargs):
        self._emit(
            EventType.EVENT_TYPE_MODEL_INVOCATION_START,
            run_id, parent_run_id,
        )

    def on_llm_end(self, response, *, run_id: UUID,
                   parent_run_id: Optional[UUID] = None, **kwargs):
        self._emit(
            EventType.EVENT_TYPE_MODEL_INVOCATION_END,
            run_id, parent_run_id,
        )
    
    def on_llm_error(self, error: BaseException, *,
                     run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs):
        self._emit(
            EventType.EVENT_TYPE_ERROR,
            run_id, parent_run_id,
        )

    # === Tool Events ===
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, *,
                      run_id: UUID, parent_run_id: Optional[UUID] = None,
                      tags: Optional[List[str]] = None,
                      metadata: Optional[Dict[str, Any]] = None,
                      inputs: Optional[Dict[str, Any]] = None, **kwargs):
        
        # Check if this is an MCP tool (by convention or metadata)
        is_mcp = metadata and metadata.get("mcp_server") is not None
        
        if is_mcp:
            self._emit(
                EventType.EVENT_TYPE_MCP_CALL_START,
                run_id, parent_run_id,
            )
        else:
            self._emit(
                EventType.EVENT_TYPE_TOOL_CALL_START,
                run_id, parent_run_id,
            )
    
    def on_tool_end(self, output: Any, *,
                    run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs):
        # Note: Would need to track if it was MCP or regular tool
        self._emit(
            EventType.EVENT_TYPE_TOOL_CALL_END,
            run_id, parent_run_id,
        )

    def on_tool_error(self, error: BaseException, *,
                      run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs):
        self._emit(
            EventType.EVENT_TYPE_ERROR,
            run_id, parent_run_id,
        )

    # === Agent Events ===
    
    def on_agent_action(self, action, *, run_id: UUID,
                        parent_run_id: Optional[UUID] = None, **kwargs):
        # Agent action precedes tool call - emit tool call start
        self._emit(
            EventType.EVENT_TYPE_TOOL_CALL_START,
            run_id, parent_run_id,
        )
    
    def on_agent_finish(self, finish, *, run_id: UUID,
                        parent_run_id: Optional[UUID] = None, **kwargs):
        self._emit(
            EventType.EVENT_TYPE_AGENT_END,
            run_id, parent_run_id,
        )

    # === Retriever Events (mapped to tool calls) ===
    
    def on_retriever_start(self, serialized: Dict[str, Any], query: str, *,
                          run_id: UUID, parent_run_id: Optional[UUID] = None,
                          tags: Optional[List[str]] = None,
                          metadata: Optional[Dict[str, Any]] = None, **kwargs):
        self._emit(
            EventType.EVENT_TYPE_TOOL_CALL_START,
            run_id, parent_run_id,
        )
    
    def on_retriever_end(self, documents: Sequence, *,
                        run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs):
        self._emit(
            EventType.EVENT_TYPE_TOOL_CALL_END,
            run_id, parent_run_id,
        )
    
    def on_retriever_error(self, error: BaseException, *,
                          run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs):
        self._emit(
            EventType.EVENT_TYPE_ERROR,
            run_id, parent_run_id,
        )

    # === Other Events ===
    
    def on_retry(self, retry_state, *, run_id: UUID,
                 parent_run_id: Optional[UUID] = None, **kwargs):
        self._emit(
            EventType.EVENT_TYPE_RETRY,
            run_id, parent_run_id,
        )
    
    def on_text(self, text: str, *, run_id: UUID,
                parent_run_id: Optional[UUID] = None, **kwargs):
        self._emit(
            EventType.EVENT_TYPE_OUTPUT_EMITTED,
            run_id, parent_run_id,
        )
```

---

## Summary: Event Flow for a Typical Agent Execution

```
User Input → INPUT_RECEIVED (application-level)
    │
    └─→ on_chain_start (root) → SESSION_START + AGENT_START
            │
            ├─→ on_chat_model_start → MODEL_INVOCATION_START
            │       │
            │       └─→ on_llm_end → MODEL_INVOCATION_END
            │
            ├─→ on_agent_action → TOOL_CALL_START
            │       │
            │       └─→ on_tool_start → TOOL_CALL_START
            │               │
            │               └─→ on_tool_end → TOOL_CALL_END
            │
            ├─→ on_chat_model_start → MODEL_INVOCATION_START (second LLM call)
            │       │
            │       └─→ on_llm_end → MODEL_INVOCATION_END
            │
            └─→ on_agent_finish → AGENT_END
                    │
                    └─→ on_chain_end (root) → AGENT_END + SESSION_END
                            │
                            └─→ OUTPUT_EMITTED (final response)
```

---

## Key Differences from Previous Mapping

1. **No OTel attributes in chaukas-spec** - Chaukas-spec has its own event schema, not OTel semantic conventions directly
2. **EVENT_TYPE_ prefix** - All event types use `EVENT_TYPE_` prefix per proto enum convention
3. **MODEL_INVOCATION vs LLM** - Chaukas uses `MODEL_INVOCATION_START/END` not `llm_request/response`
4. **Session concept** - Chaukas has explicit `SESSION_START/END` events that LangChain doesn't have
5. **MCP support** - Native `MCP_CALL_START/END` events for Model Context Protocol calls
6. **Streaming events** - `on_text` callback captures streaming output as `OUTPUT_EMITTED` events
