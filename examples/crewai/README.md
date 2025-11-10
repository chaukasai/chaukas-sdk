# CrewAI Examples

Examples demonstrating the Chaukas SDK integration with CrewAI framework for comprehensive agent observability.

## Quick Start

1. **Install dependencies:**
```bash
pip install "crewai>=1.4.1,<2.0.0" chaukas-sdk
```

2. **Set up environment variables:**
```bash
export OPENAI_API_KEY="sk-..."
export CREWAI_DISABLE_TELEMETRY=true
```

3. **Run an example:**
```bash
python crewai_example.py
```

Events will be captured in `events.jsonl` by default.

## Examples

### Core Examples (All Event Types)

- **[crewai_example.py](crewai_example.py)** - Basic multi-agent collaboration with event capture
- **[crewai_comprehensive_example.py](crewai_comprehensive_example.py)** - Full-featured example with multiple crews and complex workflows
- **[crewai_retry_example.py](crewai_retry_example.py)** - Error handling and retry logic demonstration
- **[crewai_tools_example.py](crewai_tools_example.py)** - Custom tool implementations and MCP detection

### Feature-Specific Examples

These examples require specific CrewAI features to capture additional event types:

- **[crewai_knowledge_example.py](crewai_knowledge_example.py)** ⚠️ - DATA_ACCESS events via Knowledge feature (requires `pip install chromadb`)
- **[crewai_guardrails_example.py](crewai_guardrails_example.py)** ⚠️ - POLICY_DECISION events via Task Guardrails
- **[crewai_flows_example.py](crewai_flows_example.py)** ⚠️ - SYSTEM events via CrewAI Flows orchestration

⚠️ = Requires specific CrewAI features to be configured

## Configuration

### Environment Variables

Create a `.env` file in the example directory:

```bash
# Required
OPENAI_API_KEY=sk-...
CREWAI_DISABLE_TELEMETRY=true

# Optional Chaukas Configuration
CHAUKAS_OUTPUT_MODE=file           # or "api"
CHAUKAS_OUTPUT_FILE=events.jsonl   # default output file

# For API mode (optional)
# CHAUKAS_OUTPUT_MODE=api
# CHAUKAS_ENDPOINT=https://api.chaukas.ai
# CHAUKAS_API_KEY=your-chaukas-api-key
# CHAUKAS_TENANT_ID=your-tenant-id
# CHAUKAS_PROJECT_ID=your-project-id

# Batching (optional)
CHAUKAS_BATCH_SIZE=1              # Immediate write for demos
CHAUKAS_FLUSH_INTERVAL=5.0        # Flush interval in seconds
```

### Required Variables

- `OPENAI_API_KEY` - Your OpenAI API key (CrewAI uses OpenAI by default)
- `CREWAI_DISABLE_TELEMETRY` - Set to `true` to prevent telemetry errors

## Viewing Events

After running an example, analyze captured events:

```bash
# View all events
cat events.jsonl | jq '.'

# Count event types
cat events.jsonl | jq -r '.type' | sort | uniq -c

# View specific event types
cat events.jsonl | jq 'select(.type == "AGENT_HANDOFF")'
cat events.jsonl | jq 'select(.type == "DATA_ACCESS")'
cat events.jsonl | jq 'select(.type == "POLICY_DECISION")'
cat events.jsonl | jq 'select(.type == "TOOL_CALL_START")'

# Track task lifecycle
cat events.jsonl | jq 'select(.type | contains("TASK"))'

# View LLM invocations
cat events.jsonl | jq 'select(.type | contains("MODEL_INVOCATION"))'
```

## Event Coverage

CrewAI provides the most comprehensive event capture through its event bus architecture:

### Core Events (Captured in All Examples)
- `SESSION_START` / `SESSION_END` - Crew lifecycle
- `AGENT_START` / `AGENT_END` - Agent execution
- `MODEL_INVOCATION_START` / `MODEL_INVOCATION_END` - LLM calls
- `TOOL_CALL_START` / `TOOL_CALL_END` - Tool execution
- `INPUT_RECEIVED` - User inputs
- `OUTPUT_EMITTED` - Agent outputs

### Feature-Specific Events
- `ERROR` / `RETRY` - Tool failures → **crewai_retry_example.py**
- `AGENT_HANDOFF` - Multi-agent workflows → **crewai_comprehensive_example.py**
- `MCP_CALL_START` / `MCP_CALL_END` - MCP tools → **crewai_comprehensive_example.py**
- `POLICY_DECISION` - Task Guardrails → **crewai_guardrails_example.py** ⚠️
- `DATA_ACCESS` - Knowledge Sources → **crewai_knowledge_example.py** ⚠️
- `STATE_UPDATE` - Agent Reasoning (automatic when enabled)
- `SYSTEM` - CrewAI Flows → **crewai_flows_example.py** ⚠️

**Coverage:** 13/20 core events + 7/20 feature-specific events (with features enabled)

### Comparison with OpenAI Agents

| Feature | OpenAI Agents | CrewAI |
|---------|---------------|--------|
| Core Events | 13/20 | 13/20 |
| Feature-Specific Events | 3/7 | 7/7 (with features) |
| Architecture | Method patching | Event bus |
| Multi-Agent Support | Limited | Native |
| Knowledge/RAG | Built-in | Via Knowledge feature |
| Guardrails | N/A | Via Task guardrails |
| Flows/Orchestration | N/A | Via Flows feature |

## Requirements

```bash
# Core requirements
pip install "crewai>=1.4.1,<2.0.0" chaukas-sdk

# Or with CrewAI integration extras
pip install "chaukas-sdk[crewai]"

# For Knowledge examples (DATA_ACCESS events)
pip install chromadb

# For alternative vector stores
pip install qdrant-client  # if using Qdrant
```

## Troubleshooting

### Service Unavailable Error

```
Service Unavailable encountered while exporting span batch
```

**Solution:** Disable CrewAI's built-in telemetry:
```bash
export CREWAI_DISABLE_TELEMETRY=true
```

### Missing API Key

```
Error: OPENAI_API_KEY not set
```

**Solution:** Set your OpenAI API key:
```bash
export OPENAI_API_KEY="sk-..."
```

### Import Error

```
ModuleNotFoundError: No module named 'crewai'
```

**Solution:** Install CrewAI:
```bash
pip install "crewai>=1.4.1,<2.0.0"
```

### ChromaDB Required Error

```
ModuleNotFoundError: No module named 'chromadb'
```

**Solution:** Install ChromaDB for Knowledge examples:
```bash
pip install chromadb
```

### No Events Captured

**Check:**
1. Ensure `chaukas.enable_chaukas()` is called before creating agents
2. Verify `CHAUKAS_OUTPUT_FILE` path is writable
3. Call `client.flush()` at the end of your script
4. Check for errors in console output

## Further Reading

- [CrewAI Documentation](https://docs.crewai.com)
- [chaukas-spec](https://github.com/chaukasai/chaukas-spec) - Standardized event schema
- [CrewAI Event Bus Concepts](https://docs.crewai.com/concepts/events)
- [OpenAI Examples](../openai) - Compare with OpenAI Agents integration patterns
