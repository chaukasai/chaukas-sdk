# OpenAI Agents SDK Examples

Examples demonstrating Chaukas SDK integration with OpenAI Agents.

## Quick Start

### 1. Setup Environment

Copy the example environment file and add your OpenAI API key:

```bash
cd examples/openai
cp .env.example .env
```

Edit `.env` and set your OpenAI API key:

```bash
OPENAI_API_KEY=sk-your-actual-api-key-here
```

### 2. Run Examples

**Simple Example** - Basic agent interaction:
```bash
python openai_simple.py
```

**Comprehensive Example** - Interactive menu with 8 scenarios:
```bash
python openai_comprehensive_example.py
```

That's it! Events are automatically captured to `.jsonl` files.

## Examples

### openai_simple.py
Minimal example - perfect for getting started.

### openai_comprehensive_example.py
Interactive menu with 8 scenarios demonstrating all event types:
1. Research Assistant (web search & tools)
2. Math Tutor (calculator)
3. Error Handling Demo (timeout errors)
4. Multi-Agent Handoff
5. Policy Decision & Content Safety
6. Data Access & Retrieval Tracking
7. MCP Integration (requires MCP server)

### Other Examples
- `openai_tools_example.py` - Tool calling patterns
- `openai_handoff_message_filter.py` - Agent handoffs with message filtering
- `openai_sqllite_session_example.py` - Session persistence with SQLite

## Configuration

All configuration is in `.env`:

```bash
# Required
OPENAI_API_KEY=sk-...

# Chaukas Configuration (optional)
CHAUKAS_OUTPUT_MODE=file                    # "file" or "api"
CHAUKAS_OUTPUT_FILE=openai_output.jsonl    # Output file for events
CHAUKAS_BATCH_SIZE=1                       # Events per batch

# For API mode (optional)
# CHAUKAS_ENDPOINT=https://api.chaukas.ai
# CHAUKAS_API_KEY=your-chaukas-api-key
# CHAUKAS_TENANT_ID=your-tenant
# CHAUKAS_PROJECT_ID=your-project
```

## Viewing Events

Events are saved as JSONL files. View them with `jq`:

```bash
# View all events
cat openai_output.jsonl | jq '.'

# Count event types
cat openai_output.jsonl | jq -r '.type' | sort | uniq -c

# View specific events
cat openai_output.jsonl | jq 'select(.type == "EVENT_TYPE_TOOL_CALL_START")'
```

## Requirements

```bash
pip install "openai-agents>=0.5.0,<1.0.0" chaukas-sdk
```

## Event Coverage

The comprehensive example captures **18/19 event types (94.7%)**:

✅ Captured:
- SESSION_START/END
- AGENT_START/END
- AGENT_HANDOFF
- MODEL_INVOCATION_START/END
- TOOL_CALL_START/END
- MCP_CALL_START/END (when MCP server is running)
- INPUT_RECEIVED/OUTPUT_EMITTED
- ERROR
- STATE_UPDATE
- POLICY_DECISION
- DATA_ACCESS
- SYSTEM

❌ Not Captured:
- **RETRY** - OpenAI SDK performs retries internally within the HTTP client layer. These retry attempts are invisible to external observers, so we cannot emit RETRY events. We only see the final ERROR event after all SDK retries are exhausted.

## MCP Integration

To run the MCP scenario:

1. Start the MCP server (separate terminal):
```bash
cd mcp/prompt-server
python server.py
```

2. Run comprehensive example and select option 8

## Troubleshooting

**Missing API key:**
```
❌ Error: OPENAI_API_KEY environment variable is not set
```
→ Add your API key to `.env`

**Module not found:**
```
ModuleNotFoundError: No module named 'agents'
```
→ Run `pip install "openai-agents>=0.5.0,<1.0.0"`

**Rate limits:**
→ The examples use `gpt-4o-mini` to minimize costs and rate limits
