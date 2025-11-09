# OpenAI Agents SDK Examples

Comprehensive examples demonstrating the Chaukas SDK integration with OpenAI Agents SDK.

## Event Coverage: 16/20 (80%)

**Captured Events:**
- ✅ SESSION_START/END - Session lifecycle tracking
- ✅ AGENT_START/END - Agent execution lifecycle
- ✅ MODEL_INVOCATION_START/END - LLM API calls
- ✅ TOOL_CALL_START/END - Tool execution tracking
- ✅ INPUT_RECEIVED - User input capture
- ✅ OUTPUT_EMITTED - Agent response tracking
- ✅ ERROR - Error and exception tracking
- ✅ RETRY - Retry attempt detection
- ✅ AGENT_HANDOFF - Agent-to-agent handoffs

**Not Captured** (framework limitations):
- ❌ MCP_CALL_START/END - No native MCP support in OpenAI SDK
- ❌ POLICY_DECISION - Not exposed by OpenAI Agents SDK
- ❌ DATA_ACCESS - No data access layer
- ❌ STATE_UPDATE - State changes not observable

## Examples

### 1. openai_comprehensive_example.py

**The flagship example** - Full-featured demonstration with real API calls and 5 different scenarios.

**Features:**
- Real OpenAI API integration (requires API key)
- Interactive menu with 5 comprehensive scenarios
- Custom tool implementations (`@function_tool` decorator)
- Built-in event analysis and reporting
- Error handling with retry logic
- Multi-agent handoff demonstration

**Scenarios:**
1. Research Assistant - Web search and MCP tools
2. Math Tutor - Calculator tool usage
3. Travel Planner - Weather API with retries
4. Error Handling Demo - Recovery strategies
5. Multi-Agent Handoff - Agent collaboration

**Usage:**
```bash
export OPENAI_API_KEY="sk-..."
python openai_comprehensive_example.py
```

### 2. openai_simple.py

**Quick start example** - Minimal boilerplate, single agent interaction.

**Features:**
- Simplest possible integration
- Automatic event capture with zero configuration
- Perfect for understanding the basics

**Usage:**
```bash
export OPENAI_API_KEY="sk-..."
python openai_simple.py
```

### 3. openai_tools_example.py

**Tool-focused example** - Demonstrates tool calling patterns and TOOL_CALL event capture.

**Features:**
- Multiple tool implementations
- Tool execution tracking
- Tool error handling
- Performance metrics

**Usage:**
```bash
export OPENAI_API_KEY="sk-..."
python openai_tools_example.py
```

### 4. openai_retry_example.py

**Retry logic demonstration** - Shows retry detection and error recovery patterns.

**Features:**
- Simulated API failures
- Exponential backoff strategies
- RETRY event capture
- Error recovery patterns

**Usage:**
```bash
export OPENAI_API_KEY="sk-..."
python openai_retry_example.py
```

### 5. openai_handoff_message_filter.py

**Agent handoff example** - Demonstrates multi-agent collaboration with message filtering.

**Features:**
- Agent-to-agent handoffs
- Message filtering between agents
- Context propagation
- Specialized agent roles

**Usage:**
```bash
export OPENAI_API_KEY="sk-..."
python openai_handoff_message_filter.py
```

### 6. openai_sqllite_session_example.py

**Session management example** - Shows session persistence with SQLite storage.

**Features:**
- Session state persistence
- SQLite integration
- Multi-turn conversations
- Session restoration

**Usage:**
```bash
export OPENAI_API_KEY="sk-..."
python openai_sqllite_session_example.py
```

## Requirements

```bash
# Install OpenAI Agents SDK
pip install "openai-agents>=0.5.0"

# Install Chaukas SDK
pip install chaukas-sdk

# Or install with OpenAI integration
pip install "chaukas-sdk[openai]"

# Set your OpenAI API key
export OPENAI_API_KEY="sk-..."
```

## Configuration

### Required Environment Variables

```bash
export OPENAI_API_KEY="sk-..."  # Your OpenAI API key
```

### Optional Chaukas Configuration

```bash
# Output mode (default: file)
export CHAUKAS_OUTPUT_MODE="file"  # or "api"
export CHAUKAS_OUTPUT_FILE="events.jsonl"

# For API mode
export CHAUKAS_OUTPUT_MODE="api"
export CHAUKAS_ENDPOINT="https://api.chaukas.com"
export CHAUKAS_API_KEY="your-chaukas-api-key"
export CHAUKAS_TENANT_ID="your-tenant"
export CHAUKAS_PROJECT_ID="your-project"

# Batching configuration
export CHAUKAS_BATCH_SIZE="1"  # Immediate write for demos
export CHAUKAS_FLUSH_INTERVAL="5.0"  # Flush interval in seconds
```

## Analyzing Events

All examples output events to JSONL files. Analyze them using:

```bash
# View all events
cat events.jsonl | jq '.'

# Count event types
cat events.jsonl | jq -r '.type' | sort | uniq -c

# Filter specific events
cat events.jsonl | jq 'select(.type == "TOOL_CALL_START")'

# View agent timeline
cat events.jsonl | jq 'select(.type | startswith("AGENT_"))'

# Check for errors
cat events.jsonl | jq 'select(.type == "ERROR")'
```

## Troubleshooting

### API Key Issues

```
❌ Error: OPENAI_API_KEY environment variable is not set
```

**Solution:** Set your OpenAI API key:
```bash
export OPENAI_API_KEY="sk-..."
```

### Rate Limits

If you encounter rate limits:
- Use a lower tier model (e.g., `gpt-3.5-turbo` instead of `gpt-4`)
- Add delays between requests
- Upgrade your OpenAI account tier
- The retry examples include exponential backoff

### Import Errors

```
ModuleNotFoundError: No module named 'agents'
```

**Solution:** Install OpenAI Agents SDK:
```bash
pip install openai-agents
```

## Best Practices

1. **Always set OPENAI_API_KEY** - Required for all examples
2. **Start with openai_simple.py** - Understand the basics first
3. **Use file mode for testing** - Set `CHAUKAS_OUTPUT_MODE="file"`
4. **Review captured events** - Use jq to analyze event output
5. **Handle errors gracefully** - See retry_example.py for patterns

## Further Reading

- [OpenAI Agents SDK Documentation](https://platform.openai.com/docs/agents)
- [Chaukas SDK Documentation](https://docs.chaukas.com)
- [Event Schema Reference](../../docs/OPENAI_EVENTS.md)

## Contributing

To add new OpenAI examples:
1. Place files in this directory
2. Follow the existing naming pattern: `openai_*.py`
3. Include comprehensive comments and docstrings
4. Add error handling and retry logic
5. Update this README with the new example
