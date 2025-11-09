# CrewAI Examples

Comprehensive examples demonstrating the Chaukas SDK integration with CrewAI framework.

## Event Coverage: 20/20 (100%)

CrewAI provides **complete event coverage** through its event bus architecture!

**All Events Captured:**
- ✅ SESSION_START/END - Session lifecycle
- ✅ AGENT_START/END - Agent execution
- ✅ MODEL_INVOCATION_START/END - LLM calls
- ✅ TOOL_CALL_START/END - Tool execution
- ✅ INPUT_RECEIVED - User inputs
- ✅ OUTPUT_EMITTED - Agent outputs
- ✅ ERROR - Error tracking
- ✅ RETRY - Retry attempts
- ✅ AGENT_HANDOFF - Multi-agent handoffs
- ✅ MCP_CALL_START/END - MCP tool detection
- ✅ POLICY_DECISION - Crew decision tracking
- ✅ DATA_ACCESS - Memory access tracking
- ✅ STATE_UPDATE - Task state changes

## Examples

### 1. crewai_example.py

**Basic multi-agent example** - Simple demonstration of CrewAI with full event capture.

**Features:**
- Multiple agents with different roles
- Task delegation and collaboration
- 100% event coverage
- Event bus integration

**Usage:**
```bash
export OPENAI_API_KEY="sk-..."
export CREWAI_DISABLE_TELEMETRY=true
python crewai_example.py
```

### 2. crewai_comprehensive_example.py

**Full-featured example** - Complete demonstration with multiple crews and complex workflows.

**Features:**
- Multiple crew configurations
- Complex task hierarchies
- Agent collaboration patterns
- Memory and context management
- Comprehensive event analysis

**Usage:**
```bash
export OPENAI_API_KEY="sk-..."
export CREWAI_DISABLE_TELEMETRY=true
python crewai_comprehensive_example.py
```

### 3. crewai_retry_example.py

**Retry logic demonstration** - Shows retry detection and error recovery in CrewAI.

**Features:**
- Automatic retry detection via event bus
- Backoff strategies
- Error recovery patterns
- RETRY event capture

**Usage:**
```bash
export OPENAI_API_KEY="sk-..."
export CREWAI_DISABLE_TELEMETRY=true
python crewai_retry_example.py
```

### 4. crewai_tools_example.py

**Tool usage patterns** - Demonstrates tool implementations and MCP detection.

**Features:**
- Custom tool implementations
- Tool execution tracking
- MCP tool detection
- Performance metrics
- Tool error handling

**Usage:**
```bash
export OPENAI_API_KEY="sk-..."
export CREWAI_DISABLE_TELEMETRY=true
python crewai_tools_example.py
```

## Requirements

```bash
# Install CrewAI
pip install crewai

# Install Chaukas SDK
pip install chaukas-sdk

# Or install with CrewAI integration
pip install "chaukas-sdk[crewai]"

# Set your OpenAI API key (CrewAI uses OpenAI by default)
export OPENAI_API_KEY="sk-..."

# Disable CrewAI telemetry (prevents "Service Unavailable" errors)
export CREWAI_DISABLE_TELEMETRY=true
```

## Configuration

### Required Environment Variables

```bash
export OPENAI_API_KEY="sk-..."  # CrewAI uses OpenAI for LLM
export CREWAI_DISABLE_TELEMETRY=true  # Prevents telemetry errors
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
export CHAUKAS_FLUSH_INTERVAL="5.0"
```

## Analyzing Events

CrewAI examples capture the most comprehensive event data:

```bash
# View all events
cat events.jsonl | jq '.'

# Count all 20 event types
cat events.jsonl | jq -r '.type' | sort | uniq -c

# View agent handoffs
cat events.jsonl | jq 'select(.type == "AGENT_HANDOFF")'

# Track task lifecycle
cat events.jsonl | jq 'select(.type | contains("TASK"))'

# View memory access
cat events.jsonl | jq 'select(.type == "DATA_ACCESS")'

# Check policy decisions
cat events.jsonl | jq 'select(.type == "POLICY_DECISION")'
```

## Architecture

CrewAI integration leverages the **event bus architecture**:

1. **Event Bus Listeners** - Hooks into CrewAI's internal event system
2. **Full Observability** - Captures all crew, agent, task, and tool events
3. **Zero Boilerplate** - Automatic patching with no code changes needed
4. **Distributed Tracing** - Tracks events across agent handoffs

## Troubleshooting

### CrewAI Telemetry Errors

```
Service Unavailable encountered while exporting span batch
```

**Solution:** Disable CrewAI's built-in telemetry:
```bash
export CREWAI_DISABLE_TELEMETRY=true
```

### API Key Issues

```
Error: OPENAI_API_KEY not set
```

**Solution:** CrewAI uses OpenAI by default:
```bash
export OPENAI_API_KEY="sk-..."
```

### Import Errors

```
ModuleNotFoundError: No module named 'crewai'
```

**Solution:** Install CrewAI:
```bash
pip install crewai
```

## Best Practices

1. **Always disable telemetry** - Set `CREWAI_DISABLE_TELEMETRY=true`
2. **Start with crewai_example.py** - Understand basic patterns first
3. **Use file mode for testing** - Set `CHAUKAS_OUTPUT_MODE="file"`
4. **Analyze event sequences** - Review agent handoffs and task flows
5. **Monitor memory access** - Track DATA_ACCESS events for debugging

## Key Differences from OpenAI

| Feature | OpenAI Agents | CrewAI |
|---------|---------------|--------|
| Event Coverage | 80% (16/20) | 100% (20/20) |
| Architecture | Method patching | Event bus |
| MCP Support | ❌ | ✅ |
| Memory Tracking | ❌ | ✅ |
| Policy Decisions | ❌ | ✅ |
| Multi-Agent | Limited | Native |

## Further Reading

- [CrewAI Documentation](https://docs.crewai.com)
- [Chaukas SDK Documentation](https://docs.chaukas.com)
- [CrewAI Event Bus](https://docs.crewai.com/concepts/events)

## Contributing

To add new CrewAI examples:
1. Place files in this directory
2. Follow the naming pattern: `crewai_*.py`
3. Include comprehensive comments and docstrings
4. Leverage the event bus for full observability
5. Update this README with the new example
