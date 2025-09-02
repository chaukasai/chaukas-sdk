# Chaukas SDK

One-line instrumentation for agent building SDKs. Provides comprehensive observability into agent workflows with distributed tracing, unified event schemas, and immutable audit trails.

## Features

- **One-line integration**: Simply call `chaukas.enable_chaukas()` 
- **Automatic SDK detection**: Supports OpenAI Agents, Google ADK, and CrewAI out of the box
- **Distributed tracing**: Full session, trace, and span tracking with parent-child relationships
- **Unified schema**: Normalizes events from different SDKs into consistent format
- **100% Event Coverage**: Captures all 20 chaukas-spec event types including retry attempts
- **Comprehensive observability**: Tracks LLM calls, tool usage, agent handoffs, retries, errors, and lifecycle events
- **Immutable audit trail**: All events are preserved for deep analysis and compliance

## Supported SDKs

| SDK | Event Coverage | Key Features |
|-----|---------------|--------------|
| [CrewAI](https://github.com/crewAIInc/crewAI) | **100%** (20/20) | Full event bus integration, retry tracking, multi-agent handoffs |
| [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) | **80%** (16/20) | Session management, retry detection, tool tracking, I/O events |
| [Google ADK Python](https://github.com/google/adk-python) | **25%** (5/20) | Basic agent and LLM tracking |

## Quick Start

### Installation

```bash
pip install chaukas-sdk
```

### Basic Usage

```python
import chaukas

# Set environment variables
# CHAUKAS_ENDPOINT=https://api.chaukas.com
# CHAUKAS_API_KEY=your-api-key

# Enable instrumentation (reads from environment)
chaukas.enable_chaukas()

# Your existing agent code works unchanged
# All agent interactions are automatically traced
```

### OpenAI Agents Example

```python
import chaukas
from openai.agents import Agent

chaukas.enable_chaukas()

agent = Agent(name="assistant", model="gpt-4")
result = await agent.run(messages=[...])  # Automatically traced
```

### CrewAI Example

```python
import chaukas
from crewai import Agent, Task, Crew

chaukas.enable_chaukas()

crew = Crew(agents=[...], tasks=[...])
result = crew.kickoff()  # Automatically traced
```

## Configuration

### Environment Variables

#### Required Variables
- `CHAUKAS_TENANT_ID`: Your Chaukas tenant identifier (required)
- `CHAUKAS_PROJECT_ID`: Your Chaukas project identifier (required)
- `CHAUKAS_ENDPOINT`: Your Chaukas platform endpoint (required for API mode)
- `CHAUKAS_API_KEY`: Your API key (required for API mode)

#### Optional Variables
- `CHAUKAS_OUTPUT_MODE`: Output mode - "api" or "file" (default: "api")
- `CHAUKAS_OUTPUT_FILE`: File path for file mode (default: "chaukas_events.jsonl")
- `CHAUKAS_BATCH_SIZE`: Number of events per batch (default: 20)
- `CHAUKAS_MAX_BATCH_BYTES`: Maximum batch size in bytes (default: 262144, i.e., 256KB)
- `CHAUKAS_MIN_BATCH_SIZE`: Minimum batch size for retry (default: 1)
- `CHAUKAS_ENABLE_ADAPTIVE_BATCHING`: Enable size-based batching (default: true)
- `CHAUKAS_FLUSH_INTERVAL`: Seconds between automatic flushes (default: 5.0)
- `CHAUKAS_TIMEOUT`: Request timeout in seconds (default: 30.0)
- `CHAUKAS_BRANCH`: Git branch name for context (optional)
- `CHAUKAS_TAGS`: Comma-separated tags for filtering (optional)

#### Framework-Specific Variables
- `CREWAI_DISABLE_TELEMETRY`: Set to "true" to disable CrewAI's telemetry and prevent "Service Unavailable" errors

### Advanced Configuration

```python
import chaukas

chaukas.enable_chaukas(
    endpoint="https://custom.endpoint.com",  # Override env var
    api_key="custom-key",                    # Override env var
    session_id="custom-session-123",         # Optional session ID
    config={
        "auto_detect": True,                 # Auto-detect installed SDKs
        "enabled_integrations": ["openai_agents", "crewai"],  # Explicit list
        "batch_size": 100,                   # Event batching
        "flush_interval": 5.0,               # Auto-flush interval
    }
)
```

## Batching and Performance

The SDK implements intelligent batching to optimize event transmission and prevent memory issues:

### Adaptive Batching Features
- **Automatic size-based flushing**: Events are sent when batch reaches size limit (not just count)
- **Dynamic batch splitting**: On 503 errors, batches are automatically split and retried
- **Memory-efficient processing**: Prevents high memory usage errors
- **Configurable thresholds**: Fine-tune batching behavior for your use case

### How It Works
1. Events are queued until batch size or byte limit is reached
2. If the server returns a 503 high memory error, the batch is automatically split in half
3. Retry continues with progressively smaller batches until successful
4. All events are preserved - no data loss during retries

### Performance Tuning
- For high-volume applications, increase `CHAUKAS_BATCH_SIZE` and `CHAUKAS_MAX_BATCH_BYTES`
- For real-time applications, decrease `CHAUKAS_BATCH_SIZE` and `CHAUKAS_FLUSH_INTERVAL`
- For memory-constrained environments, decrease `CHAUKAS_MAX_BATCH_BYTES`

## Troubleshooting

### CrewAI "Service Unavailable" Errors
If you see errors like "Transient error Service Unavailable encountered while exporting span batch" when using CrewAI, this is from CrewAI's own telemetry system trying to send data to their servers. To disable it:

```bash
export CREWAI_DISABLE_TELEMETRY=true
```

Or in your Python script:
```python
import os
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
```

This only disables CrewAI's telemetry - Chaukas SDK will continue to capture all events normally.

## Development

### Setup

```bash
git clone <repository>
cd chaukas-sdk
pip install -e ".[dev]"
```

### Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_client.py

# Run with coverage
pytest --cov=chaukas
```

### Code Quality

```bash
# Format code
black chaukas/ tests/ examples/

# Sort imports
isort chaukas/ tests/ examples/

# Type checking
mypy chaukas/
```

## Event Coverage

The SDK achieves **100% coverage** of all 20 chaukas-spec event types:

### Session & Agent Events
- `SESSION_START` / `SESSION_END` - Track entire user sessions
- `AGENT_START` / `AGENT_END` - Agent execution lifecycle
- `AGENT_HANDOFF` - Task delegation between agents

### Model & Tool Events  
- `MODEL_INVOCATION_START` / `MODEL_INVOCATION_END` - LLM API calls
- `TOOL_CALL_START` / `TOOL_CALL_END` - Tool executions
- `MCP_CALL_START` / `MCP_CALL_END` - Model Context Protocol calls

### I/O Events
- `INPUT_RECEIVED` - User inputs and prompts
- `OUTPUT_EMITTED` - Agent responses and outputs

### Operational Events
- `ERROR` - Error tracking with recovery information
- `RETRY` - Automatic retry detection with backoff strategies *(NEW)*
- `POLICY_DECISION` - Policy enforcement and guardrails
- `DATA_ACCESS` - Data source access tracking
- `STATE_UPDATE` - Agent state changes
- `SYSTEM` - General system events

### RETRY Event Details (NEW)
The SDK now automatically captures retry attempts when:
- LLM calls fail due to rate limits (429) or service errors (503)
- Tool executions timeout or experience network failures
- Tasks fail with retryable errors

Each RETRY event includes:
- Attempt number and maximum attempts
- Backoff strategy (exponential, linear, immediate)
- Delay in milliseconds before next attempt
- Detailed reason for the retry
- Agent context for traceability

## License

MIT License# chaukas-sdk
