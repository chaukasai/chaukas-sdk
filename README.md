# Chaukas SDK

One-line instrumentation for agent building SDKs. Provides comprehensive observability into agent workflows with distributed tracing, unified event schemas, and immutable audit trails.

## Features

- **One-line integration**: Simply call `chaukas.enable_chaukas()` 
- **Automatic SDK detection**: Supports OpenAI Agents, Google ADK, and CrewAI out of the box
- **Distributed tracing**: Full session, trace, and span tracking with parent-child relationships
- **Unified schema**: Normalizes events from different SDKs into consistent format
- **Comprehensive coverage**: Captures LLM calls, tool usage, agent handoffs, user interactions, and lifecycle events
- **Immutable audit trail**: All events are preserved for deep analysis and compliance

## Supported SDKs

- [OpenAI Agents SDK](https://github.com/openai/openai-agents-python)
- [Google ADK Python](https://github.com/google/adk-python) 
- [CrewAI](https://github.com/crewAIInc/crewAI)

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

## Event Types

The SDK captures these event types:

- **LLM Events**: `llm.request`, `llm.response`, `llm.stream.*`
- **Tool Events**: `tool.call`, `tool.response`, `tool.error`
- **Agent Events**: `agent.start`, `agent.end`, `agent.error`, `agent.handoff`
- **Session Events**: `session.start`, `session.end`
- **User Events**: `user.input`, `user.output`
- **MCP Events**: `mcp.call`, `mcp.response`
- **Guardrail Events**: `guardrail.check`, `guardrail.violation`
- **Artifact Events**: `artifact.create`, `artifact.update`

## License

MIT License# chaukas-sdk
