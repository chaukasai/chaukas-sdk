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

- `CHAUKAS_ENDPOINT`: Your Chaukas platform endpoint (required)
- `CHAUKAS_API_KEY`: Your API key (required)

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
