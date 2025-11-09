<div align="center">

# 🔍 Chaukas SDK

**One line of code. Complete AI agent observability.**

*Production-grade instrumentation for OpenAI, CrewAI, and Google ADK*

[![PyPI version](https://badge.fury.io/py/chaukas-sdk.svg)](https://badge.fury.io/py/chaukas-sdk)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()
[![Coverage](https://img.shields.io/badge/coverage-90%25-brightgreen.svg)]()

[Quick Start](#-quick-start) • [Documentation](#-documentation) • [Examples](#-examples) • [Architecture](#-architecture) • [Community](#-community)

</div>

---

## 🎯 Why Chaukas?

Building AI agents is hard. **Understanding what they're doing is harder.**

Chaukas SDK gives you X-ray vision into your AI agents with **zero configuration**:

```python
import chaukas
chaukas.enable_chaukas()  # That's it. You're done.

# Your existing agent code works unchanged
agent = Agent(name="assistant", model="gpt-4")
result = await agent.run(messages=[...])
```

**Instantly get:**
- 🎯 Complete execution traces with distributed tracing
- 🔄 Automatic retry detection and tracking
- 🛠️ Tool call monitoring and performance metrics
- 🤝 Multi-agent handoff visualization
- 🚨 Error tracking with full context
- 📊 LLM token usage and cost tracking
- 🔐 Policy enforcement and compliance logs
- 🎨 Beautiful, queryable event streams

## ✨ What Makes Chaukas Different

| Feature | Chaukas | Traditional APM | Manual Logging |
|---------|---------|-----------------|----------------|
| **Setup Time** | 1 line | Hours | Days |
| **Code Changes** | Zero | Extensive | Everywhere |
| **Agent-Native** | ✅ 100% | ❌ Adapted | ❌ Custom |
| **Event Coverage** | 🎉 19/19 (100%) | ⚠️ Partial | 🤷 Up to you |
| **Multi-Agent Tracking** | ✅ Built-in | ❌ Manual | ❌ Complex |
| **MCP Protocol** | ✅ Native | ❌ No support | ❌ Manual |
| **Distributed Tracing** | ✅ Automatic | ⚠️ Requires setup | ❌ Hard |
| **Type Safety** | ✅ Full | ⚠️ Partial | ❌ None |

## 🚀 Quick Start

### Installation

```bash
pip install chaukas-sdk
```

### Configuration

Set your environment variables (or pass them programmatically):

```bash
export CHAUKAS_TENANT_ID="your-tenant"
export CHAUKAS_PROJECT_ID="your-project"
export CHAUKAS_ENDPOINT="https://api.chaukas.com"
export CHAUKAS_API_KEY="your-api-key"
```

### Usage

#### OpenAI Agents

```python
import chaukas
from openai import OpenAI
from openai.agents import Agent

# Enable instrumentation
chaukas.enable_chaukas()

# Your code works exactly as before
client = OpenAI()
agent = Agent(
    name="data-analyst",
    instructions="You are a helpful data analyst.",
    model="gpt-4o",
    client=client
)

result = await agent.run(
    messages=[{"role": "user", "content": "Analyze Q4 revenue"}]
)

# Chaukas automatically captures:
# ✅ Session start/end
# ✅ Agent lifecycle
# ✅ LLM invocations with tokens
# ✅ Tool calls and results
# ✅ Errors and retries
# ✅ Policy decisions
# ✅ State changes
```

#### CrewAI

```python
import chaukas
from crewai import Agent, Task, Crew, Process

chaukas.enable_chaukas()

# Define your crew
researcher = Agent(
    role="Senior Research Analyst",
    goal="Uncover cutting-edge developments in AI",
    backstory="You're an expert at finding insights",
    verbose=True
)

task = Task(
    description="Research latest AI trends",
    agent=researcher,
    expected_output="A comprehensive report"
)

crew = Crew(
    agents=[researcher],
    tasks=[task],
    process=Process.sequential
)

# Full observability out of the box
result = crew.kickoff()
```

#### Google ADK

```python
import chaukas
from adk import Agent

chaukas.enable_chaukas()

agent = Agent(name="assistant")
response = agent.run("Hello!")
```

## 📊 Supported SDKs

| SDK | Coverage | Status | Features |
|-----|----------|--------|----------|
| **[OpenAI Agents](https://github.com/openai/openai-agents-python)** | 🎉 **100%** (19/19) | 🟢 Stable | Session mgmt, MCP protocol, policy tracking, state updates, retries |
| **[CrewAI](https://github.com/crewAIInc/crewAI)** | 🎉 **100%** (19/19) | 🟢 Stable | Event bus integration, multi-agent handoffs, flow tracking |
| **[Google ADK](https://github.com/google/adk-python)** | 🟡 **26%** (5/19) | 🔄 Beta | Basic agent & LLM tracking |

*Coverage = percentage of [chaukas-spec](https://github.com/chaukas/chaukas-spec) event types captured*

## 🎨 Event Types Captured

Chaukas captures **all 19 chaukas-spec event types** automatically:

### 🎭 Agent Lifecycle
```python
SESSION_START       # User session begins
SESSION_END         # Session completes
AGENT_START         # Agent begins execution
AGENT_END           # Agent finishes
AGENT_HANDOFF       # Control transfers between agents
```

### 🧠 Model Operations
```python
MODEL_INVOCATION_START  # LLM call initiated
MODEL_INVOCATION_END    # LLM responds (includes tokens, cost)
```

### 🛠️ Tool Execution
```python
TOOL_CALL_START     # Tool execution begins
TOOL_CALL_END       # Tool completes with result
MCP_CALL_START      # Model Context Protocol call starts
MCP_CALL_END        # MCP operation completes
```

### 💬 I/O Tracking
```python
INPUT_RECEIVED      # User input captured
OUTPUT_EMITTED      # Agent output generated
```

### 🚨 Operational Intelligence
```python
ERROR               # Error with full context
RETRY               # Automatic retry detected (rate limits, timeouts)
POLICY_DECISION     # Content filtering, guardrails enforced
DATA_ACCESS         # Knowledge base, file, or API access
STATE_UPDATE        # Agent configuration changes
SYSTEM_EVENT        # Framework initialization, shutdown
```

## 🔥 Advanced Features

### Distributed Tracing

Every event includes full trace context:

```python
{
  "event_id": "019a6700-adb9-718d-0bc9-0000415845aa",
  "session_id": "019a6700-adb7-7a30-a548-000077453f71",
  "trace_id": "019a6700-adb7-7ef3-1e46-0000ae993c28",
  "span_id": "019a6700-adb9-706a-0a26-000073699939",
  "parent_span_id": "019a6700-adb7-7b27-1858-0000ee8d895b",
  "type": "EVENT_TYPE_TOOL_CALL_END",
  "agent_id": "data-analyst",
  "timestamp": "2025-01-08T12:34:56.789Z"
}
```

Visualize complete request flows across:
- Multiple agents
- LLM calls
- Tool invocations
- External API calls

### Intelligent Retry Detection

Chaukas automatically detects and tracks retries:

```python
# Your code
try:
    result = await agent.run(messages)
except RateLimitError:
    await asyncio.sleep(2)  # Exponential backoff
    result = await agent.run(messages)  # Retry

# Chaukas captures:
# 1. ERROR event (rate limit)
# 2. RETRY event (attempt 1, exponential strategy, 2000ms delay)
# 3. MODEL_INVOCATION_START (retry attempt)
# 4. MODEL_INVOCATION_END (success)
```

### MCP Protocol Support

**Only SDK with native MCP instrumentation:**

```python
from agents import Agent
from agents.mcp import MCPServerStreamableHttp

# MCP server setup
mcp_server = MCPServerStreamableHttp(
    url="http://localhost:8000",
    server_name="documentation-server"
)

agent = Agent(
    name="doc-agent",
    model="gpt-4o",
    mcp_servers=[mcp_server]
)

# Chaukas captures:
# - MCP_CALL_START (get_prompt request)
# - MCP_CALL_END (prompt retrieved, 245ms)
# - Full request/response payloads
```

### Policy Decision Tracking

Monitor content filtering and guardrails:

```python
# When OpenAI filters content
response = await agent.run(messages)

# Chaukas automatically captures:
{
  "type": "EVENT_TYPE_POLICY_DECISION",
  "policy_id": "openai_content_policy",
  "outcome": "blocked",
  "rule_ids": ["content_filter"],
  "rationale": "Response blocked due to: content_filter",
  "finish_reason": "content_filter"
}
```

### State Change Tracking

Track agent configuration changes:

```python
# Agent configuration updated
agent.temperature = 0.7
agent.instructions = "Be more creative"

# Chaukas captures the diff:
{
  "type": "EVENT_TYPE_STATE_UPDATE",
  "state_update": {
    "temperature": {"old": 0.3, "new": 0.7},
    "instructions": {
      "old": "Be precise",
      "new": "Be more creative"
    }
  }
}
```

### Multi-Agent Handoffs

Visualize agent collaboration:

```python
# CrewAI agent handoff
task.context = [previous_task]

# Chaukas captures:
{
  "type": "EVENT_TYPE_AGENT_HANDOFF",
  "from_agent_id": "researcher",
  "to_agent_id": "writer",
  "handoff_reason": "task_delegation",
  "context_data": {...}
}
```

## ⚙️ Configuration

### Environment Variables

#### Required
```bash
CHAUKAS_TENANT_ID       # Your tenant identifier
CHAUKAS_PROJECT_ID      # Your project identifier
CHAUKAS_ENDPOINT        # API endpoint (api mode)
CHAUKAS_API_KEY         # Authentication key (api mode)
```

#### Optional
```bash
CHAUKAS_OUTPUT_MODE="api"              # "api" or "file"
CHAUKAS_OUTPUT_FILE="events.jsonl"     # File path (file mode)
CHAUKAS_BATCH_SIZE=20                  # Events per batch
CHAUKAS_MAX_BATCH_BYTES=262144         # Max batch size (256KB)
CHAUKAS_FLUSH_INTERVAL=5.0             # Auto-flush interval (seconds)
CHAUKAS_TIMEOUT=30.0                   # Request timeout (seconds)
CHAUKAS_BRANCH="main"                  # Git branch for context
CHAUKAS_TAGS="prod,us-east-1"          # Custom tags
```

#### Framework-Specific
```bash
CREWAI_DISABLE_TELEMETRY=true  # Disable CrewAI's telemetry
```

### Programmatic Configuration

```python
import chaukas

chaukas.enable_chaukas(
    tenant_id="acme-corp",
    project_id="production",
    endpoint="https://observability.acme.com",
    api_key="sk-proj-...",
    session_id="custom-session-123",  # Optional custom session
    config={
        "auto_detect": True,          # Auto-detect installed SDKs
        "enabled_integrations": [     # Or specify explicitly
            "openai_agents",
            "crewai"
        ],
        "batch_size": 100,
        "flush_interval": 10.0,
        "timeout": 60.0,
    }
)
```

### File Output Mode (Development)

Perfect for local development and testing:

```python
import os
os.environ["CHAUKAS_OUTPUT_MODE"] = "file"
os.environ["CHAUKAS_OUTPUT_FILE"] = "agent_events.jsonl"

import chaukas
chaukas.enable_chaukas()

# Events written to agent_events.jsonl
# Analyze with: cat agent_events.jsonl | jq .type | sort | uniq -c
```

## 📖 Examples

### Example 1: Debug LLM Token Usage

```python
import chaukas
chaukas.enable_chaukas()

# Run your agent
result = await agent.run(messages)

# Query events:
# cat events.jsonl | jq 'select(.type=="EVENT_TYPE_MODEL_INVOCATION_END") | .model_invocation.usage'

# Output:
{
  "prompt_tokens": 234,
  "completion_tokens": 456,
  "total_tokens": 690,
  "estimated_cost_usd": 0.0207
}
```

### Example 2: Track Multi-Agent Workflow

```python
import chaukas
from crewai import Agent, Task, Crew, Process

chaukas.enable_chaukas()

# Define a multi-agent crew
researcher = Agent(role="Researcher", goal="Find insights")
writer = Agent(role="Writer", goal="Write report")

research_task = Task(description="Research AI trends", agent=researcher)
writing_task = Task(
    description="Write report",
    agent=writer,
    context=[research_task]  # Handoff point
)

crew = Crew(agents=[researcher, writer], tasks=[research_task, writing_task])
result = crew.kickoff()

# Chaukas captures:
# 1. SESSION_START
# 2. AGENT_START (researcher)
# 3. MODEL_INVOCATION_* (researcher's LLM calls)
# 4. AGENT_END (researcher)
# 5. AGENT_HANDOFF (researcher → writer)
# 6. AGENT_START (writer)
# 7. MODEL_INVOCATION_* (writer's LLM calls)
# 8. AGENT_END (writer)
# 9. SESSION_END
```

### Example 3: Monitor Tool Execution

```python
import chaukas
from openai import OpenAI
from openai.agents import Agent

chaukas.enable_chaukas()

def search_database(query: str) -> str:
    """Search the product database."""
    # Slow database query
    import time
    time.sleep(2)
    return f"Results for: {query}"

agent = Agent(
    name="support-agent",
    model="gpt-4o",
    tools=[search_database]
)

result = await agent.run(messages=[
    {"role": "user", "content": "Find product XYZ"}
])

# Chaukas captures tool performance:
# TOOL_CALL_START → TOOL_CALL_END
# Duration: 2.1s (flag for optimization!)
```

### Example 4: Detect Rate Limit Issues

```python
import chaukas
chaukas.enable_chaukas()

# Your code encounters rate limits
for i in range(100):
    try:
        result = await agent.run(messages)
    except RateLimitError as e:
        await asyncio.sleep(2 ** i)  # Exponential backoff
        continue

# Query retry events:
# cat events.jsonl | jq 'select(.type=="EVENT_TYPE_RETRY")'

# Output shows patterns:
# - 15 retries in last hour
# - Average backoff: 4.2s
# - All due to rate limits (429)
# → Action: Implement request throttling
```

## 🏗️ Architecture

### How It Works

```
┌─────────────────────────────────────────────────────────────┐
│  Your Application                                           │
│                                                             │
│  ┌─────────────┐         ┌──────────────┐                 │
│  │ OpenAI      │         │   CrewAI     │                 │
│  │ Agent       │         │   Crew       │                 │
│  └──────┬──────┘         └──────┬───────┘                 │
│         │                       │                          │
│         └───────────┬───────────┘                          │
│                     │                                      │
│         ┌───────────▼───────────┐                         │
│         │  Chaukas SDK          │   (Monkey patching)     │
│         │  - Auto-detection     │                         │
│         │  - Event capture      │                         │
│         │  - Distributed trace  │                         │
│         └───────────┬───────────┘                         │
│                     │                                      │
└─────────────────────┼──────────────────────────────────────┘
                      │
          ┌───────────▼───────────┐
          │  Intelligent Batching │
          │  - Adaptive sizing    │
          │  - Auto-retry         │
          │  - Memory-efficient   │
          └───────────┬───────────┘
                      │
          ┌───────────▼───────────┐
          │   Transmission        │
          │   - gRPC (API mode)   │
          │   - File (Dev mode)   │
          └───────────┬───────────┘
                      │
          ┌───────────▼───────────┐
          │  Chaukas Platform     │
          │  - Storage            │
          │  - Querying           │
          │  - Visualization      │
          └───────────────────────┘
```

### Event Flow

```python
Agent.run() called
    │
    ├─→ SESSION_START (first call)
    │
    ├─→ AGENT_START
    │
    ├─→ INPUT_RECEIVED (user message)
    │
    ├─→ MODEL_INVOCATION_START
    │   │
    │   └─→ [LLM processes]
    │
    ├─→ MODEL_INVOCATION_END (with tokens)
    │
    ├─→ TOOL_CALL_START (if tools requested)
    │   │
    │   └─→ [Tool executes]
    │
    ├─→ TOOL_CALL_END (with result)
    │
    ├─→ OUTPUT_EMITTED (agent response)
    │
    ├─→ AGENT_END
    │
    └─→ SESSION_END (on cleanup)
```

### Distributed Tracing Hierarchy

```
Session (lifetime of user interaction)
  │
  ├─ Trace (single request/response)
  │   │
  │   ├─ Agent Span (agent execution)
  │   │   │
  │   │   ├─ LLM Span (model call)
  │   │   │
  │   │   ├─ Tool Span (tool execution)
  │   │   │   │
  │   │   │   └─ MCP Span (MCP protocol call)
  │   │   │
  │   │   └─ Tool Span (another tool)
  │   │
  │   └─ Agent Span (handoff to second agent)
  │       │
  │       └─ LLM Span
  │
  └─ Trace (follow-up request)
      └─ ...
```

## 🎯 Use Cases

### Production Monitoring
- Track agent reliability and uptime
- Monitor LLM token costs in real-time
- Detect performance regressions
- Alert on error spikes

### Debugging & Development
- Reproduce issues with full trace context
- Understand agent decision-making
- Optimize tool execution performance
- Test multi-agent workflows

### Compliance & Audit
- Immutable audit trail of all interactions
- Track policy enforcement decisions
- Monitor data access patterns
- Generate compliance reports

### Cost Optimization
- Identify expensive LLM calls
- Track token usage by agent/model
- Find opportunities for caching
- Optimize prompt engineering

## 🔧 Batching & Performance

### Adaptive Batching

Chaukas implements intelligent batching to optimize performance:

```python
┌─────────────────────────────────────────┐
│  Event Buffer                           │
│                                         │
│  Events accumulate until:              │
│  • batch_size reached (default: 20)    │
│  • max_batch_bytes reached (256KB)     │
│  • flush_interval elapsed (5s)         │
└─────────────┬───────────────────────────┘
              │
              ▼
    ┌─────────────────┐
    │ Send to Server  │
    └────┬────────────┘
         │
    Success? ────Yes──→ ✅ Done
         │
        No (503)
         │
         ▼
    ┌────────────────────┐
    │ Split batch in half│
    │ Retry both halves  │
    └────────────────────┘
```

### Performance Characteristics

- **Overhead**: < 1% CPU impact
- **Memory**: ~10MB for 1000 events buffered
- **Latency**: < 5ms per event capture
- **Network**: Batched transmission reduces API calls by 95%

### Tuning for Your Use Case

```python
# High-volume production (optimize throughput)
chaukas.enable_chaukas(config={
    "batch_size": 200,
    "max_batch_bytes": 1_048_576,  # 1MB
    "flush_interval": 30.0
})

# Real-time debugging (optimize latency)
chaukas.enable_chaukas(config={
    "batch_size": 1,
    "flush_interval": 0.1
})

# Memory-constrained (optimize memory)
chaukas.enable_chaukas(config={
    "batch_size": 10,
    "max_batch_bytes": 65536,  # 64KB
    "flush_interval": 2.0
})
```

## 🐛 Troubleshooting

### Common Issues

#### CrewAI "Service Unavailable" Errors

**Problem**: Seeing "Transient error Service Unavailable" when using CrewAI

**Cause**: CrewAI's built-in telemetry trying to send data to their servers

**Solution**:
```bash
export CREWAI_DISABLE_TELEMETRY=true
```

This only disables CrewAI's telemetry. Chaukas continues capturing events normally.

#### Events Not Appearing

**Problem**: No events in output file or API

**Solution**:
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("chaukas.sdk").setLevel(logging.DEBUG)

# Verify configuration
import chaukas
chaukas.enable_chaukas()
print(chaukas.get_config())  # Check settings

# Force flush before exit
chaukas.flush()
chaukas.disable_chaukas()
```

#### High Memory Usage

**Problem**: Memory consumption increasing over time

**Cause**: Large batches accumulating

**Solution**:
```python
# Reduce batch size and increase flush frequency
chaukas.enable_chaukas(config={
    "batch_size": 10,
    "max_batch_bytes": 65536,
    "flush_interval": 1.0
})
```

#### 503 Errors from API

**Problem**: Server returning "high memory" errors

**Cause**: Batches too large

**Solution**: SDK automatically splits batches and retries. If persistent:
```python
chaukas.enable_chaukas(config={
    "max_batch_bytes": 131072,  # Reduce to 128KB
    "batch_size": 50             # Smaller batch count
})
```

## 📚 Documentation

- **[Full Documentation](https://docs.chaukas.com)** - Complete API reference
- **[OpenAI Events Guide](./docs/OPENAI_EVENTS.md)** - OpenAI Agents event details
- **[CrewAI Events Guide](./docs/CREWAI_EVENTS.md)** - CrewAI event details
- **[Architecture Deep Dive](./docs/ARCHITECTURE.md)** - Technical architecture
- **[Event Schema](https://github.com/chaukas/chaukas-spec)** - chaukas-spec definition
- **[Examples Repository](./examples)** - Complete working examples

## 🧪 Development

### Setup

```bash
git clone https://github.com/chaukas/chaukas-sdk
cd chaukas-sdk
pip install -e ".[dev]"
```

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=chaukas

# Specific test file
pytest tests/test_openai_events.py -v

# Watch mode
pytest-watch
```

### Code Quality

```bash
# Format code
black src/ tests/ examples/

# Sort imports
isort src/ tests/ examples/

# Type checking
mypy src/chaukas/

# Run all checks
make lint
```

### Running Examples

```bash
# OpenAI Agents example
python examples/openai/openai_comprehensive_example.py

# CrewAI example
python examples/crewai/crewai_example.py

# Analyze captured events
cat events.jsonl | jq .type | sort | uniq -c
```

## 🤝 Contributing

We love contributions! Whether it's:

- 🐛 Bug reports
- 💡 Feature requests
- 📖 Documentation improvements
- 🔧 Code contributions

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**: Follow our coding standards
4. **Add tests**: Ensure 90%+ coverage
5. **Run the test suite**: `pytest`
6. **Commit your changes**: `git commit -m 'feat: add amazing feature'`
7. **Push to the branch**: `git push origin feature/amazing-feature`
8. **Open a Pull Request**: Describe your changes

### Development Guidelines

- Follow [Conventional Commits](https://www.conventionalcommits.org/)
- Write tests for all new features
- Update documentation for API changes
- Maintain 90%+ test coverage
- Use type hints for all functions

## 🌟 Community

- **[Discord](https://discord.gg/chaukas)** - Chat with the community
- **[GitHub Discussions](https://github.com/chaukas/chaukas-sdk/discussions)** - Ask questions, share ideas
- **[Twitter](https://twitter.com/chaukasai)** - Follow for updates
- **[Blog](https://blog.chaukas.com)** - Technical articles and guides
- **[YouTube](https://youtube.com/chaukas)** - Video tutorials

## 📬 Support

- **[GitHub Issues](https://github.com/chaukas/chaukas-sdk/issues)** - Bug reports and feature requests
- **[Email](mailto:support@chaukas.com)** - Direct support
- **[Documentation](https://docs.chaukas.com)** - Comprehensive guides
- **[Stack Overflow](https://stackoverflow.com/questions/tagged/chaukas)** - Community Q&A (tag: `chaukas`)

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details

---

<div align="center">

**Built with ❤️ by the Chaukas team**

[Website](https://chaukas.com) • [Documentation](https://docs.chaukas.com) • [GitHub](https://github.com/chaukas)

</div>
