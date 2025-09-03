# Chaukas SDK Examples

This directory contains comprehensive examples demonstrating how to use the Chaukas SDK with various agent frameworks.

## Overview

| Example | Framework | Features | Event Coverage |
|---------|-----------|----------|----------------|
| `openai_comprehensive_example.py` | OpenAI Agents | Real API calls, 5 scenarios, tool usage | 16/20 events |
| `openai_retry_example.py` | OpenAI Agents | Retry logic demonstration | Focus on RETRY events |
| `crewai_example.py` | CrewAI | Multi-agent collaboration | 20/20 events |
| `crewai_retry_example.py` | CrewAI | Retry with event bus | RETRY event capture |
| `crewai_tools_example.py` | CrewAI | Tool usage patterns | TOOL_CALL events |
| `google_adk_example.py` | Google ADK | Basic agent tracking | 5/20 events |

## Requirements

### General Requirements
```bash
pip install chaukas-sdk
```

### Framework-Specific Requirements

#### OpenAI Agents
```bash
pip install openai-agents

# Set your API key
export OPENAI_API_KEY="your-api-key-here"
```

#### CrewAI
```bash
pip install crewai

# Set your API key (CrewAI uses OpenAI)
export OPENAI_API_KEY="your-api-key-here"

# Optional: Disable CrewAI telemetry to avoid errors
export CREWAI_DISABLE_TELEMETRY=true
```

#### Google ADK
```bash
pip install google-adk

# Set your API key
export GOOGLE_API_KEY="your-api-key-here"
```

## OpenAI Comprehensive Example

The most feature-rich example demonstrating real API integration with the OpenAI Agents SDK.

### Features

- **Real API Calls**: Uses actual OpenAI API (no mocks)
- **API Key Validation**: Checks for OPENAI_API_KEY at startup
- **5 Comprehensive Scenarios**: Different agent configurations and use cases
- **Tool Implementations**: Custom tools with `@function_tool` decorator
- **Error Handling**: Retry logic with exponential backoff
- **Event Analysis**: Built-in event analysis and reporting

### Scenarios

1. **Research Assistant**
   - Uses web_search and mcp_context_fetch tools
   - Demonstrates tool calling patterns
   - Information retrieval and summarization

2. **Math Tutor**
   - Calculator tool for computations
   - Step-by-step problem solving
   - Educational interactions

3. **Travel Planner**
   - Weather API integration
   - Retry logic demonstration
   - Multi-turn conversations

4. **Error Handling Demo**
   - API error simulation
   - Recovery strategies
   - Graceful failure handling

5. **Multi-Agent Handoff**
   - Sequential agent collaboration
   - Context passing between agents
   - Specialized agent roles

### Tools

| Tool | Description | Purpose |
|------|-------------|---------|
| `web_search` | Simulates web search | Information retrieval |
| `calculator` | Safe math evaluation | Numerical computations |
| `weather_api` | Weather information | Location-based data |
| `get_current_time` | Time in UTC/local | Temporal queries |
| `mcp_context_fetch` | MCP protocol simulation | Context retrieval |

### Running the Example

```bash
# Set your API key
export OPENAI_API_KEY="sk-..."

# Run the interactive example
python examples/openai_comprehensive_example.py

# View help
python examples/openai_comprehensive_example.py --help
```

### Interactive Menu

The example provides an interactive menu:
```
1. Research Assistant (web search & tools)
2. Math Tutor (calculator)
3. Travel Planner (weather & retries)
4. Error Handling Demo
5. Multi-Agent Handoff
6. Run All Scenarios
7. Analyze Captured Events
0. Exit
```

### Event Coverage (16/20)

**Captured Events:**
- ✅ SESSION_START/END - Session lifecycle
- ✅ AGENT_START/END - Agent execution
- ✅ MODEL_INVOCATION_START/END - LLM calls
- ✅ TOOL_CALL_START/END - Tool execution
- ✅ INPUT_RECEIVED - User inputs
- ✅ OUTPUT_EMITTED - Agent outputs
- ✅ ERROR - Error tracking
- ✅ RETRY - Retry attempts
- ✅ AGENT_HANDOFF - Agent collaboration

**Not Captured** (framework limitations):
- ❌ MCP_CALL_START/END - No native MCP support
- ❌ POLICY_DECISION - Not exposed by SDK
- ❌ DATA_ACCESS - No data system
- ❌ STATE_UPDATE - State not observable

## OpenAI Retry Example

Demonstrates retry logic and error handling with the OpenAI Agents SDK.

### Features
- Simulated API failures
- Exponential backoff strategies
- RETRY event capture
- Error recovery patterns

### Usage
```bash
python examples/openai_retry_example.py
```

## CrewAI Examples

### CrewAI Basic Example

Multi-agent collaboration with full event coverage.

```bash
python examples/crewai_example.py
```

**Features:**
- Multiple agents with different roles
- Task delegation and handoffs
- 100% event coverage (20/20)
- Event bus integration

### CrewAI Retry Example

Retry logic with CrewAI's event bus.

```bash
python examples/crewai_retry_example.py
```

**Features:**
- Automatic retry detection
- Backoff strategies
- Event bus listeners
- Error recovery

### CrewAI Tools Example

Tool usage patterns in CrewAI.

```bash
python examples/crewai_tools_example.py
```

**Features:**
- Custom tool implementations
- Tool execution tracking
- MCP tool detection
- Performance metrics

## Google ADK Example

Basic integration with Google's ADK.

```bash
python examples/google_adk_example.py
```

**Features:**
- Agent lifecycle tracking
- LLM invocation capture
- Basic event coverage (5/20)

## Event Output

All examples save events to JSONL files:
- `openai_comprehensive_output.jsonl`
- `crewai_output.jsonl`
- `google_adk_output.jsonl`

View events:
```bash
# Pretty print events
cat openai_comprehensive_output.jsonl | jq '.'

# Count event types
cat openai_comprehensive_output.jsonl | jq -r '.type' | sort | uniq -c
```

## Configuration

### Environment Variables

All examples use these Chaukas configuration variables:
```bash
export CHAUKAS_TENANT_ID="demo_tenant"
export CHAUKAS_PROJECT_ID="your_project"
export CHAUKAS_OUTPUT_MODE="file"  # or "api"
export CHAUKAS_OUTPUT_FILE="events.jsonl"
export CHAUKAS_BATCH_SIZE="1"  # Immediate write for demos
```

### API Mode

To send events to Chaukas platform:
```bash
export CHAUKAS_OUTPUT_MODE="api"
export CHAUKAS_ENDPOINT="https://api.chaukas.com"
export CHAUKAS_API_KEY="your-chaukas-api-key"
```

## Troubleshooting

### OpenAI API Key Issues
```
❌ Error: OPENAI_API_KEY environment variable is not set
```
**Solution:** Set your OpenAI API key:
```bash
export OPENAI_API_KEY="sk-..."
```

### Rate Limits
The examples include retry logic for rate limits. If you encounter persistent rate limits:
- Use a lower rate tier model (e.g., `gpt-3.5-turbo` instead of `gpt-4`)
- Add delays between requests
- Upgrade your OpenAI account tier

### CrewAI Telemetry Errors
```
Service Unavailable encountered while exporting span batch
```
**Solution:** Disable CrewAI telemetry:
```bash
export CREWAI_DISABLE_TELEMETRY=true
```

### Module Import Errors
```
ModuleNotFoundError: No module named 'agents'
```
**Solution:** Install the required framework:
```bash
pip install openai-agents  # For OpenAI
pip install crewai         # For CrewAI
pip install google-adk     # For Google ADK
```

## Contributing

To add new examples:
1. Follow the existing pattern with comprehensive comments
2. Include event analysis functionality
3. Provide both real and simulated scenarios
4. Document all configuration requirements
5. Add error handling and retry logic

## License

MIT License - See parent directory for details.