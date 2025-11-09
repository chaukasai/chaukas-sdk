# Chaukas SDK Examples

This directory contains comprehensive examples demonstrating how to use the Chaukas SDK with various agent frameworks.

## Directory Structure

The examples are organized by SDK/framework:

```
examples/
├── openai/          # OpenAI Agents SDK examples
├── crewai/          # CrewAI framework examples
├── google_adk/      # Google ADK examples
└── README.md        # This file
```

## Overview

| SDK | Examples | Event Coverage | Documentation |
|-----|----------|----------------|---------------|
| **OpenAI Agents** | 6 examples | 16/20 events (80%) | [openai/README.md](openai/README.md) |
| **CrewAI** | 4 examples | 20/20 events (100%) | [crewai/README.md](crewai/README.md) |
| **Google ADK** | 1 example | 5/20 events (25%) | [google_adk/README.md](google_adk/README.md) |

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

## Quick Start

Choose your SDK and explore the examples:

### OpenAI Agents SDK
```bash
# View available examples
ls examples/openai/

# Run comprehensive example
python examples/openai/openai_comprehensive_example.py

# See full documentation
cat examples/openai/README.md
```

### CrewAI
```bash
# View available examples
ls examples/crewai/

# Run basic example
python examples/crewai/crewai_example.py

# See full documentation
cat examples/crewai/README.md
```

### Google ADK
```bash
# View example
ls examples/google_adk/

# Run example
python examples/google_adk/google_adk_example.py

# See full documentation
cat examples/google_adk/README.md
```

## Event Output

All examples save events to JSONL files in their respective directories. View and analyze captured events:

```bash
# Pretty print events
cat events.jsonl | jq '.'

# Count event types
cat events.jsonl | jq -r '.type' | sort | uniq -c

# Filter specific event types
cat events.jsonl | jq 'select(.type == "AGENT_START")'
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
1. Place examples in the appropriate SDK subdirectory
2. Follow the existing pattern with comprehensive comments
3. Include event analysis functionality
4. Provide both real and simulated scenarios
5. Document all configuration requirements in the SDK's README
6. Add error handling and retry logic

## License

MIT License - See parent directory for details.