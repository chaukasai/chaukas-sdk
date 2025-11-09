# Google ADK Examples

Example demonstrating the Chaukas SDK integration with Google's Agent Development Kit (ADK).

## Event Coverage: 5/20 (25%)

Google ADK integration is currently **in early development** with basic event capture.

**Captured Events:**
- ✅ AGENT_START/END - Agent lifecycle
- ✅ MODEL_INVOCATION_START/END - LLM calls
- ✅ ERROR - Basic error tracking

**Not Yet Captured:**
- ❌ SESSION_START/END - Not implemented
- ❌ TOOL_CALL_START/END - Tool tracking needed
- ❌ INPUT_RECEIVED/OUTPUT_EMITTED - I/O tracking needed
- ❌ RETRY - Retry detection needed
- ❌ AGENT_HANDOFF - Handoff tracking needed
- ❌ MCP_CALL_START/END - MCP integration needed
- ❌ POLICY_DECISION - Not exposed by ADK
- ❌ DATA_ACCESS - Data layer not observable
- ❌ STATE_UPDATE - State tracking needed

## Example

### google_adk_example.py

**Basic integration example** - Minimal demonstration of ADK with Chaukas SDK.

**Features:**
- Agent lifecycle tracking
- LLM invocation capture
- Basic error tracking
- Simple event output

**Usage:**
```bash
export GOOGLE_API_KEY="your-google-api-key"
python google_adk_example.py
```

## Requirements

```bash
# Install Google ADK
pip install adk

# Install Chaukas SDK
pip install chaukas-sdk

# Or install with Google integration
pip install "chaukas-sdk[google]"

# Set your Google API key
export GOOGLE_API_KEY="your-google-api-key"
```

## Configuration

### Required Environment Variables

```bash
export GOOGLE_API_KEY="your-google-api-key"
```

### Optional Chaukas Configuration

```bash
# Output mode (default: file)
export CHAUKAS_OUTPUT_MODE="file"
export CHAUKAS_OUTPUT_FILE="events.jsonl"

# For API mode
export CHAUKAS_OUTPUT_MODE="api"
export CHAUKAS_ENDPOINT="https://api.chaukas.com"
export CHAUKAS_API_KEY="your-chaukas-api-key"
export CHAUKAS_TENANT_ID="your-tenant"
export CHAUKAS_PROJECT_ID="your-project"
```

## Analyzing Events

The Google ADK example currently captures basic events:

```bash
# View all events
cat events.jsonl | jq '.'

# Count event types (expect ~5 types)
cat events.jsonl | jq -r '.type' | sort | uniq -c

# View agent lifecycle
cat events.jsonl | jq 'select(.type | contains("AGENT"))'

# View LLM calls
cat events.jsonl | jq 'select(.type | contains("MODEL_INVOCATION"))'
```

## Current Limitations

The Google ADK integration is in **early development**:

1. **Limited Event Coverage** - Only 25% of events captured
2. **No Tool Tracking** - TOOL_CALL events not yet implemented
3. **No Session Management** - SESSION events not tracked
4. **Basic Error Handling** - Only simple error capture

## Roadmap

Planned improvements for Google ADK integration:

- [ ] SESSION_START/END tracking
- [ ] TOOL_CALL_START/END for tool usage
- [ ] INPUT_RECEIVED/OUTPUT_EMITTED for I/O
- [ ] RETRY detection and tracking
- [ ] AGENT_HANDOFF for multi-agent scenarios
- [ ] Enhanced error tracking with context
- [ ] Performance metrics and latency tracking

## Troubleshooting

### API Key Issues

```
Error: GOOGLE_API_KEY not set
```

**Solution:** Set your Google API key:
```bash
export GOOGLE_API_KEY="your-google-api-key"
```

### Import Errors

```
ModuleNotFoundError: No module named 'adk'
```

**Solution:** Install Google ADK:
```bash
pip install adk
```

### Limited Event Output

If you see fewer events than expected, this is normal. The Google ADK integration currently captures only basic events. See the roadmap above for planned improvements.

## Comparison with Other SDKs

| Feature | OpenAI Agents | CrewAI | Google ADK |
|---------|---------------|--------|------------|
| Event Coverage | 80% (16/20) | 100% (20/20) | 25% (5/20) |
| Maturity | Stable | Stable | Early Development |
| Session Tracking | ✅ | ✅ | ❌ |
| Tool Tracking | ✅ | ✅ | ❌ |
| Multi-Agent | ✅ | ✅ | ❌ |
| Production Ready | ✅ | ✅ | ⚠️ Development |

## Contributing

We welcome contributions to improve Google ADK integration!

**High Priority Areas:**
1. Tool call tracking (TOOL_CALL_START/END)
2. Session management (SESSION_START/END)
3. I/O tracking (INPUT_RECEIVED/OUTPUT_EMITTED)
4. Retry detection

To contribute:
1. Review the OpenAI and CrewAI wrappers for patterns
2. Implement missing event types in `src/chaukas/sdk/integrations/google_adk.py`
3. Add tests in `tests/test_google_adk.py`
4. Update this README with new capabilities
5. Submit a pull request

## Further Reading

- [Google ADK Documentation](https://cloud.google.com/generative-ai-app-builder/docs/agent-development-kit)
- [Chaukas SDK Documentation](https://docs.chaukas.com)
- [Integration Development Guide](../../docs/INTEGRATION_GUIDE.md)

## Support

For questions or issues with Google ADK integration:
- Open an issue on [GitHub](https://github.com/chaukasai/chaukas-sdk/issues)
- Check existing issues for known limitations
- Consider contributing improvements!
