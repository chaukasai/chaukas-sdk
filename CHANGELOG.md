# Changelog

All notable changes to the Chaukas SDK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-11-09

### Added

**Initial Release** - Open-source SDK implementing the [chaukas-spec](https://github.com/chaukasai/chaukas-spec) for AI agent observability

#### Framework Support

- **OpenAI Agents** (`openai-agents>=0.5.0,<1.0.0`) - **19/19 events** (100% coverage)
  - Full event capture including sessions, agents, LLM calls, tools, MCP protocol, retries, errors, and more
  - Examples: basic usage, comprehensive workflows, retry patterns, MCP integration
  - Production ready

- **CrewAI** (`crewai>=1.4.1,<2.0.0`) - **19/19 events** (100% coverage)
  - Event bus integration for complete observability
  - Support for multi-agent handoffs, knowledge sources, guardrails, and flows
  - Examples: basic crews, comprehensive scenarios, retry patterns, tools, knowledge, guardrails, and flows
  - Production ready

- **Google ADK** (latest) - **5/19 events** (26% coverage)
  - Basic agent and LLM tracking
  - Under active development

#### Core Features

- **19 Event Types** - Complete implementation of [chaukas-spec](https://github.com/chaukasai/chaukas-spec)
  - Agent lifecycle: SESSION_START/END, AGENT_START/END, AGENT_HANDOFF
  - Model operations: MODEL_INVOCATION_START/END
  - Tool execution: TOOL_CALL_START/END, MCP_CALL_START/END
  - I/O tracking: INPUT_RECEIVED, OUTPUT_EMITTED
  - Operational: ERROR, RETRY, POLICY_DECISION, DATA_ACCESS, STATE_UPDATE, SYSTEM

- **Zero-Configuration Instrumentation**
  - One-line setup: `chaukas.enable_chaukas()`
  - Automatic framework detection and patching
  - No code changes required

- **Distributed Tracing**
  - Session/trace/span hierarchy
  - UUID7 event IDs for time-ordered tracking
  - Parent-child relationship tracking

- **Flexible Output**
  - File mode: JSONL output for local development
  - API mode: gRPC streaming to Chaukas platform
  - Configurable batching and flushing

- **Production Ready**
  - Comprehensive test suite
  - Type-safe with full type hints
  - Extensive examples and documentation
  - Apache 2.0 licensed

#### Documentation & Examples

- Complete README with quick start, examples, and troubleshooting
- Framework-specific READMEs (OpenAI, CrewAI, Google ADK)
- 10+ working examples demonstrating all event types
- Architecture documentation

---

[0.1.0]: https://github.com/chaukasai/chaukas-sdk/releases/tag/v0.1.0
