# Changelog

All notable changes to the Chaukas SDK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - TBD

### Added

- **LangChain** (`langchain>=0.1.0,<2.0`) - **17/19 events** (~95% coverage)
  - Full event capture for LangChain chains, agents, and tools
  - Support for LCEL (LangChain Expression Language) pipelines
  - Automatic retry detection and tracking
  - MCP protocol support (via `mcp_server` metadata)
  - Streaming output capture (via `on_text` callback)
  - Custom events (via `on_custom_event` callback)
  - Examples: basic chains, agents with tools, LCEL pipelines, RAG, streaming
  - Production ready
  - **Note:** POLICY_DECISION and STATE_UPDATE require app-level instrumentation
  - **Note:** LangChain 1.x requires Python ≥3.10

### Changed

- **OpenAI Agents** - Event coverage updated to **18/19 events** (94.7%)
  - RETRY events are not supported for OpenAI Agents SDK
  - Reason: OpenAI SDK performs retries internally within its HTTP client layer, making retry attempts invisible to external instrumentation
  - All other event types fully supported including ERROR events (captured after SDK retries are exhausted)
  - Removed retry-related examples from OpenAI documentation

### Fixed

- **OpenAI Agents** - Type safety improvements
  - Replaced fragile string comparisons with proper `isinstance()` type checking
  - Added imports for OpenAI tool types: `FileSearchTool`, `WebSearchTool`, `CodeInterpreterTool`, `HostedMCPTool`, `LocalShellTool`, `ComputerTool`
  - Added imports for OpenAI exception types: `APIError`, `RateLimitError`, `APIConnectionError`, `Timeout`, `AuthenticationError`
  - Fixed undefined variable bug in `on_llm_end` hook
  - Fixed invalid `finish_reason` values (removed non-existent 'content_policy' and 'moderation' checks)

- **OpenAI Agents** - Code quality improvements
  - Reduced code duplication (~48 lines removed, 24% reduction)
  - Created helper methods for attribute access (`_get_agent_id`, `_get_agent_name`, `_get_model`, etc.)
  - Consolidated async/sync method pairs with shared helpers
  - Improved tool span lookup from O(n) to O(1) performance
  - Added `_parse_llm_response()` for safe response validation
  - Moved `asyncio` import to module level for better performance

- **OpenAI Agents** - Enhanced data access tracking
  - Added support for `LocalShellTool` (datasource: "local_shell")
  - Added support for `ComputerTool` (datasource: "computer")
  - Now tracks 5 data access tool types (previously 3)

- **OpenAI Agents** - Error handling improvements
  - Fixed ERROR events not being emitted in error scenarios
  - Added AGENT_END events (FAILED status) when errors occur
  - Improved agent stack management for proper cleanup on errors

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
