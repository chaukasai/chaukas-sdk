# Changelog

All notable changes to the Chaukas SDK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **MCP (Model Context Protocol) Event Support for OpenAI Agents** - NEW! 🎉
  - Added MCP_CALL_START and MCP_CALL_END event capture
  - Patches `_MCPServerWithClientSession` for proper method interception
  - Captures `get_prompt()` and `call_tool()` operations with full context
  - Includes server name, URL, operation type, request/response data, execution time
  - Works with `MCPServerStreamableHttp` and other MCP server implementations
  - Increased OpenAI Agents coverage from 80% to 84% (14/19 → 16/19 events)
  - Further increased to 100% with POLICY_DECISION, STATE_UPDATE, and SYSTEM_EVENT

- **Enhanced OpenAI Agents Integration** - Achieved 100% event coverage (19/19 events) 🎉
  - Created `OpenAIAgentsWrapper` using base classes and utilities
  - Added SESSION_START/END lifecycle tracking
  - Added TOOL_CALL_START/END tracking from LLM responses
  - Added INPUT_RECEIVED and OUTPUT_EMITTED for I/O tracking
  - Added RETRY event support with intelligent error detection
  - Added AGENT_HANDOFF for multi-agent control transfer
  - Added DATA_ACCESS for tool-based data retrieval tracking
  - Added POLICY_DECISION for content filtering and policy enforcement
  - Added STATE_UPDATE for agent configuration change tracking
  - Added SYSTEM_EVENT for SDK lifecycle and operational events
  - Session management with automatic start/end
  - Comprehensive test suite with 10+ test cases
  - Working examples with retry simulation and MCP integration

- **RETRY Event Support for CrewAI Integration** - Achieved 100% chaukas-spec event coverage
  - Implemented automatic retry detection for LLM calls, tool executions, and task failures
  - Added intelligent retry tracking with attempt counters per agent/model/tool
  - Supports multiple backoff strategies: exponential, linear, and immediate
  - Automatically detects retryable errors based on error patterns (rate limits, timeouts, network errors)
  - Retry counters reset on successful completion to handle multiple failure scenarios
  - Created comprehensive test suite for retry event validation
  - Added two example scripts demonstrating retry event capture

- **Reusable Integration Components**
  - Created `BaseIntegrationWrapper` base class for common functionality
  - Extracted `RetryDetector` utility for retry tracking across integrations
  - Extracted `EventPairManager` for START/END event correlation
  - Reduced code duplication by ~40% across integrations

### Enhanced
- **OpenAI Agents Support**
  - Increased event coverage from 25% to 100% (5 events → 19 events) 🎉
  - Added MCP protocol support (unique to OpenAI Agents integration)
  - Added POLICY_DECISION, STATE_UPDATE, and SYSTEM_EVENT for complete coverage
  - MonkeyPatcher now uses enhanced wrapper when available
  - Automatic fallback to standard wrapper for compatibility

- **CrewAI Event Bus Listener** 
  - Extended error handlers to detect and emit RETRY events
  - Added `_is_retryable_error()` helper method for intelligent error classification
  - Integrated retry tracking into existing LLM, tool, task, and agent error handlers

### Fixed
- **CrewAI Integration Test** - Updated to work with refactored wrapper architecture

## [0.1.0] - 2024-01-01

### Added
- Initial release with support for OpenAI Agents, Google ADK, and CrewAI
- Core event types: SESSION, AGENT, MODEL_INVOCATION, TOOL_CALL, MCP_CALL, INPUT/OUTPUT, ERROR, POLICY_DECISION, DATA_ACCESS, STATE_UPDATE, SYSTEM
- Distributed tracing with session/trace/span hierarchy
- UUID7 event IDs for time-ordered tracking
- File and API output modes
- Adaptive batching for optimal performance
- Proto-compliant event schema