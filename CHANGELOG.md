# Changelog

All notable changes to the Chaukas SDK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **RETRY Event Support for CrewAI Integration** - Achieved 100% chaukas-spec event coverage
  - Implemented automatic retry detection for LLM calls, tool executions, and task failures
  - Added intelligent retry tracking with attempt counters per agent/model/tool
  - Supports multiple backoff strategies: exponential, linear, and immediate
  - Automatically detects retryable errors based on error patterns (rate limits, timeouts, network errors)
  - Retry counters reset on successful completion to handle multiple failure scenarios
  - Created comprehensive test suite for retry event validation
  - Added two example scripts demonstrating retry event capture

### Enhanced
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