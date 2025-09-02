# Changelog

All notable changes to the Chaukas SDK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Enhanced OpenAI Agents Integration** - Achieved 80% event coverage (16/20 events)
  - Created `OpenAIAgentsEnhancedWrapper` using base classes and utilities
  - Added SESSION_START/END lifecycle tracking
  - Added TOOL_CALL_START tracking from LLM responses
  - Added INPUT_RECEIVED and OUTPUT_EMITTED for I/O tracking
  - Added RETRY event support with intelligent error detection
  - Session management with automatic start/end
  - Comprehensive test suite with 10+ test cases
  - Working example with retry simulation

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
  - Increased event coverage from 25% to 80% (5 events → 16 events)
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