#!/usr/bin/env python3
"""
Event Statistics Summarizer for Chaukas SDK

Analyzes Chaukas event JSONL files and provides statistical summaries including:
- Event type distribution
- Agent activity tracking
- Tool usage statistics
- Error and retry analysis
- Event coverage reporting

This tool can be used both as a standalone CLI tool and imported as a library.

Usage:
    # As a CLI tool
    python tools/summarize_event_stats.py output.jsonl
    python tools/summarize_event_stats.py output.jsonl --max-items 5
    python tools/summarize_event_stats.py output.jsonl --title "My Custom Analysis"

    # As a library
    from tools.summarize_event_stats import summarize_event_stats
    stats = summarize_event_stats("output.jsonl")
    print(stats['total_events'])
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def summarize_event_stats(
    filename: str,
    title: Optional[str] = None,
    max_items: int = 3,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Analyze captured events from a JSONL file and display statistical summary.

    Args:
        filename: Path to the JSONL file containing Chaukas events
        title: Custom title for the analysis report (defaults to "Event Analysis")
        max_items: Maximum number of detail items to show for each category (default: 3)
                  Use -1 to show all items
        verbose: If True, includes additional debug information

    Returns:
        Dictionary containing analysis results with keys:
        - total_events: Total number of events
        - event_counts: Dict of event type counts
        - agent_events: Dict of agent activity
        - tool_calls: Dict of tool usage
        - handoffs: List of handoff events
        - retries: List of retry events
        - errors: List of error events
        - mcp_calls: List of MCP call events
        - data_access: List of data access events
        - policy_decisions: List of policy decision events
        - state_updates: List of state update events
        - coverage_percent: Percentage of event types captured

    Raises:
        FileNotFoundError: If the specified file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    # Use default title if not provided
    if title is None:
        title = "Event Analysis"

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    try:
        with open(filename, "r") as f:
            events = [json.loads(line) for line in f if line.strip()]

        if not events:
            print("❌ No events captured")
            return {"total_events": 0}

        # Count event types
        event_counts = defaultdict(int)
        agent_events = defaultdict(list)
        tool_calls = defaultdict(int)
        handoffs = []
        retries = []
        errors = []
        mcp_calls = []
        data_access = []
        policy_decisions = []
        state_updates = []

        for event in events:
            event_type = event.get("type", "UNKNOWN")
            event_counts[event_type] += 1

            # Categorize events
            if event_type == "EVENT_TYPE_AGENT_START":
                agent_name = event.get("agent_name", "unknown")
                agent_events[agent_name].append("START")
            elif event_type == "EVENT_TYPE_AGENT_END":
                agent_name = event.get("agent_name", "unknown")
                agent_events[agent_name].append("END")
            elif event_type == "EVENT_TYPE_TOOL_CALL_START":
                if "tool_call" in event:
                    tool_name = event["tool_call"].get("tool_name", "unknown")
                    tool_calls[tool_name] += 1
            elif event_type == "EVENT_TYPE_AGENT_HANDOFF":
                handoffs.append(event)
            elif event_type == "EVENT_TYPE_RETRY":
                retries.append(event)
            elif event_type == "EVENT_TYPE_ERROR":
                errors.append(event)
            elif event_type == "EVENT_TYPE_MCP_CALL_START":
                mcp_calls.append(event)
            elif event_type == "EVENT_TYPE_DATA_ACCESS":
                data_access.append(event)
            elif event_type == "EVENT_TYPE_POLICY_DECISION":
                policy_decisions.append(event)
            elif event_type == "EVENT_TYPE_STATE_UPDATE":
                state_updates.append(event)

        # Display summary
        print(f"\n📊 Total Events Captured: {len(events)}")
        print("\n📈 Event Type Distribution:")

        # All 19 event types that Chaukas supports
        all_event_types = [
            "SESSION_START",
            "SESSION_END",
            "AGENT_START",
            "AGENT_END",
            "AGENT_HANDOFF",
            "MODEL_INVOCATION_START",
            "MODEL_INVOCATION_END",
            "TOOL_CALL_START",
            "TOOL_CALL_END",
            "MCP_CALL_START",
            "MCP_CALL_END",
            "INPUT_RECEIVED",
            "OUTPUT_EMITTED",
            "ERROR",
            "RETRY",
            "POLICY_DECISION",
            "DATA_ACCESS",
            "STATE_UPDATE",
            "SYSTEM",
        ]

        for event_type in all_event_types:
            count = event_counts.get(f"EVENT_TYPE_{event_type}", 0)
            status = "✅" if count > 0 else "⭕"
            print(f"  {status} {event_type:25} : {count:3} events")

        # Agent activity
        if agent_events:
            print(f"\n👥 Agent Activity:")
            for agent_name, activities in agent_events.items():
                starts = activities.count("START")
                ends = activities.count("END")
                status = "✅" if starts == ends else "⚠️"
                print(f"  {status} {agent_name:20} : {starts} starts, {ends} ends")

        # Tool usage
        if tool_calls:
            print(f"\n🔧 Tool Usage:")
            for tool_name, count in sorted(
                tool_calls.items(), key=lambda x: x[1], reverse=True
            ):
                print(f"  - {tool_name:20} : {count} calls")

        # Determine how many items to show
        limit = None if max_items == -1 else max_items

        # Agent handoffs
        if handoffs:
            print(f"\n🤝 Agent Handoffs: {len(handoffs)}")
            for handoff in handoffs[:limit]:
                if "agent_handoff" in handoff:
                    h = handoff["agent_handoff"]
                    print(
                        f"  - {h.get('from_agent_name', 'N/A')} → {h.get('to_agent_name', 'N/A')}"
                    )

        # MCP calls
        if mcp_calls:
            print(f"\n🔌 MCP Context Calls: {len(mcp_calls)}")
            for mcp in mcp_calls[:limit]:
                if "mcp_call" in mcp:
                    m = mcp["mcp_call"]
                    print(
                        f"  - {m.get('operation', 'N/A')} from {m.get('server_name', 'N/A')}"
                    )

        # Data access
        if data_access:
            print(f"\n📚 Data Access Events: {len(data_access)}")
            for da in data_access[:limit]:
                if "data_access" in da:
                    d = da["data_access"]
                    print(
                        f"  - {d.get('data_source', 'N/A')}: {d.get('query', 'N/A')[:50]}..."
                    )

        # Policy decisions
        if policy_decisions:
            print(f"\n⚖️ Policy Decisions: {len(policy_decisions)}")
            for pd in policy_decisions[:limit]:
                if "policy_decision" in pd:
                    p = pd["policy_decision"]
                    print(
                        f"  - {p.get('policy_name', 'N/A')}: {p.get('decision', 'N/A')}"
                    )

        # State updates
        if state_updates:
            print(f"\n💾 State Updates: {len(state_updates)}")
            for su in state_updates[:limit]:
                if "state_update" in su:
                    s = su["state_update"]
                    print(
                        f"  - {s.get('state_key', 'N/A')}: {s.get('old_value', 'N/A')} → {s.get('new_value', 'N/A')}"
                    )

        # Errors and retries
        if errors:
            print(f"\n❌ Errors: {len(errors)}")
            for error in errors[:limit]:
                if "error" in error:
                    e = error["error"]
                    print(
                        f"  - {e.get('error_code', 'N/A')}: {e.get('error_message', 'N/A')[:50]}..."
                    )

        if retries:
            print(f"\n🔄 Retries: {len(retries)}")
            for retry in retries[:limit]:
                if "retry" in retry:
                    r = retry["retry"]
                    print(
                        f"  - Attempt {r.get('attempt', 'N/A')}: {r.get('reason', 'N/A')[:50]}..."
                    )

        # Coverage calculation
        captured_types = sum(
            1 for et in all_event_types if event_counts.get(f"EVENT_TYPE_{et}", 0) > 0
        )
        coverage_percent = (captured_types / len(all_event_types)) * 100

        print(
            f"\n📋 Event Coverage: {captured_types}/{len(all_event_types)} types captured ({coverage_percent:.1f}%)"
        )

        # Return structured data
        return {
            "total_events": len(events),
            "event_counts": dict(event_counts),
            "agent_events": dict(agent_events),
            "tool_calls": dict(tool_calls),
            "handoffs": handoffs,
            "retries": retries,
            "errors": errors,
            "mcp_calls": mcp_calls,
            "data_access": data_access,
            "policy_decisions": policy_decisions,
            "state_updates": state_updates,
            "coverage_percent": coverage_percent,
        }

    except FileNotFoundError:
        print(f"❌ Output file '{filename}' not found")
        raise
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON: {e}")
        raise
    except Exception as e:
        print(f"❌ Error analyzing events: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        raise


def main():
    """CLI interface for the event statistics summarizer."""
    parser = argparse.ArgumentParser(
        description="Analyze Chaukas event JSONL files and generate statistical summaries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a specific file
  python tools/summarize_event_stats.py output.jsonl

  # Show more details per category
  python tools/summarize_event_stats.py output.jsonl --max-items 10

  # Show all items (no limit)
  python tools/summarize_event_stats.py output.jsonl --max-items -1

  # Custom title
  python tools/summarize_event_stats.py output.jsonl --title "Production Analysis"
        """,
    )

    parser.add_argument(
        "filename",
        help="Path to the JSONL file containing Chaukas events",
    )

    parser.add_argument(
        "--title",
        "-t",
        help="Custom title for the analysis report",
        default=None,
    )

    parser.add_argument(
        "--max-items",
        "-m",
        type=int,
        default=3,
        help="Maximum number of detail items to show per category (use -1 for all)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output with debug information",
    )

    args = parser.parse_args()

    # Check if file exists
    if not Path(args.filename).exists():
        print(f"Error: File '{args.filename}' not found")
        sys.exit(1)

    try:
        summarize_event_stats(
            args.filename,
            title=args.title,
            max_items=args.max_items,
            verbose=args.verbose,
        )
    except Exception:
        sys.exit(1)


if __name__ == "__main__":
    main()
