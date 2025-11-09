"""
Test script to verify POLICY_DECISION, STATE_UPDATE, and SYSTEM_EVENT capture.
"""
import asyncio
import json
from agents import Agent, Runner

from chaukas import sdk as chaukas

# Enable Chaukas with file output
chaukas.enable_chaukas()

async def main():
    print("Testing new event types: POLICY_DECISION, STATE_UPDATE, SYSTEM_EVENT")
    print("=" * 70)

    # Create an agent - this should trigger STATE_UPDATE and SYSTEM_EVENT
    agent = Agent(
        name="Test Agent",
        instructions="You are a helpful assistant.",
        model="gpt-4o-mini"
    )

    # Run the agent - this should capture all events
    message = "Say hello!"
    result = await Runner.run(starting_agent=agent, input=message)

    print(f"\nAgent response: {result.final_output[:100]}...")
    print("\nChecking captured events...")

    # Check the event file
    try:
        with open("openai_handoff_events.jsonl", "r") as f:
            lines = f.readlines()

        # Get last 20 events (from this run)
        recent_events = lines[-20:] if len(lines) > 20 else lines

        event_types = {}
        for line in recent_events:
            event = json.loads(line)
            event_type = event.get("type", "UNKNOWN")
            event_types[event_type] = event_types.get(event_type, 0) + 1

        print("\nEvent types captured in this run:")
        for event_type, count in sorted(event_types.items()):
            print(f"  {event_type}: {count}")

        # Check for new event types
        new_events_found = []
        if "EVENT_TYPE_SYSTEM" in event_types:
            new_events_found.append("SYSTEM_EVENT")
        if "EVENT_TYPE_STATE_UPDATE" in event_types:
            new_events_found.append("STATE_UPDATE")
        if "EVENT_TYPE_POLICY_DECISION" in event_types:
            new_events_found.append("POLICY_DECISION")

        print("\n" + "=" * 70)
        if new_events_found:
            print(f"✅ Successfully captured new event types: {', '.join(new_events_found)}")
        else:
            print("⚠️  No new event types found (POLICY_DECISION may require specific finish_reason)")

        print("\nNote: POLICY_DECISION events are only emitted when:")
        print("  - finish_reason is 'content_filter', 'content_policy', or 'moderation'")
        print("  - finish_reason is 'length' (token limit exceeded)")

    except FileNotFoundError:
        print("❌ Event file not found")
    except Exception as e:
        print(f"❌ Error reading events: {e}")

if __name__ == "__main__":
    asyncio.run(main())
