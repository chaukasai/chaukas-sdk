import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the same directory as this script
script_dir = Path(__file__).parent
load_dotenv(script_dir / ".env")

from agents import Agent, CodeInterpreterTool, Runner, trace

# Set up environment for Chaukas
os.environ["CHAUKAS_TENANT_ID"] = "demo_tenant"
os.environ["CHAUKAS_PROJECT_ID"] = "openai_tools_demo"
os.environ["CHAUKAS_OUTPUT_MODE"] = "file"
os.environ["CHAUKAS_OUTPUT_FILE"] = "openai_tools_events.jsonl"
os.environ["CHAUKAS_BATCH_SIZE"] = "1"  # Immediate write for demo

from chaukas import sdk as chaukas

chaukas.enable_chaukas()


async def main():
    agent = Agent(
        name="Code interpreter",
        # Note that using gpt-5 model with streaming for this tool requires org verification
        # Also, code interpreter tool does not support gpt-5's minimal reasoning effort
        model="gpt-4.1",
        instructions="You love doing math.",
        tools=[
            CodeInterpreterTool(
                tool_config={"type": "code_interpreter", "container": {"type": "auto"}},
            )
        ],
    )

    with trace("Code interpreter example"):
        print("Solving math problem...")
        result = Runner.run_streamed(
            agent, "What is the square root of273 * 312821 plus 1782?"
        )
        async for event in result.stream_events():
            if (
                event.type == "run_item_stream_event"
                and event.item.type == "tool_call_item"
                and event.item.raw_item.type == "code_interpreter_call"
            ):
                print(f"Code interpreter code:\n```\n{event.item.raw_item.code}\n```\n")
            elif event.type == "run_item_stream_event":
                print(f"Other event: {event.item.type}")

        print(f"Final output: {result.final_output}")


if __name__ == "__main__":
    asyncio.run(main())
