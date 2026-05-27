"""
DO NOT PUT YOUR API KEY ANYWHERE IN THIS CODE

In the Terminal, run the following command to pip install the anthropic Python library:
pip install -U anthropic

Run this command to set the API key as an environment variable:
YOU MUST REPLACE the value (code on right of equal sign) with your actual API key
export ANTHROPIC_API_KEY="sk-ant-..."

"""

import anthropic

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

# ── Step 1: Create the Agent ──────────────────────────────────────────────────
# Do this ONCE. Save the returned agent.id for reuse across sessions.
agent = client.beta.agents.create(
    name="coding-assistant",
    model="claude-opus-4-7",
    system="""You are a senior Python engineer.
When given a task, write clean, well-documented code,
execute it to verify it works, then report the result.""",
    tools=[
        {"type": "agent_toolset_20260401"}  # bash, file I/O, web search, etc.
    ],
)
print(f"Agent created: {agent.id}  (version {agent.version})")

# ── Step 2: Create the Environment ───────────────────────────────────────────
# Also do this once. Defines the sandbox container the agent runs inside.
environment = client.beta.environments.create(
    name="python-dev",
    config={
        "type": "cloud",
        "networking": {"type": "unrestricted"},
        "packages": {
            "pip": ["requests", "pandas", "numpy"]  # pre-installed in container
        }
    }
)
print(f"Environment created: {environment.id}")

# ── Step 3: Start a Session ───────────────────────────────────────────────────
# A session ties a specific agent version to an environment for one task.
session = client.beta.sessions.create(
    agent={"type": "agent", "id": agent.id, "version": agent.version},
    environment_id=environment.id,
    title="Fibonacci task",
)
print(f"Session started: {session.id}")

# ── Step 4: Send a Message and Stream the Response ───────────────────────────
with client.beta.sessions.events.stream(session.id) as stream:
    # Send the task AFTER opening the stream
    client.beta.sessions.events.send(
        session.id,
        events=[{
            "type": "user.message",
            "content": [{
                "type": "text",
                "text": "Write a Python script that generates the first 20 "
                        "Fibonacci numbers and saves them to fibonacci.txt"
            }]
        }]
    )

    # Process events as they stream in
    for event in stream:
        match event.type:
            case "agent.message":
                for block in event.content:
                    print(block.text, end="", flush=True)
            case "agent.tool_use":
                print(f"\n[Tool: {event.name}]")
            case "session.status_idle":
                print("\n\n✅ Agent finished.")
                break
            case "session.status_error":
                print(f"\n❌ Error: {event}")
                break