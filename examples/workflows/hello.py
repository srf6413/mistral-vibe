"""Minimal demo workflow: 2 phases, 3 sequential `agent()` calls.

Used by the Integration phase's live demo (`/workflows run examples/workflows/hello.py`).
Deliberately trivial prompts so a demo run is fast and cheap.
"""

meta = {
    "name": "hello",
    "description": "Say hello, then ask two quick follow-up questions.",
    "phases": [
        {"title": "Greet", "detail": "Get a friendly hello from the agent."},
        {"title": "Follow up", "detail": "Ask two small follow-up questions."},
    ],
}


async def main(wf, args):
    name = args.get("name", "world")

    async with wf.phase("Greet"):
        greeting = await wf.agent(
            f"Reply with a short, friendly one-sentence hello to {name}.", label="greet"
        )
        wf.log(f"greeting status: {greeting.status}")

    async with wf.phase("Follow up"):
        favorite_color = await wf.agent(
            "In one short sentence, name your favorite color and why.",
            label="favorite-color",
        )
        favorite_number = await wf.agent(
            "In one short sentence, name your favorite number and why.",
            label="favorite-number",
        )
        wf.log(f"follow-up statuses: {favorite_color.status}, {favorite_number.status}")
