# test_generation.py
import asyncio
from co_ai.agents.generation import GenerationAgent

async def run():
    agent = GenerationAgent()
    result = await agent.run({
        "goal": "What are possible risks to Tesla's Q4 2024 performance?"
    })
    print("Generated Hypotheses:")
    for i, h in enumerate(result["hypotheses"], 1):
        print(f"{i}. {h}")

if __name__ == "__main__":
    asyncio.run(run())
