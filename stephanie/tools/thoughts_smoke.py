from __future__ import annotations
import asyncio
import argparse
from stephanie.core.app_context import AppContext
from stephanie.components.thoughts.agents.thought_processor_agent import ThoughtProcessorAgent

RAW = [
    {"content": "We should verify the dataset split before training.", "kind": "verify", "tags": ["data"], "evidence": []},
    {"content": "Use expectile=0.7 for SICQL stability.", "kind": "decide", "tags": ["sicql"], "evidence": []},
    {"content": "If HRM disagrees with SICQL, zoom tile #12.", "kind": "repair", "tags": ["vpm","hrm"], "evidence": []},
]

async def main(args):
    app = AppContext.load()  # Your standard loader
    agent = ThoughtProcessorAgent(app)
    out = await agent.run({
        "goal_text": args.goal,
        "run_id": args.run_id,
        "raw_thoughts": RAW,
    })
    print("Committed cube:", out.get("thought_cube_id"))
    if out.get("thought_vpm_path"):
        print("VPM tile:", out["thought_vpm_path"])

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--goal", default="Calibrate thought quality for policy training")
    p.add_argument("--run_id", default="thoughts_demo")
    args = p.parse_args()
    asyncio.run(main(args))
