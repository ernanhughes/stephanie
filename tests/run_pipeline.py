# run_pipeline.py
import asyncio
import hydra
from omegaconf import DictConfig
from co_ai.agents.generation import GenerationAgent

@hydra.main(config_path="configs", config_name="pipeline", version_base=None)
def run(cfg: DictConfig):
    async def main():
        agent = GenerationAgent(cfg)
        result = await agent.run({
            "goal": "What are possible risks to Tesla's Q4 2024 performance?"
        })
        print("Generated Hypotheses:")
        for i, h in enumerate(result["hypotheses"], 1):
            print(f"{i}. {h}")
    asyncio.run(main())

if __name__ == "__main__":
    run()
