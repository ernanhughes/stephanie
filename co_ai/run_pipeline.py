# run_pipeline.py
import asyncio
import logging

import hydra
from omegaconf import DictConfig

from co_ai.logs import JSONLogger
from co_ai.memory.vector_store import VectorMemory
from co_ai.supervisor import Supervisor


@hydra.main(config_path="../configs", config_name="pipeline", version_base=None)
def run(cfg: DictConfig):
    async def main():

        supervisor = Supervisor(cfg, memory=VectorMemory(), logger=JSONLogger())
        goal = "What are the key financial risks for Tesla in Q4 2024?"
        run_id = "tesla_q4_test"
        use_grafting = True

        result = await supervisor.run_pipeline_config(goal, run_id, use_grafting)
        print("\n--- Final Summary ---")
        print(result)

    asyncio.run(main())

if __name__ == "__main__":
    # Suppress HTTPX logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    # Suppress LiteLLM logs
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    run()
