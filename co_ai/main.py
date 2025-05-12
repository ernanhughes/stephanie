# main.py
import asyncio
import logging
import uuid

import hydra
from omegaconf import DictConfig, OmegaConf

from co_ai.logs.json_logger import JSONLogger
from co_ai.memory.vector_store import VectorMemory
from co_ai.supervisor import Supervisor


@hydra.main(version_base=None, config_path="../config", config_name="config")
def run(cfg: DictConfig):
    async def main():
        print(f"Initial Config:\n{OmegaConf.to_yaml(cfg)}")

        logger = JSONLogger(log_path=cfg.logging.logger.log_path)
        memory = VectorMemory(cfg=cfg.db, logger=logger)

        supervisor = Supervisor(cfg=cfg, memory=memory, logger=logger)

        run_id = str(uuid.uuid4())
        run_id = "Test Run"
        result = await supervisor.run_pipeline_config(
            {"goal":"The USA is on the verge of defaulting on its debt", "run_id":run_id}
        )

        print("Pipeline Result:", result)
        if (cfg.report.generate_report):
            supervisor.generate_report(result, run_id=run_id)
    asyncio.run(main())


if __name__ == "__main__":
        # Suppress HTTPX logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    # Suppress LiteLLM logs
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    run()
