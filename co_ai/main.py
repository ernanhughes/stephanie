# main.py
import hydra
from omegaconf import OmegaConf
from omegaconf import DictConfig
from co_ai.supervisor import Supervisor
from co_ai.memory.vector_store import VectorMemory
from co_ai.logs.json_logger import JSONLogger
import asyncio
import logging


@hydra.main(version_base=None, config_path="../config", config_name="config")
def run(cfg: DictConfig):
    async def main():
        print(OmegaConf.to_yaml(cfg))

        logger = JSONLogger(log_path=cfg.logging.logger.log_path)
        memory = VectorMemory(cfg=cfg.db, logger=logger)

        supervisor = Supervisor(cfg=cfg, memory=memory, logger=logger)

        result = await supervisor.run_pipeline_config(
            goal="Describe how the earth may be flat"
        )

        print("Pipeline Result:", result)
    asyncio.run(main())


if __name__ == "__main__":
        # Suppress HTTPX logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    # Suppress LiteLLM logs
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    run()
