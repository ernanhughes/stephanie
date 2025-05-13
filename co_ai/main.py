# main.py
import asyncio
import logging
import uuid
from datetime import datetime, timezone
import re
import os

import hydra
from omegaconf import DictConfig, OmegaConf

from co_ai.logs.json_logger import JSONLogger
from co_ai.memory.vector_store import VectorMemory
from co_ai.supervisor import Supervisor


@hydra.main(version_base=None, config_path="../config", config_name="config")
def run(cfg: DictConfig):
    async def main():
        print(f"Initial Config:\n{OmegaConf.to_yaml(cfg)}")

        # run_id = str(uuid.uuid4())
        run_id = "Test Run"
        log_path = get_log_file_path(run_id, cfg)
        logger = JSONLogger(log_path=log_path)
        memory = VectorMemory(cfg=cfg.db, logger=logger)

        supervisor = Supervisor(cfg=cfg, memory=memory, logger=logger)

        result = await supervisor.run_pipeline_config(
            {"goal":"The USA is on the verge of defaulting on its debt", "run_id":run_id}
        )

        print("Pipeline Result:", result)
        if (cfg.report.generate_report):
            supervisor.generate_report(result, run_id=run_id)
    asyncio.run(main())

def get_log_file_path(run_id:str, cfg: DictConfig) -> str:
    # Get the path to the log file
    if cfg.logging.logger.get("log_file", None):
        print(f"Log file path: {cfg.logging.logger.log_file}")
        return cfg.logging.logger.log_file
    
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    safe_run_id = re.sub(r"[\\W_]+", "_", run_id)  # remove/replace unsafe chars
    log_filename = f"{safe_run_id}_{timestamp}.jsonl"
    os.makedirs(cfg.logging.logger.log_path, exist_ok=True)
    log_file_path = os.path.join(cfg.logging.logger.log_path, log_filename)
    print(f"Log file path: {log_file_path}")
    return log_file_path



if __name__ == "__main__":
        # Suppress HTTPX logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    # Suppress LiteLLM logs
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    run()
