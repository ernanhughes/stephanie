# stephanie/main.py
from __future__ import annotations

import asyncio
import logging
import os

from stephanie.services.bus.zmq_broker import ZmqBrokerGuard

os.environ.setdefault("MPLBACKEND", "Agg")

import hydra
from omegaconf import DictConfig, OmegaConf

from stephanie.core.logging import JSONLogger
from stephanie.memory.memory_tool import MemoryTool
from stephanie.supervisor import Supervisor
from stephanie.utils.file_utils import (save_context_result,
                                        save_to_timestamped_file)
from stephanie.utils.run_utils import generate_run_id, get_log_file_path

logger = logging.getLogger(__name__)


@hydra.main(config_path="../config", config_name="config", version_base=None)
def run(cfg: DictConfig):
    async def main():
        save_to_timestamped_file(data=OmegaConf.to_yaml(cfg), file_prefix="used_config", file_extension="yaml", output_dir="logs")

        # Setup logger and memory
        run_id = generate_run_id(cfg.goal.goal_text if "goal" in cfg else "batch")
        log_path = get_log_file_path(run_id, cfg)
        log = JSONLogger(log_path=log_path)
        await ZmqBrokerGuard.ensure_started()  # detached=True by default
        memory = MemoryTool(cfg=cfg, logger=log)

        # Create supervisor
        supervisor = Supervisor(cfg=cfg, memory=memory, logger=log)

        # ✅ Batch Mode: input_file provided
        if "input_file" in cfg and cfg.input_file:
            logger.info(f"📂 Batch mode: Loading from file: {cfg.input_file}")
            result = await supervisor.run_pipeline_config(
                    {"input_file": cfg.input_file}
                )
            logger.info(
                    f"✅ Batch run completed for file: {cfg.input_file}: {str(result)[:100]}"
                )
                
            return

        # ✅ Single goal mode
        logger.info(f"🟢 Running pipeline with run_id={run_id}")
        logger.info(f"🧠 Goal: {cfg.goal}")
        logger.info(f"📁 Config source: {str(cfg)[:100]}...")

        goal = OmegaConf.to_container(cfg.goal, resolve=True)
        context = {
            "goal": goal,
            "run_id": run_id,
        }

        result = await supervisor.run_pipeline_config(context)
        if cfg.report.get("save_context_result", False):
            save_context_result(log_path, result)

        if cfg.report.generate_report:
            supervisor.generate_report(result)

    asyncio.run(main())



if __name__ == "__main__":
    # Suppress HTTPX logs
    logging.getLogger().addFilter(lambda record: len(record.getMessage().strip()) > 10)
    for name in ("numba", "httpcore", "httpcore.http11", "httpx", "LiteLLM", "transformers", "zeromodel", "zeromodel.config", "hnswlib", "matplotlib", "urllib3", "asyncio","PIL", "pdfminer", "sentence_transformers"):
        logging.getLogger(name).setLevel(logging.CRITICAL)
        logging.getLogger(name).propagate = False
    run()