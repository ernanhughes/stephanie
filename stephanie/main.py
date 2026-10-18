# stephanie/main.py
from __future__ import annotations

import asyncio
import logging
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import hydra
from omegaconf import DictConfig, OmegaConf

from stephanie.core.logging import JSONLogger
from stephanie.memory.memory_tool import MemoryTool
from stephanie.supervisor import Supervisor
from stephanie.utils.file_utils import (save_context_result,
                                        save_to_timestamped_file)
from stephanie.utils.run_utils import generate_run_id, get_log_file_path
from stephanie.utils.secret_utils import redact_secrets

logger = logging.getLogger(__name__)


@hydra.main(config_path="../config", config_name="config", version_base=None)
def run(cfg: DictConfig):
    async def main():
        memory = None
        supervisor = None
        try:
            safe_cfg = OmegaConf.create(redact_secrets(cfg))
            save_to_timestamped_file(data=OmegaConf.to_yaml(safe_cfg), file_prefix="used_config", file_extension="yaml", output_dir="logs")

            # Setup logger and configured memory.
            log_path = get_log_file_path(cfg)
            log = JSONLogger(log_path=log_path)
            memory = MemoryTool(cfg=cfg, logger=log)

            # Create supervisor
            supervisor = Supervisor(cfg=cfg, memory=memory, logger=log)

            # Batch Mode: input_file provided
            if "input_file" in cfg and cfg.input_file:
                logger.info(f"📂 Batch mode: Loading from file: {cfg.input_file}")
                result = await supervisor.run_pipeline_config(
                    {"input_file": cfg.input_file}
                )
                logger.info(
                    f"✅ Batch run completed for file: {cfg.input_file}: {str(result)[:100]}"
                )
                return

            # Single goal mode
            run_id = generate_run_id(cfg.goal.goal_text if "goal" in cfg else "batch")
            logger.info(f"🟢 Running pipeline with run_id={run_id}")
            logger.info(f"🧠 Goal: {cfg.goal}")
            logger.info("📁 Config composed by Hydra")

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
        finally:
            try:
                if supervisor is not None:
                    await supervisor.container.shutdown()
            finally:
                if memory is not None:
                    memory.close()

    asyncio.run(main())


def main():
    """Console-script entry point declared by pyproject.toml."""
    run()



if __name__ == "__main__":
    # Suppress HTTPX logs
    logging.getLogger().addFilter(lambda record: len(record.getMessage().strip()) > 10)
    for name in ("numba", "httpcore", "httpcore.http11", "httpx", "LiteLLM", "transformers", "zeromodel", "zeromodel.config", "hnswlib", "matplotlib", "urllib3", "asyncio","PIL", "pdfminer", "sentence_transformers"):
        logging.getLogger(name).setLevel(logging.CRITICAL)
        logging.getLogger(name).propagate = False
    main()
