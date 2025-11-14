# stephanie/main.py
from __future__ import annotations

import asyncio
import json
import logging
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import hydra
import yaml
from omegaconf import DictConfig, OmegaConf

from stephanie.core.logging import JSONLogger
from stephanie.data.plan_trace import ExecutionStep, PlanTrace
from stephanie.memory.memory_tool import MemoryTool
from stephanie.supervisor import Supervisor
from stephanie.utils.file_utils import save_to_timestamped_file
from stephanie.utils.run_utils import generate_run_id, get_log_file_path
from contextlib import suppress

logger = logging.getLogger(__name__)


@hydra.main(config_path="../config", config_name="config", version_base=None)
def run(cfg: DictConfig):
    async def main():
        save_to_timestamped_file(data=OmegaConf.to_yaml(cfg), file_prefix="used_config", file_extension="yaml", output_dir="logs")

        # Setup logger and memory
        run_id = generate_run_id(cfg.goal.goal_text if "goal" in cfg else "batch")
        log_path = get_log_file_path(run_id, cfg)
        log = JSONLogger(log_path=log_path)
        memory = MemoryTool(cfg=cfg, logger=log)

        # Create supervisor
        supervisor = Supervisor(cfg=cfg, memory=memory, logger=log)

        # ✅ Batch Mode: input_file provided
        if "input_file" in cfg and cfg.input_file:
            logger.info(f"📂 Batch mode: Loading from file: {cfg.input_file}")
            try:
                result = await supervisor.run_pipeline_config(
                        {"input_file": cfg.input_file}
                    )
                logger.info(
                        f"✅ Batch run completed for file: {cfg.input_file}: {str(result)[:100]}"
                    )
            finally:
                # orderly shutdown; ignore CancelledError noise on exit
                from stephanie.services.bus.zmq_broker import ZMQBroker
                with suppress(asyncio.CancelledError):
                    await ZMQBroker.close()
                
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

        try:
            result = await supervisor.run_pipeline_config(context)
        finally:
            # orderly shutdown; ignore CancelledError noise on exit
            from stephanie.services.bus.zmq_broker import ZMQBroker
            with suppress(asyncio.CancelledError):
                await ZMQBroker.close()

        if cfg.report.get("save_context_result", False):
            save_json_result(log_path, result)

        if cfg.report.generate_report:
            supervisor.generate_report(result)

    asyncio.run(main())


def save_yaml_result(log_path: str, result: dict):
    report_path = log_path.replace(".jsonl", ".yaml")
    with open(report_path, "w", encoding="utf-8") as f:
        yaml.dump(result, f, allow_unicode=True, sort_keys=False)
    logger.info(f"✅ Result saved to: {report_path}")



def default_serializer(obj):
    import numpy as np
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'to_dict'):
        return obj.to_dict()
    elif isinstance(obj, ExecutionStep):
        return obj.to_dict()
    # Handle PlanTrace objects
    elif isinstance(obj, PlanTrace):
        return obj.to_dict()
    # Handle DictConfig objects from Hydra
    elif hasattr(obj, '_get_node'):
        return OmegaConf.to_container(obj, resolve=True, enum_to_str=True)
    # If we still can't serialize, raise the error
    raise TypeError(f"Type {type(obj)} not serializable")

def save_json_result(log_path: str, result: dict):
    report_path = log_path.replace(".jsonl", "_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=default_serializer)
    logger.info(f"✅ JSON result saved to: {report_path}")




if __name__ == "__main__":
    # Suppress HTTPX logs
    logging.getLogger().addFilter(lambda record: len(record.getMessage().strip()) > 10)
    for name in ("numba", "httpcore", "httpcore.http11", "httpx", "LiteLLM", "transformers", "zeromodel", "hnswlib", "matplotlib", "urllib3", "asyncio","PIL", "pdfminer"):
        logging.getLogger(name).setLevel(logging.CRITICAL)
        logging.getLogger(name).propagate = False
    run()
