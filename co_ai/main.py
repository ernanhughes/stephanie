# main.py
import asyncio
import logging
import yaml

import hydra
from omegaconf import DictConfig, OmegaConf

from co_ai.logs import JSONLogger
from co_ai.memory import MemoryTool
from co_ai.supervisor import Supervisor
from co_ai.utils import generate_run_id, get_log_file_path


@hydra.main(version_base=None, config_path="../config", config_name="config")
def run(cfg: DictConfig):
    async def main():
        print(f"Initial Config:\n{OmegaConf.to_yaml(cfg)}")

        run_id = generate_run_id(cfg.goal)
        log_path = get_log_file_path(run_id, cfg)
        logger = JSONLogger(log_path=log_path)
        memory = MemoryTool(cfg=cfg.db, logger=logger)

        supervisor = Supervisor(cfg=cfg, memory=memory, logger=logger)

        print(f"🟢 Running pipeline with run_id={run_id}")
        print(f"🧠 Goal: {cfg.goal}")
        print(f"📁 Config source: {str(cfg)[:100]}...")  

        result = await supervisor.run_pipeline_config(
             {"goal": cfg.get("goal", ""), "run_id": run_id}
        )

        save_yaml_result(log_path, result)

        if cfg.report.generate_report:
            supervisor.generate_report(result, run_id=run_id)
    asyncio.run(main())


def save_yaml_result(log_path: str, result: dict):
    report_path = log_path.replace(".jsonl", ".yaml")
    with open(report_path, 'w', encoding='utf-8') as f:
        yaml.dump(result, f, allow_unicode=True, sort_keys=False)
    print(f"✅ Result saved to: {report_path}")


if __name__ == "__main__":
        # Suppress HTTPX logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    # Suppress LiteLLM logs
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    run()
