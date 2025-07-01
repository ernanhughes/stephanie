# main.py
import asyncio
import json
import logging
import os
from datetime import datetime

import hydra
import yaml
from omegaconf import DictConfig, OmegaConf

from stephanie.logs import JSONLogger
from stephanie.memory import MemoryTool
from stephanie.scoring.score_bundle import ScoreBundle
from stephanie.supervisor import Supervisor
from stephanie.utils import generate_run_id, get_log_file_path


@hydra.main(config_path="../config", config_name="pipelines/cot", version_base=None)
def run(cfg: DictConfig):
    async def main():
        save_config_to_timestamped_file(cfg=cfg)
                                        
        # Setup logger and memory
        run_id = generate_run_id(cfg.goal.goal_text if "goal" in cfg else "batch")
        log_path = get_log_file_path(run_id, cfg)
        logger = JSONLogger(log_path=log_path)
        memory = MemoryTool(cfg=cfg.db, logger=logger)

        supervisor = Supervisor(cfg=cfg, memory=memory, logger=logger)

        # ✅ Batch Mode: input_file provided
        if "input_file" in cfg and cfg.input_file:
            print(f"📂 Batch mode: Loading from file: {cfg.input_file}")
            result = await supervisor.run_pipeline_config({"input_file": cfg.input_file})
            print(f"✅ Batch run completed for file: {cfg.input_file}: {str(result)[:100]}")
            return

        # ✅ Single goal mode
        print(f"🟢 Running pipeline with run_id={run_id}")
        print(f"🧠 Goal: {cfg.goal}")
        print(f"📁 Config source: {str(cfg)[:100]}...")

        goal = OmegaConf.to_container(cfg.goal, resolve=True)
        context = {
            "goal": goal,
            "run_id": run_id,
        }

        result = await supervisor.run_pipeline_config(context)

        save_json_result(log_path, result)

        if cfg.report.generate_report:
            supervisor.generate_report(result, run_id=run_id)

    asyncio.run(main())


def save_yaml_result(log_path: str, result: dict):
    report_path = log_path.replace(".jsonl", ".yaml")
    with open(report_path, "w", encoding="utf-8") as f:
        yaml.dump(result, f, allow_unicode=True, sort_keys=False)
    print(f"✅ Result saved to: {report_path}")

def default_serializer(obj):
    if isinstance(obj, ScoreBundle):
        return obj.to_dict()
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, set):
        return list(obj)
    raise TypeError(f"Type {type(obj)} not serializable")

def save_json_result(log_path: str, result: dict):
    report_path = log_path.replace(".jsonl", "_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=default_serializer)
    print(f"✅ JSON result saved to: {report_path}")

def save_config_to_timestamped_file(log_path="logs", cfg: DictConfig=None):
        """
        Saves the current Hydra config to a timestamped YAML file.
        """
        os.makedirs(log_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_path = os.path.join(log_path, f"config_{timestamp}.yaml")
        with open(config_path, "w") as f:
            f.write(OmegaConf.to_yaml(cfg))
        print(f"🔧 Saved config to {config_path}")

if __name__ == "__main__":
    # Suppress HTTPX logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    # Suppress LiteLLM logs
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    run()
