# co_ai/run_pipeline.py
import asyncio
import logging
import sys

import dspy
import yaml

from co_ai.logs import JSONLogger
from co_ai.supervisor import Supervisor


class BaseRunner:
    def __init__(self, config_path="configs/pipeline.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.supervisor = Supervisor()
        self.logger = JSONLogger()
        self.configure_lm()

    def configure_lm(self):
        model_config = self.config.get("model", {})
        model_name = model_config.get("name", "ollama_chat/qwen2.5")
        model_base = model_config.get("api_base", "http://localhost:11434")
        api_key = model_config.get("api_key", "")

        lm = dspy.LM(model_name, api_base=model_base, api_key=api_key)
        dspy.configure(lm=lm)
        logging.info(f"[LM] Configured model: {model_name} at {model_base}")
        self.logger.log({"event": "lm_configured", "model": model_name, "base": model_base})

class PipelineRunner(BaseRunner):
    async def run(self):
        goal = self.config["pipeline"].get("goal")
        run_id = self.config["pipeline"].get("run_id", "default_run")
        use_grafting = self.config["pipeline"].get("use_grafting", False)

        logging.info(f"[Pipeline] Running single goal: {goal}")
        self.logger.log({"event": "pipeline_run_start", "goal": goal, "run_id": run_id})
        await self.supervisor.run_pipeline_config(goal, run_id, use_grafting)
        self.logger.log({"event": "pipeline_run_complete", "run_id": run_id})

class BatchPipelineRunner(BaseRunner):
    def __init__(self, config_path="configs/pipeline.yaml", batch_path=None):
        super().__init__(config_path)
        self.batch_path = batch_path or "configs/hypotheses.txt"

    async def run_batch(self):
        with open(self.batch_path, "r") as f:
            goals = [line.strip() for line in f if line.strip()]

        for i, goal in enumerate(goals):
            run_id = f"batch_run_{i+1}"
            logging.info(f"[Batch] Running {i+1}/{len(goals)}: {goal}")
            self.logger.log({"event": "batch_item_start", "index": i+1, "goal": goal, "run_id": run_id})
            await self.supervisor.run_pipeline_config(goal, run_id)
            self.logger.log({"event": "batch_item_complete", "run_id": run_id})

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if len(sys.argv) > 1 and sys.argv[1] == "batch":
        runner = BatchPipelineRunner(config_path="configs/pipeline.yaml", batch_path="configs/tesla.txt")
        asyncio.run(runner.run_batch())
    else:
        runner = PipelineRunner()
        asyncio.run(runner.run())
