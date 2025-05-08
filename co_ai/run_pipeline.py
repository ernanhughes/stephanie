# co_ai/run_pipeline.py
import dspy
import yaml

from co_ai.supervisor import Supervisor


class PipelineRunner:
    def __init__(self, config_path="configs/pipeline.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.supervisor = Supervisor()

    def run(self):
        goal = self.config["pipeline"].get("goal")
        run_id = self.config["pipeline"].get("run_id", "default_run")
        use_grafting = self.config["pipeline"].get("use_grafting", False)

        lm = dspy.LM('ollama_chat/qwen2.5', api_base='http://localhost:11434', api_key='')
        dspy.configure(lm=lm)

        self.supervisor.run_pipeline_config(goal, run_id, use_grafting)

if __name__ == "__main__":
    runner = PipelineRunner()
    runner.run()
