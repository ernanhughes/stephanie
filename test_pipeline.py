# test_pipeline.py
import asyncio
from co_ai.supervisor import Supervisor
from co_ai.logs.json_logger import JSONLogger
import yaml
import dspy
from dspy import LM

def configure_dspy():
    model_config = {
        "name": "ollama_chat/qwen2.5",
        "api_base": "http://localhost:11434",
        "api_key": None,
    }
    lm = LM(
        model_config["name"],
        api_base=model_config["api_base"],
        api_key=model_config.get("api_key")
    )
    dspy.configure(lm=lm)


def load_config(config_path="configs/pipeline.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

async def run_test_pipeline():
    configure_dspy()  # 🔥 Add this first

    config = load_config()
    goal = config["pipeline"]["goal"]
    run_id = config["pipeline"].get("run_id", "test_run")
    use_grafting = config["pipeline"].get("use_grafting", False)

    supervisor = Supervisor()
    logger = JSONLogger()

    print(f"Running pipeline test: goal='{goal}', run_id='{run_id}'")
    await supervisor.run_pipeline_config(goal, run_id, use_grafting)
    print("✅ Pipeline executed successfully.")

if __name__ == "__main__":
    asyncio.run(run_test_pipeline())
