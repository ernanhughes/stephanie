# test_pipeline.py
from co_ai.run_pipeline import PipelineRunner

if __name__ == "__main__":
    runner = PipelineRunner(config_path="configs/pipeline.yaml")
    runner.run()
