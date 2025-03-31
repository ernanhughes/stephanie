# test_pipeline.py
from run_pipeline import PipelineRunner

if __name__ == "__main__":
    runner = PipelineRunner(config_path="configs/pipeline.yaml")
    runner.run()
