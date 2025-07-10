# worldview_pipeline_runner.py

from datetime import datetime

from stephanie.agents.base_agent import BaseAgent
from stephanie.registry.pipeline import PipelineRegistry
from stephanie.utils.pipeline_runner import PipelineRunner


class WorldviewPipelineRunner(BaseAgent):
    """
    Runs pipelines within a given worldview context.
    Orchestrates execution based on worldview goals and configuration.
    """
    def __init__(self, cfg, memory=None, logger=None, full_cfg=None):
        super().__init__(cfg, memory, logger)
        self.full_cfg = full_cfg
        self.pipeline_registry_path = cfg.get("pipeline_registry_path", "config/registry/pipeline.yaml")
        self.pipeline_registry = PipelineRegistry(self.pipeline_registry_path)
        self.runner = PipelineRunner(full_cfg, memory=memory, logger=logger)
                

    async def run(self, context: dict) -> dict:
        """
        Runs a pipeline associated with a specific goal from the worldview.

        Args:
            pipeline_name (str): Optionally override the default pipeline for the goal
            input_overrides (dict): Optional overrides for runtime input
        """
        goal = context.get("goal")
        goal_id = goal.get("id") if goal else None
        database_path = context.get("worldview_path")
        pipelines = context.get("pipelines")

        for pipeline_key in pipelines:
            pipeline_def = self.pipeline_registry.get_pipeline(pipeline_key)
            if not pipeline_def:
                self.logger.log("PipelineNotFound", {
                    "goal_id": goal_id,
                    "pipeline": pipeline_key,
                    "timestamp": datetime.utcnow().isoformat(),
                })  

            inputs = context.get("inputs", {})

            self.logger.log("PipelineExecutionStarted", {
                "goal_id": goal_id,
                "pipeline": pipeline_key,
                "timestamp": datetime.utcnow().isoformat(),
                "inputs": inputs,
            })

            result = await self.runner.run(
                pipeline_def=pipeline_def,
                context=context,
                tag=context.get("tag", "runtime")
            )

            self.logger.log("PipelineExecutionFinished", {
                "goal_id": goal_id,
                "pipeline": pipeline_key,
                "timestamp": datetime.utcnow().isoformat(),
                "output_summary": str(result)[:300],
            })

        return result
