# worldview_pipeline_runner.py

from datetime import datetime
from stephanie.worldview.worldview import Worldview
from stephanie.registry.pipeline import PipelineRegistry
from stephanie.agents.logger_mixin import LoggingMixin


class WorldviewPipelineRunner(LoggingMixin):
    """
    Runs pipelines within a given worldview context.
    Orchestrates execution based on worldview goals and configuration.
    """

    def __init__(self, worldview: Worldview, logger=None):
        self.worldview = worldview
        self.logger = logger or self._init_logger()

    def run(self, goal_id: str, pipeline_name: str = None, input_overrides: dict = None):
        """
        Runs a pipeline associated with a specific goal from the worldview.

        Args:
            goal_id (str): The ID of the goal to pursue
            pipeline_name (str): Optionally override the default pipeline for the goal
            input_overrides (dict): Optional overrides for runtime input
        """
        goal = self.worldview.get_goal(goal_id)
        if not goal:
            raise ValueError(f"Goal {goal_id} not found in worldview")

        pipeline_key = pipeline_name or goal.get("default_pipeline")
        if not pipeline_key:
            raise ValueError(f"No pipeline specified for goal {goal_id}")

        pipeline = get_pipeline_by_name(pipeline_key)
        if not pipeline:
            raise ValueError(f"Pipeline '{pipeline_key}' not found")

        inputs = goal.get("input") or {}
        if input_overrides:
            inputs.update(input_overrides)

        self.logger.log("PipelineExecutionStarted", {
            "goal_id": goal_id,
            "pipeline": pipeline_key,
            "timestamp": datetime.utcnow().isoformat(),
            "inputs": inputs,
        })

        # Run the pipeline in the context of worldview beliefs/tools
        result = pipeline.run(
            inputs=inputs,
            context={
                "beliefs": self.worldview.get_belief_context(),
                "tools": self.worldview.get_enabled_tools(),
                "embedding": self.worldview.get_embedding_model(),
                "memory": self.worldview.memory,
            }
        )

        self.logger.log("PipelineExecutionFinished", {
            "goal_id": goal_id,
            "pipeline": pipeline_key,
            "timestamp": datetime.utcnow().isoformat(),
            "output_summary": str(result)[:300],
        })

        return result
