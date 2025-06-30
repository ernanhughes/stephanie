# co_ai/agents/pipeline/pipeline_runner.py

import copy

from omegaconf import OmegaConf

from co_ai.agents.base_agent import BaseAgent
from co_ai.supervisor import Supervisor


class PipelineRunnerAgent(BaseAgent):
    """
    A general-purpose agent for running arbitrary Co AI pipelines from configuration.
    Accepts a pipeline definition (list of stages), goal context, and optional overrides.
    """

    def __init__(self, cfg, memory=None, logger=None, full_cfg=None):
        super().__init__(cfg, memory, logger)
        self.full_cfg = full_cfg

    async def run(self, context: dict) -> dict:
        # Step 1: Retrieve pipeline stages
        pipeline_def = context.get("pipeline_stages")
        if not pipeline_def:
            self.logger.log("PipelineRunnerMissingStages", {"context": context})
            return {"status": "error", "message": "Missing pipeline_stages in context."}

        # Step 2: Inject pipeline into config
        pipeline_cfg = self.inject_pipeline_config(pipeline_def, tag=context.get("tag", "runtime"))
        full_cfg = OmegaConf.merge(self.full_cfg, pipeline_cfg)

        # Step 3: Run pipeline via Supervisor
        supervisor = Supervisor(full_cfg, memory=self.memory, logger=self.logger)
        result = await supervisor.run_pipeline_config(context)

        # Step 4: Return result
        return {
            "status": "success",
            "result": result,
            "best_score": result.get("best_score", 0.0),
            "selected": result.get("selected"),
        }

    from omegaconf import OmegaConf

    def inject_pipeline_config(
        self, pipeline_def: list, tag: str = "runtime"
    ) -> OmegaConf:
        """
        Injects a pipeline definition into the full config structure.
        Replaces the pipeline stages and agent blocks in the config.
        """
        try:
            full_cfg_dict = OmegaConf.to_container(self.full_cfg, resolve=True)

            full_cfg_dict["pipeline"]["tag"] = tag
            full_cfg_dict["pipeline"]["stages"] = pipeline_def
            full_cfg_dict["agents"] = {stage["name"]: stage for stage in pipeline_def}

            return OmegaConf.create(full_cfg_dict)

        except Exception as e:
            if hasattr(self, "logger") and self.logger:
                self.logger.log(
                    "PipelineInjectionError",
                    {"error": str(e), "pipeline_def": pipeline_def, "tag": tag},
                )
            raise e  # Reraise the exception after logging so it can be handled upstream if needed
