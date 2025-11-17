# stephanie/agents/pipeline/pipeline_runner.py
from __future__ import annotations

from stephanie.agents.base_agent import BaseAgent
from stephanie.tools.pipeline_runner import PipelineRunner


class PipelineRunnerAgent(BaseAgent):
    def __init__(self, cfg, memory, container, logger, full_cfg=None):
        super().__init__(cfg, memory, container, logger)
        self.full_cfg = full_cfg
        self.runner = PipelineRunner(full_cfg, memory=memory, container=container, logger=logger)

    async def run(self, context: dict) -> dict:
        pipeline_def = context.get("pipeline_stages")
        if not pipeline_def:
            self.logger.log("PipelineRunnerMissingStages", {"context": context})
            return {"status": "error", "message": "Missing pipeline_stages in context."}

        result = await self.runner.run(
            pipeline_def=pipeline_def,
            context=context,
            tag=context.get("tag", "runtime"),
        )

        return result
