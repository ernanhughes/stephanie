# stephanie/agents/prompt_metrics_agent.py
from stephanie.services.metrics_service import MetricsService
from stephanie.agents.base_agent import BaseAgent

class MetricsAgent(BaseAgent):
    def __init__(self, cfg, memory=None, container=None, logger=None, full_cfg=None):
        super().__init__(cfg, memory, container, logger)
        self.ms = MetricsService(cfg, memory, logger, container)

    async def run(self, context):
        goal = ((context.get("goal") or {}).get("goal_text") or "").strip()
        text = (context.get("final_prompt") or context.get("prompt") or "").strip()
        if not text:
            self.logger.log("MetricsSkipped", {"reason": "no text"})
            return context
        out = await self.ms.build(goal=goal, text=text, context=context)
        context["metrics"] = out
        self.logger.log("MetricsReady", {"dim": out["meta"]["dim"]})
        return context
