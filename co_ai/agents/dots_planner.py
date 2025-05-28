from co_ai.agents.base import BaseAgent
from co_ai.utils.goal_classifier import classify_goal_strategy  # See below
from co_ai.constants import STRATEGY, GOAL

class DOTSPlannerAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.strategy_map = cfg.get("strategy_routes", {})
        self.default_strategy = cfg.get("default_strategy", "default")

    async def run(self, context: dict) -> dict:
        goal = context.get(GOAL)
        strategy = classify_goal_strategy(goal)

        pipeline = self.strategy_map.get(strategy, self.strategy_map[self.default_strategy])

        context["strategy"] = strategy
        context["suggested_pipeline"] = pipeline

        self.logger.log("DOTSPlanGenerated", {
            "strategy": strategy,
            "pipeline": pipeline
        })

        return context
