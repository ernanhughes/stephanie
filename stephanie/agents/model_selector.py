# stephanie/agents/model_selector.py
from stephanie.agents.base_agent import BaseAgent
from stephanie.constants import GOAL


class ModelSelectorAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.model_rankings = {}

    async def run(self, context: dict) -> dict:
        goal = self.memory.goals.get_or_create(context.get(GOAL))
        preferences = context.get("preferences", [])

        # Use metadata to select best model
        best_model = self._select_best_model(goal, preferences)

        context["model"] = best_model
        return context

    def _select_best_model(self, goal: str, preferences: list):
        """Select best model based on historical rankings"""
        if not preferences:
            return "qwen3"  # default

        if "novelty" in preferences:
            return "mistral"
        elif "biological_plausibility" in preferences:
            return "phi3"
        elif "simplicity" in preferences:
            return "llama3.2-3b"
        else:
            return "qwen3"  # fallback
