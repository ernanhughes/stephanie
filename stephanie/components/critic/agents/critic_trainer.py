# stephanie/components/critic/agents/critic_trainer_agent.py
from __future__ import annotations

from stephanie.agents.base_agent import BaseAgent
from stephanie.components.critic.model.critic_trainer import CriticTrainer

class CriticTrainerAgent(BaseAgent):
    """
    Thin wrapper around TinyCriticTrainer so training can be invoked
    inside Stephanie pipelines.
    """
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.trainer = CriticTrainer(cfg, memory, container, logger)

    async def run(self, context: dict) -> dict:
        result = self.trainer.train_from_dataset()
        context["tiny_critic_stats"] = {
            "cv": result.cv_summary,
            "holdout": result.holdout_summary,
            "model_path": result.model_path,
            "meta_path": result.meta_path,
        }
        return context
