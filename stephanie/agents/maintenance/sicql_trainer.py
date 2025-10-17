# stephanie/agents/maintenance/sicql_trainer.py
from __future__ import annotations

from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.scorable import ScorableType
from stephanie.scoring.training.preference_pair_builder import \
    PreferencePairBuilder
from stephanie.scoring.training.sicql_trainer import SICQLTrainer


class SICQLTrainerAgent(BaseAgent):
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.pair_builder = PreferencePairBuilder(memory, logger)
        self.target_type = cfg.get("target_type", ScorableType.CONVERSATION_TURN)
        self.limit = cfg.get("limit", 1000)
        self.trainer = SICQLTrainer(cfg, memory, container, logger)

    async def run(self, context: dict) -> dict:
        results = {}
        for dimension in self.trainer.dimensions:
            pairs_by_dim = self.pair_builder.get_training_pairs_by_dimension(
                dimension=dimension,
            )
            samples = pairs_by_dim.get(dimension, [])
            if not samples:
                self.logger.log("NoSamplesFound", {"dimension": dimension})
                continue
            stats = self.trainer.train(samples, dimension)
            if "error" not in stats:
                results[dimension] = stats

        context["training_stats"] = results
        return context
