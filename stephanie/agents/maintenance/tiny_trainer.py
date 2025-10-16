# stephanie/agents/maintenance/tiny_trainer.py
from __future__ import annotations

from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.scorable import ScorableType
from stephanie.scoring.training.preference_pair_builder import \
    PreferencePairBuilder
from stephanie.scoring.training.tiny_recursion_trainer import TinyTrainer


class TinyTrainerAgent(BaseAgent):
    """
    Agent to train the Tiny model for multiple dimensions.
    Uses SICQL Q-values as training targets for each goal/document pair.
    """
    def __init__(self, cfg, memory, container, logger, full_cfg):
        super().__init__(cfg, memory, container, logger)
        self.dimensions = ["faithfulness"]
        self.dimensions2 = ["reasoning", "knowledge", "clarity", "faithfulness", "coverage"]
        self.pair_builder = PreferencePairBuilder(memory, logger)
        self.trainer = TinyTrainer(full_cfg.scorer.hrm, memory, container=container, logger=logger)
        self.target_type = cfg.get("target_type", ScorableType.CONVERSATION_TURN)
        self.limit = cfg.get("limit", 1000)
        self.max_documents = cfg.get("max_documents", 500)

    async def run(self, context: dict) -> dict:
        results = {}
        for dimension in self.dimensions:
            pairs_by_dim = self.pair_builder.get_training_pairs_by_dimension(
                dimension=dimension,
                target_type=self.target_type,
                limit=self.limit
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
