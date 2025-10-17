# stephanie/agents/maintenance/svm_trainer.py
from __future__ import annotations

from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.training.preference_pair_builder import \
    PreferencePairBuilder
from stephanie.scoring.training.svm_trainer import SVMTrainer


class SVMTrainerAgent(BaseAgent):
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.trainer = SVMTrainer(cfg, memory, container, logger)
        self.dimensions = cfg.get("dimensions", [])

    async def run(self, context: dict) -> dict:
        builder = PreferencePairBuilder(self.memory, logger=self.logger)
        results = {}

        for dimension in self.dimensions:
            pairs = builder.get_training_pairs_by_dimension(
                dimension=dimension,
            )
            examples = pairs.get(dimension, [])

            if not examples:
                self.logger.log("SVMNoTrainingPairs", {"dimension": dimension})
                continue

            self.logger.log("SVMTrainerInvoked", {"dimension": dimension, "count": len(examples)})
            stats = self.trainer.train(examples, dimension=dimension)

            if "error" in stats:
                self.logger.log("SVMTrainingError", {"dimension": dimension, **stats})
            else:
                self.logger.log("SVMTrainingCompleted", stats)

            results[dimension] = stats

        context[self.output_key] = {
            "training_stats": results
        }
        return context
