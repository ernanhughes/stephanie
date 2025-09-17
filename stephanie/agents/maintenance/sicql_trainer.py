# stephanie/agents/maintenance/sicql_trainer_agent.py

from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.training.preference_pair_builder import \
    PreferencePairBuilder
from stephanie.scoring.training.sicql_trainer import SICQLTrainer


class SICQLTrainerAgent(BaseAgent):
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.trainer = SICQLTrainer(cfg, memory, container, logger)
        self.dimensions = cfg.get("dimensions", [])
        self.pair_builder = PreferencePairBuilder(memory.session, logger)

    async def run(self, context: dict) -> dict:
        goal = context.get("goal", {})
        results = {}

        # Get prepared training pairs

        for dimension in self.dimensions:
            pairs_by_dim = self.pair_builder.get_training_pairs_by_dimension(
                dim=[dimension],
                goal=goal.get("goal_text"),
                limit=100
            )
            pairs = pairs_by_dim.get(dimension, [])
            samples = [
                {
                    "title": pair["title"],
                    "output": pair["output_a"],
                    "score": pair["value_a"],
                }
                for pair in pairs
            ]

            if not samples:
                self.logger.log(
                    "NoSamplesForDimension", {"dimension": dimension}
                )
                continue

            stats = self.trainer.train(samples, dimension)

            if "error" in stats:
                self.logger.log(
                    "TrainingError",
                    {"dimension": dimension, "error": stats["error"]},
                )
                continue

            self.logger.log(
                "TrainingCompleted", {"dimension": dimension, "stats": stats}
            )

            results[dimension] = stats

        context[self.output_key] = {
            "training_stats": results
        }
        return context
