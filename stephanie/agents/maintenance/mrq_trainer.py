# stephanie/agents/maintenance/mrq_trainer_agent.py

from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.scorable import ScorableFactory, ScorableType
from stephanie.scoring.training.mrq_trainer import MRQTrainer
from stephanie.scoring.training.preference_pair_builder import \
    PreferencePairBuilder


class MRQTrainerAgent(BaseAgent):
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.pair_builder = PreferencePairBuilder(memory, logger)
        self.target_type = cfg.get("target_type", "conversation_turn")
        self.limit = cfg.get("limit", 1000)
        self.trainer = MRQTrainer(cfg, memory=memory, container=container, logger=logger)


    def _extract_samples(self, context):
        goal = context.get("goal", {})
        documents = context.get("documents", [])
        samples = []
        for doc in documents:
            scorable = ScorableFactory.from_dict(doc, ScorableType.DOCUMENT)
            score = self.memory.scores.get_score(goal_id=goal["id"], scorable_id=scorable.id)
            if score:
                samples.append({
                    "title": goal.get("goal_text", ""),
                    "output": scorable.text,
                    "score": score.score
                })
        return samples

    async def run(self, context: dict) -> dict:
        """
        Agent entry point to train MRQ models for all configured dimensions.
        """
        results = {}
        for dimension in self.trainer.dimensions:
            pairs_by_dim = self.pair_builder.get_training_pairs_by_dimension(
                dim=[dimension],
                limit=self.limit,
                target_type=self.target_type, 
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
