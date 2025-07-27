# stephanie/agents/maintenance/mrq_trainer_agent.py

from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.mrq.preference_pair_builder import PreferencePairBuilder
from stephanie.scoring.scorable_factory import ScorableFactory, TargetType
from stephanie.scoring.training.mrq_trainer import MRQTrainer


class MRQTrainerAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.pair_builder = PreferencePairBuilder(memory.session, logger)
        self.trainer = MRQTrainer(cfg, memory=memory, logger=logger)


    def _extract_samples(self, context):
        goal = context.get("goal", {})
        documents = context.get("documents", [])
        samples = []
        for doc in documents:
            scorable = ScorableFactory.from_dict(doc, TargetType.DOCUMENT)
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
        goal = context.get("goal", {})
        results = {}
        for dimension in self.trainer.dimensions:
            pairs_by_dim = self.pair_builder.get_training_pairs_by_dimension(
                dim=[dimension],
                goal=goal.get("goal_text"),
                limit=100
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
