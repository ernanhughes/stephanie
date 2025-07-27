# stephanie/agents/mrq_scoring.py
from stephanie.agents.base_agent import BaseAgent
from stephanie.evaluator import MRQSelfEvaluator
from stephanie.models import EvaluationORM


class MRQScoringAgent(BaseAgent):
    def __init__(self, cfg, memory, logger):
        super().__init__(cfg, memory, logger)
        self.evaluator = MRQSelfEvaluator(memory=memory, logger=logger)
        self.score_source = cfg.get("score_source", "mrq")

    async def run(self, context: dict) -> dict:
        goal = context.get("goal")
        goal_text = goal["goal_text"]
        hypotheses = self.memory.hypotheses.get_by_goal(goal_text)
        count_scored = 0

        for hypothesis in hypotheses:
            if not hypothesis.prompt or not scorable.text:
                continue

            existing_score = self.memory.evaluations.get_by_hypothesis_id(
                hypothesis.id, source=self.score_source
            )
            if existing_score:
                continue  # Skip if already scored by MR.Q

            # Run evaluator
            result = self.evaluator.score_single(
                prompt=hypothesis.prompt.prompt_text, output=scorable.text
            )

            # Handle result: could be float or dict of dimensions
            if isinstance(result, dict):
                score_value = result.get("overall", 0.0)
                dimensions = {k: v for k, v in result.items() if k != "overall"}
            else:
                score_value = result
                dimensions = {}

            rationale = (
                f"MRQSelfEvaluator assigned a score of {score_value:.4f} "
                f"based on hypothesis embedding alignment."
            )

            score_obj = EvaluationORM(
                goal_id=hypothesis.goal_id,
                target_type="hypothesis",
                target_id=hypothesis.id,
                agent_name=self.name,
                model_name=self.model_name,
                embedding_type=self.memory.embedding.type,
                evaluator_name=self.evaluator.__class__.__name__,
                scores=result,
                pipeline_run_id=context.get("pipeline_run_id"),
                extra_data=self.cfg,
                dimensions=dimensions,
            )

            self.memory.evaluations.insert(score_obj)
            count_scored += 1

        self.logger.log(
            "MRQScoringComplete",
            {
                "goal": goal,
                "scored_count": count_scored,
                "total_hypotheses": len(hypotheses),
            },
        )
        return context
