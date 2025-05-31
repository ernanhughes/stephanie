from co_ai.agents import BaseAgent
from co_ai.evaluator import MRQSelfEvaluator
from co_ai.models import ScoreORM


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
            if not hypothesis.prompt or not hypothesis.text:
                continue
            existing_score = self.memory.scores.get_by_hypothesis_id(
                hypothesis.id, source=self.score_source
            )
            if existing_score:
                continue  # Skip if already scored by MR.Q

            score_value = self.evaluator.score_single(
                prompt=hypothesis.prompt.prompt_text, output=hypothesis.text
            )

            score_obj = ScoreORM(
                goal_id=hypothesis.goal_id,  # if available
                hypothesis_id=hypothesis.id,
                agent_name=self.name,  # 🔥 Required!
                model_name=self.model_name,  # Optional
                evaluator_name="MRQScoringAgent",  # Descriptive
                score_type=self.score_source,  # "mrq"
                score=score_value,
                rationale=(
                    f"MRQSelfEvaluator assigned a score of {score_value:.4f} "
                    f"based on the hypothesis embedding's alignment with the prompt: '{hypothesis.prompt.prompt_text[:100]}...'"
                ),
                pipeline_run_id=context.get("pipeline_run_id"),
                extra_data=self.cfg,
            )
            self.memory.scores.insert(score_obj)

        self.logger.log(
            "MRQScoringComplete",
            {
                "goal": goal,
                "scored_count": count_scored,
                "total_hypotheses": len(hypotheses),
            },
        )
        return context
