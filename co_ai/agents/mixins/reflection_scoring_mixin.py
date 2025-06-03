from co_ai.scoring import ReflectionScore
from co_ai.models import ScoreORM, ScoreRuleLinkORM
from co_ai.constants import HYPOTHESES, GOAL, GOAL_TEXT, REFLECTION


class ReflectionScoringMixin:
    """
    A mixin that provides reflection and scoring functionality.
    Can be used in agents like ReflectionAgent, MetaReviewAgent, SharpeningAgent, etc.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reflection_scorer = None

    def get_reflection_scorer(self):
        """Lazy initialization of reflection scorer"""
        if not self.reflection_scorer:
            self.reflection_scorer = ReflectionScore(
                self.cfg, memory=self.memory, logger=self.logger
            )
        return self.reflection_scorer

    def reflect_on_hypothesis(self, hyp: dict, context: dict) -> str:
        """
        Generate a reflection on a hypothesis using an LLM prompt.
        If reflection already exists, skips regeneration unless forced.
        """
        hyp_text = hyp.get("text")
        hyp_id = self.get_hypothesis_id(hyp)

        if REFLECTION in hyp and hyp[REFLECTION]:
            self.logger.log(
                "ReflectionSkipped",
                {
                    "reason": "already_exists",
                    "hypothesis_id": hyp_id,
                    "reflection_snippet": hyp[REFLECTION][:100],
                },
            )
            return hyp[REFLECTION]

        prompt = self.prompt_loader.load_prompt(
            self.cfg,
            {
                **context,
                **{HYPOTHESES: hyp_text, GOAL: context.get(GOAL).get(GOAL_TEXT)},
            },
        )

        reflection = self.call_llm(prompt, context).strip()
        self.memory.hypotheses.update_reflection(hyp_id, reflection)
        hyp[REFLECTION] = reflection

        self.logger.log(
            "ReflectionGenerated",
            {"hypothesis_id": hyp_id, "reflection_snippet": reflection[:200]},
        )

        return reflection

    def score_hypothesis_with_reflection(self, hyp: dict, context: dict) -> dict:
        """
        Score a hypothesis using its reflection content.
        Also saves the score to the database with optional rule linking.
        """
        hyp_id = self.get_hypothesis_id(hyp)
        goal = context.get(GOAL)

        # Ensure reflection is available
        reflection = self.reflect_on_hypothesis(hyp, context)

        # Score it
        scorer = self.get_reflection_scorer()
        reflection_score = scorer.get_score(hyp, context)

        # Save score
        score_obj = ScoreORM(
            goal_id=self.get_goal_id(goal),
            hypothesis_id=hyp_id,
            agent_name=self.name,
            model_name=self.model_name,
            evaluator_name="ReflectionAgent",
            score_type="self_reward",
            score=reflection_score,
            extra_data=self.cfg,  # Optional: track adapter or config
        )
        score_id = self.memory.scores.insert(score_obj)

        # Link to rules if any
        rule_apps = self.memory.rule_applications.get_for_goal_and_hypothesis(
            goal_id=score_obj.goal_id,
            hypothesis_id=hyp_id,
        )

        for ra in rule_apps:
            link = ScoreRuleLinkORM(score_id=score_id, rule_application_id=ra.id)
            self.memory.session.add(link)

        self.memory.session.commit()

        hyp_text = hyp.get("text")
        self.logger.log(
            "ReflectionScoreStored",
            {
                "hypothesis": hyp_text[:60],
                "score": score_obj.to_dict(),
            },
        )

        return {
            "score": reflection_score,
            REFLECTION: reflection,
            "id": hyp_id,
            "scores": scorer.scores,
        }
