from co_ai.agents.base import BaseAgent
from co_ai.constants import GOAL, GOAL_TEXT, HYPOTHESES, REFLECTION, TEXT
from co_ai.models import ScoreORM, ScoreRuleLinkORM
from co_ai.scoring import ReflectionScore


class ReflectionAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

    async def run(self, context: dict) -> dict:
        goal = context.get(GOAL)
        hypotheses = self.get_hypotheses(context)
        # Run reflection logic
        reflections = []
        reflection_scorer = ReflectionScore(self.cfg, self.memory, self.logger)
        for hyp in hypotheses:
            self.logger.log("ReflectingOnHypothesis", {HYPOTHESES: hyp})

            hyp_text = hyp.get(TEXT)
            prompt = self.prompt_loader.load_prompt(self.cfg, {
                **context,
                **{HYPOTHESES: hyp_text, GOAL:goal.get(GOAL_TEXT)}
            })

            hyp_id = self.get_hypothesis_id(hyp)
            reflection = self.call_llm(prompt, context).strip()
            self.memory.hypotheses.update_reflection(hyp_id, reflection)
            hyp[REFLECTION] = reflection

            # Compute structured score
            reflection_score = reflection_scorer.get_score(hyp, context)

            score_obj = ScoreORM(
                goal_id=self.get_goal_id(context.get(GOAL)),
                hypothesis_id=hyp_id,
                agent_name=self.name,
                model_name=self.model_name,
                evaluator_name="ReflectionAgent",
                score_type="self_reward",
                score=reflection_score,
                extra_data=self.cfg  # Optional: track adapter or config
            )

            score_id = self.memory.scores.insert(score_obj)

            # Optional: if rules applied, link score to rule applications
            rule_apps = self.memory.rule_applications.get_for_goal_and_hypothesis(
                goal_id=score_obj.goal_id,
                hypothesis_id=hyp_id,
            )

            for ra in rule_apps:
                link = ScoreRuleLinkORM(score_id=score_id, rule_application_id=ra.id)
                self.memory.session.add(link)

            self.memory.session.commit()

            reflections.append({
                "reflection": reflection,
                "score": reflection_score,
                "score_id": score_id
            })

            self.logger.log(
                "ReflectionScoreStored",
                {
                    "hypothesis": hyp_text[:60],
                    "score": score_obj.to_dict(),
                },
            )
        context[self.output_key] = reflections
        return context
