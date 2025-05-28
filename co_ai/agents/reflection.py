from co_ai.agents.base import BaseAgent
from co_ai.constants import GOAL, HYPOTHESES, REFLECTION, TEXT
from co_ai.scoring import ReflectionScore


class ReflectionAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

    async def run(self, context: dict) -> dict:
        goal = self.extract_goal_text(context.get(GOAL))
        hypotheses = self.get_hypotheses(context)
        # Run reflection logic
        reflections = []
        reflection_scorer = ReflectionScore(self.cfg, self.memory, self.logger)
        for hyp in hypotheses:
            self.logger.log("ReflectingOnHypothesis", {HYPOTHESES: hyp})

            hyp_text = hyp.get(TEXT)
            prompt = self.prompt_loader.load_prompt(self.cfg, {
                **context,
                **{HYPOTHESES: hyp_text}
            })

            hyp_id = self.get_hypothesis_id(hyp)
            reflection = self.call_llm(prompt, context).strip()
            self.memory.hypotheses.update_reflection(hyp_id, reflection)
            hyp[REFLECTION] = reflection

            reflection_score = reflection_scorer.get_score(hyp, context)

            reflections.append({"reflection":reflection, "score": reflection_score})
            self.logger.log(
                "ReflectionScoreComputed",
                {"goal": goal, "reflected_count": len(reflections)},
            )

        context[self.output_key] = reflections
        return context
