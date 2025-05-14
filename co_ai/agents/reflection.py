# co_ai/agents/reflection.py
from co_ai.agents.base import BaseAgent
from co_ai.constants import GOAL, SKIP_IF_COMPLETED


class ReflectionAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.cfg = cfg

    async def run(self, context: dict) -> dict:
        goal = context.get(GOAL, "")
        if self.cfg.get(SKIP_IF_COMPLETED, False):
            results  = self._get_completed(context)
            if results:
                self.logger.log("ReflectionAgent", {GOAL: goal})
                return results

        hypotheses = context.get("hypotheses", [])
        reviews = []

        for h in hypotheses:
            self.log(f"Generating reflection for: {h}")
            prompt = self.prompt_loader.load_prompt(self.cfg, {**context, **{"hypotheses":h}})
            review = self.call_llm(prompt).strip()
            self.memory.hypotheses.store_review(h, review)
            reviews.append({"hypotheses": h, "review": review})

        context["reviews"] = reviews
        self.logger.log("GeneratedReviews", {
            "goal": goal,
            "reviews": reviews
        })

        return context
