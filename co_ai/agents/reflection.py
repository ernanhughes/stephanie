# co_ai/agents/reflection.py
from co_ai.agents.base import BaseAgent


class ReflectionAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.cfg = cfg

    async def run(self, context: dict) -> dict:
        goal = context.get("goal", "")
        if self.cfg.get("skip_if_completed", False):
            results  = self._get_completed(context)
            if results:
                self.logger.log("ReflectionAgent", {"goal": goal})
                return results

        hypotheses = context.get("hypothesis", [])
        reviews = []

        for h in hypotheses:
            self.log(f"Generating review for: {h}")
            prompt = self.prompt_loader.load_prompt(self.cfg, {**context, **{"hypothesis":h}})
            review = self.call_llm(prompt).strip()
            self.memory.store_review(h, review)
            reviews.append({"hypothesis": h, "review": review})

        context["reviews"] = reviews
        self.logger.log("GeneratedReviews", {
            "goal": goal,
            "reviews": reviews
        })

        if self.cfg.get("save_context", False):
            self._save_context(context)
        return context
