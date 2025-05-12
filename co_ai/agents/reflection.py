# co_ai/agents/reflection.py
from co_ai.agents.base import BaseAgent


class ReflectionAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.cfg = cfg

    async def run(self, context: dict) -> dict:
        goal = context.get("goal", "")

        hypotheses = context.get("hypotheses", [])
        reviews = []

        for h in hypotheses:
            self.log(f"Generating review for: {h}")
            prompt = self.prompt_loader.load_prompt({**self.cfg, **{"hypothesis":h}}, context)
            review = self.call_llm(prompt).strip()
            self.memory.store_review(h, review)
            reviews.append({"hypothesis": h, "review": review})

        context["reviews"] = reviews
        self.logger.log("GeneratedReviews", {
            "goal": goal,
            "reviews": reviews
        })
        return context
