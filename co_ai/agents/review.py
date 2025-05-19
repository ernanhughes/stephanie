# co_ai/agents/review.py
from co_ai.agents.base import BaseAgent
from co_ai.constants import HYPOTHESES, REVIEW, GOAL

class ReviewAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.source = cfg.get("source", "context")
        self.batch_size = cfg.get("batch_size", 10)
        self.auto_run = cfg.get("auto_run", False)

    async def run(self, context: dict) -> dict:
        hypotheses = self.get_hypotheses(context)
        reviews = []
        for h in hypotheses:
            prompt = self.prompt_loader.load_prompt(self.cfg, {**context, **{HYPOTHESES:h}})
            review = self.call_llm(prompt, context)
            self.memory.hypotheses.store_review(h, review)
            reviews.append({HYPOTHESES: h, REVIEW: review})
        context[self.output_key] = reviews
        return context

    def get_hypotheses_from_db(self, goal: str):
        return self.memory.hypotheses.get_unreviewed(goal=goal, limit=self.batch_size)
