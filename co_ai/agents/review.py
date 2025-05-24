# co_ai/agents/review.py
from co_ai.agents.base import BaseAgent
from co_ai.constants import HYPOTHESES, REVIEW


class ReviewAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

    async def run(self, context: dict) -> dict:
        hypotheses = self.get_hypotheses(context)
        reviews = []
        for h in hypotheses:
            prompt = self.prompt_loader.load_prompt(self.cfg, {**context, **{HYPOTHESES:h}})
            review = self.call_llm(prompt, context)
            self.memory.hypotheses.store_review(h, review)
            reviews.append(review)
        context[self.output_key] = reviews
        return context
