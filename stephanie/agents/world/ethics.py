# stephanie/agents/world/ethics.py
from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.mixins.ethics_scoring_mixin import EthicsScoringMixin


class Ethics(EthicsScoringMixin, BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

    async def run(self, context: dict) -> dict:
        documents = context.get(self.input_key, [])
        ethics_reviews = []

        for doc in documents:
            # Score the document on ethics dimensions
            score = self.score_ethics(doc, context)
            self.logger.log(
                "EthicsScoreComputed",
                score,
            )
            ethics_reviews.append(score.to_dict())
        context[self.output_key] = ethics_reviews
        self.logger.log("EthicsReviewsGenerated", {"count": len(ethics_reviews)})

        prompt = self.prompt_loader.load_prompt(self.cfg, context)
        response = self.call_llm(prompt, context=context)
        self.logger.log(
            "PromptExecuted",
            {
                "prompt_text": prompt[:200],
                "response_snippet": response[:200],
            },
        )
        context["final_ethics_review"] = response
        return context
