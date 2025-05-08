# co_ai/agents/meta_review.py
from dspy import InputField, Module, OutputField, Predict, Signature

from co_ai.agents.base import BaseAgent


class MetaReviewSignature(Signature):
    """Summarize and synthesize the evolved hypotheses into a cohesive research direction."""
    evolved_hypotheses = InputField()
    summary = OutputField()

class MetaReviewModule(Module):
    def __init__(self):
        super().__init__()
        self.summarizer = Predict(MetaReviewSignature)

    def forward(self, evolved_hypotheses: list[str]) -> str:
        combined = "\n".join(f"- {h}" for h in evolved_hypotheses)
        return self.summarizer(evolved_hypotheses=combined).summary

class MetaReviewAgent(BaseAgent):
    def __init__(self, memory):
        super().__init__(memory)
        self.reviewer = MetaReviewModule()

    async def run(self, input_data: dict) -> dict:
        evolved = input_data.get("evolved", [])
        hypothesis_texts = [h.text if hasattr(h, 'text') else h for h in evolved]

        self.log("Summarizing results via MetaReview DSPy module...")
        summary = self.reviewer.forward(hypothesis_texts)

        self.memory.log_summary(summary)
        return {"summary": summary}
