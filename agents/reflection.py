# ai_co_scientist/agents/reflection.py
from agents.base import BaseAgent
from memory.hypothesis_model import Hypothesis
from dspy import Signature, Module, Predict, InputField, OutputField

class ReflectSignature(Signature):
    """Critique a hypothesis for clarity, novelty, and testability."""
    hypothesis = InputField()
    review = OutputField()

class ReflectHypothesis(Module):
    def __init__(self):
        super().__init__()
        self.reviewer = Predict(ReflectSignature)

    def forward(self, hypothesis: str) -> str:
        return self.reviewer(hypothesis=hypothesis).review

class ReflectionAgent(BaseAgent):
    def __init__(self, memory):
        super().__init__(memory)
        self.reflector = ReflectHypothesis()

    async def run(self, input_data: dict) -> dict:
        hypotheses = input_data.get("hypotheses", [])
        self.log("Reviewing hypotheses via DSPy...")

        reviewed = []
        for hyp in hypotheses:
            text = hyp.text if isinstance(hyp, Hypothesis) else hyp
            review = self.reflector.forward(text)
            reviewed.append({"hypothesis": text, "review": review})
            self.memory.store_review(text, review)

        return {"reviewed": reviewed}
