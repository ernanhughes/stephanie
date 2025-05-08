# ai_co_scientist/agents/generation.py
import asyncio
from agents.base import BaseAgent
from memory.hypothesis_model import Hypothesis
from co_ai.tools.web_search import WebSearchTool
from dspy import Signature, Module, Predict, InputField, OutputField

class GenerateSignature(Signature):
    """Generate three novel, testable hypotheses based on a research goal and its context."""
    goal = InputField()
    context = InputField()
    hypotheses = OutputField()

class GenerateHypotheses(Module):
    def __init__(self):
        super().__init__()
        self.generator = Predict(GenerateSignature)

    def forward(self, goal: str, context: str) -> str:
        return self.generator(goal=goal, context=context).hypotheses

class GenerationAgent(BaseAgent):
    def __init__(self, memory):
        super().__init__(memory)
        self.web_search = WebSearchTool()
        self.generator = GenerateHypotheses()

    async def run(self, input_data: dict) -> dict:
        goal = input_data.get("goal", "")
        self.log("Searching for related context...")
        documents = await self.web_search.search(goal)
        context = "\n".join(documents)

        self.log("Generating hypotheses via DSPy...")
        result = self.generator.forward(goal, context)

        hypotheses = []
        for hyp_text in result.splitlines():
            if not hyp_text.strip():
                continue
            hypothesis = Hypothesis(goal=goal, text=hyp_text.strip())
            self.memory.store_hypothesis(hypothesis)
            hypotheses.append(hypothesis)

        return {"hypotheses": hypotheses}
