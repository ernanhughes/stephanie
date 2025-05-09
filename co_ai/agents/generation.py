# co_ai/agents/generation.py
from dspy import Predict
from dspy import Signature, InputField, OutputField

from co_ai.agents.base import BaseAgent

class GenerateSignature(Signature):
    goal = InputField()
    hypotheses = OutputField()  # Let DSPy fall back to raw string parsing

class GenerationAgent(BaseAgent):
    def __init__(self, memory=None, logger=None):
        super().__init__(memory, logger)
        self.generator = Predict(GenerateSignature, lm=self.lm)

    async def run(self, input_data: dict) -> dict:
        goal = input_data.get("goal")
        self.log(f"Generating hypotheses for: {goal}")
        result = self.generator(goal=goal)
        raw_output = result.hypotheses
        hypotheses = self.extract_list_items(raw_output)

        self.log(f"Parsed {len(hypotheses)} hypotheses.")
        return {"hypotheses": hypotheses}
