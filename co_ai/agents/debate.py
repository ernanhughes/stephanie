# co_ai/agents/debate.py
from dspy import Predict, Signature, InputField, OutputField
from dspy import LM
from co_ai.agents.base import BaseAgent
from typing import List

class DebateSignature(Signature):
    hypothesis = InputField()
    review = OutputField()

class OptimistDebaterAgent(BaseAgent):
    def __init__(self, memory=None, logger=None, model_config=None):
        super().__init__(memory=memory, logger=logger)
        self.model_config = model_config or {
            "name": "ollama_chat/mistral",
            "api_base": "http://localhost:11434",
            "api_key": None,
        }
        lm = LM(
            self.model_config["name"],
            api_base=self.model_config["api_base"],
            api_key=self.model_config.get("api_key")
        )
        self.predictor = Predict(DebateSignature, lm=lm)

    async def run(self, input_data: dict) -> dict:
        hypotheses = input_data.get("hypotheses", [])
        self.log(f"Running OptimistDebater on {len(hypotheses)} hypotheses...")
        results = []
        for h in hypotheses:
            response = self.predictor(hypothesis=h)
            results.append({
                "hypothesis": h,
                "review": response.review,
                "persona": "Optimist"
            })
        return {"reviews": results}

class SkepticDebaterAgent(BaseAgent):
    def __init__(self, memory=None, logger=None, model_config=None):
        super().__init__(memory=memory, logger=logger)
        self.model_config = model_config or {
            "name": "ollama_chat/qwen3",
            "api_base": "http://localhost:11434",
            "api_key": None,
        }
        lm = LM(
            self.model_config["name"],
            api_base=self.model_config["api_base"],
            api_key=self.model_config.get("api_key")
        )
        self.predictor = Predict(DebateSignature, lm=lm)

    async def run(self, input_data: dict) -> dict:
        hypotheses = input_data.get("hypotheses", [])
        self.log(f"Running SkepticDebater on {len(hypotheses)} hypotheses...")
        results = []
        for h in hypotheses:
            response = self.predictor(hypothesis=h)
            results.append({
                "hypothesis": h,
                "review": response.review,
                "persona": "Skeptic"
            })
        return {"reviews": results}
