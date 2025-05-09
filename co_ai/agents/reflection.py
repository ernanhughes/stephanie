# co_ai/agents/reflection.py
from dspy import Predict, Signature, InputField, OutputField
from dspy import LM
from co_ai.agents.base import BaseAgent

class ReflectionSignature(Signature):
    hypothesis = InputField()
    review = OutputField()

class ReflectionAgent(BaseAgent):
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
        self.reflector = Predict(ReflectionSignature, lm=lm)

    async def run(self, input_data: dict) -> dict:
        hypotheses = input_data.get("hypotheses", [])
        self.log(f"Reflecting on {len(hypotheses)} hypotheses...")
        reviews = []
        for h in hypotheses:
            result = self.reflector(hypothesis=h)
            reviews.append({
                "hypothesis": h,
                "review": result.review,
                "persona": "Reflection"
            })
        return {"reviews": reviews}
