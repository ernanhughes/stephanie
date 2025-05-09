# co_ai/agents/meta_review.py
from dspy import Predict, Signature, InputField, OutputField
from dspy import LM

from co_ai.agents.base import BaseAgent

class MetaReviewSignature(Signature):
    evolved_hypotheses = InputField()  # passed as a joined string
    summary = OutputField()

class MetaReviewAgent(BaseAgent):
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
        self.reviewer = Predict(MetaReviewSignature, lm=lm)

    async def run(self, input_data: dict) -> dict:
        evolved = input_data.get("evolved", [])
        hypothesis_texts = [h.text if hasattr(h, 'text') else h for h in evolved]

        self.log("Summarizing results via MetaReview DSPy module...")
        combined = "\n".join(f"- {h}" for h in hypothesis_texts)
        result = self.reviewer(evolved_hypotheses=combined)

        self.memory.log_summary(result.summary)
        return {"summary": result.summary}
