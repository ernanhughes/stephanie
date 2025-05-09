# co_ai/agents/meta_review.py
from co_ai.agents.base import BaseAgent


class MetaReviewAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

    async def run(self, input_data: dict) -> dict:
        evolved = input_data.get("evolved", [])
        hypothesis_texts = [h.text if hasattr(h, 'text') else h for h in evolved]

        self.log("Summarizing evolved hypotheses into a meta-review...")
        prompt = (
            "You are a research lead tasked with summarizing the following evolved hypotheses.\n\n"
            "Combine them into a clear, strategic research direction:\n\n" +
            "\n".join(f"- {h}" for h in hypothesis_texts)
        )
        summary = self.call_llm(prompt).strip()

        self.memory.log_summary(summary)
        return {"summary": summary}
