# co_ai/agents/reflection.py
from co_ai.agents.base import BaseAgent


class ReflectionAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

    async def run(self, input_data: dict) -> dict:
        hypotheses = input_data.get("hypotheses", [])
        reviews = []

        for h in hypotheses:
            prompt = (
                f"Critique the following hypothesis for clarity, novelty, and testability:\n\n"
                f"{h}\n\n"
                f"Provide a short analysis addressing all three aspects."
            )
            review = self.call_llm(prompt).strip()
            self.memory.store_review(h, review)
            reviews.append({"hypothesis": h, "review": review})

        return {"reviewed": reviews}
