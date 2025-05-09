# co_ai/agents/generation.py
from co_ai.agents.base import BaseAgent


class GenerationAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

    async def run(self, input_data: dict) -> dict:
        goal = input_data.get("goal")
        self.log(f"Generating hypotheses for: {goal}")

        prompt = f"Generate three hypotheses related to: {goal}"
        raw_output = self.call_llm(prompt)
        hypotheses = self.extract_list_items(raw_output)

        self.log(f"Parsed {len(hypotheses)} hypotheses.")
        return {"hypotheses": hypotheses}
