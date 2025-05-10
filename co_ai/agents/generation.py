# co_ai/agents/generation.py
import re

from co_ai.agents.base import BaseAgent


class GenerationAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

    async def run(self, input_data: dict) -> dict:
        goal = input_data.get("goal", "")
        self.log(f"Generating hypotheses for: {goal}")

        # Use values from config
        prompt = self.build_prompt(goal)
        response = self.call_llm(prompt)
        hypotheses = self.extract_list_items(response)

        self.log(f"Parsed {len(hypotheses)} hypotheses.")
        return {"hypotheses": hypotheses}

    def build_prompt(self, goal):
        return self.prompt_template.format(goal=goal)

    def extract_list_items(self, text: str) -> list[str]:
        matches = re.findall(self.prompt_match_re, text, flags=re.DOTALL)
        return [match.strip() for match in matches]
