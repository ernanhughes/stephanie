# co_ai/agents/generation.py
import re

from co_ai.agents.base import BaseAgent


class GenerationAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

    async def run(self, context: dict) -> dict:
        goal = context.get("goal", "")
        self.log(f"Generating hypotheses for: {goal}")

        # Use values from config
        prompt = self.prompt_loader.load_prompt(self.cfg, context=context)
        response = self.call_llm(prompt)
        hypotheses = self.extract_list_items(response)

        self.log(f"Parsed {len(hypotheses)} hypotheses.")
        context["hypotheses"] = hypotheses
        self.logger.log("GeneratedHypotheses", {
            "goal": goal,
            "hypotheses": hypotheses
        })
        return context

    def extract_list_items(self, text: str) -> list[str]:
        matches = re.findall(self.prompt_match_re, text, flags=re.DOTALL)
        return [match.strip() for match in matches]
