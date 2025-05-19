# co_ai/agents/sharpening.py
from co_ai.agents import BaseAgent
from co_ai.evaluator import MRQSelfEvaluator
import re

class SharpeningAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.target = cfg.get("target", "generation")
        self.device = cfg.get("device", "cpu")
        self.evaluator = MRQSelfEvaluator(memory, device=self.device)

    async def sharpen(self, prompt, context: dict):
        merged = {**context, **{"prompt": prompt}}
        prompt_template = self.prompt_loader.from_file(
            "sharpening.txt", self.cfg, merged
        )

        response = self.call_llm(prompt_template, context)
        return response, self._get_self_reward(response, context)

    async def run(self, context: dict):
        prompts = context.get("prompt_history", {}).get(self.target, [])
        for data in prompts:
            prompt = data.get("prompt")
            if prompt:
                sharpened_output, scores = await self.sharpen(prompt, context)
                log_entry = {
                    "prompt": prompt,
                    "sharpened_output": sharpened_output,
                    "scores": scores,
                }
                self.logger.log("SharpeningWithMRQEvaluation", log_entry)

        return context

    def _get_self_reward(self, response: str, context: dict) -> float:
        merged = {**context, **{"response":response}}
        critique_prompt = self.prompt_loader.from_file("self_reward.txt", self.cfg, merged)
        result = self.call_llm(critique_prompt, context)
        match = re.search(r"<self_reward>(\d+)</self_reward>", result)
        return int(match.group(1)) if match else 7  # Default to average