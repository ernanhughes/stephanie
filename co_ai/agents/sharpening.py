# co_ai/agents/sharpening.py
from co_ai.agents import BaseAgent
from co_ai.evaluator import MRQSelfEvaluator


class SharpeningAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.target = cfg.get("target", "generation")
        self.device = cfg.get("device", "cpu")
        self.evaluator = MRQSelfEvaluator(memory, device=self.device)

    async def generate_output(self, prompt, context):
        response = await self.llm(prompt, context)
        return response

    async def sharpen(self, prompt, context: dict):
        goal = context.get("goal")
        self.logger.log("Sharpening", {"Prompt": prompt, "goal": goal})
        merged = {**context, **{"prompt": prompt}}
        prompt = self.prompt_loader.load_prompt(self.cfg, merged)
        response = self.generate_output(prompt, merged)
        return response, 100

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
