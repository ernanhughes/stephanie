# co_ai/agents/sharpening.py
from co_ai.agents import BaseAgent
from co_ai.evaluator import MRQSelfEvaluator
from co_ai.memory.memory_types import SharpeningPrediction
import re

class SharpeningAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.target = cfg.get("target", "generation")
        self.device = cfg.get("device", "cpu")
        self.evaluator = MRQSelfEvaluator(memory, device=self.device)

    async def run(self, context: dict):
        goal = context.get("goal")
        limit = self.cfg.get("limit", 100)

        dpo_samples = self.memory.mrq.get_training_pairs(goal=goal, limit=limit)
        if not dpo_samples:
            self.logger.log("NoDPOSamplesFound", {"goal": goal})
            return context
        predictions = []
        for sample in dpo_samples:
            preferred_output, scores = self.evaluator.evaluate(
                sample["prompt"],
                sample["output_a"],
                sample["output_b"]
            )

            prediction = {
                "prompt": sample["prompt"],
                "output_a": sample["output_a"],
                "output_b": sample["output_b"],
                "preferred": sample["preferred"],
                "predicted": "a" if preferred_output == sample["output_a"] else "b",
                "scores": scores
            }
            predictions.append(prediction)

        context["sharpening_predictions"] = predictions
        return context



        prompts = context.get("prompt_history2", {}).get(self.target, [])
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

    async def sharpen(self, prompt, context: dict):
        merged = {**context, **{"prompt": prompt}}
        prompt_template = self.prompt_loader.from_file(
            "sharpening.txt", self.cfg, merged
        )

        response = self.call_llm(prompt_template, context)
        return response, self._get_self_reward(response, context)

    def _get_self_reward(self, response: str, context: dict) -> float:
        merged = {**context, **{"response":response}}
        critique_prompt = self.prompt_loader.from_file("self_reward.txt", self.cfg, merged)
        result = self.call_llm(critique_prompt, context)
        match = re.search(r"<self_reward>(\d+)</self_reward>", result)
        return int(match.group(1)) if match else 7  # Default to average


    def save_predictions(self, goal, context: dict):
        for prediction in context["sharpening_predictions"]:
            self.memory.mrq.insert_sharpening_prediction(prediction, goal)
            self.logger.log("SharpeningPrediction", prediction)
