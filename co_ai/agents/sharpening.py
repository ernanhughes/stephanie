# co_ai/agents/sharpening.py
from co_ai.agents import BaseAgent
from co_ai.evaluator import MRQSelfEvaluator

class SharpeningAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.target = cfg.get("target", "generation")
        self.device = cfg.get("device", "cpu")
        self.evaluator = MRQSelfEvaluator(memory, logger, device=self.device)
        self.templates = cfg.get("templates", ["critic"])

    async def run(self, context: dict):
        goal = context.get("goal")

        self.evaluator.train_from_database(
            goal=goal,
            limit=1000,
            epochs=self.cfg.get("epochs", 5),
            lr=self.cfg.get("lr", 1e-4))

        prompts = context.get("prompt_history", {}).get(self.target, [])
        results = []
        for data in prompts:
            result = self.run_selected(data, context)
            results.append(result)
        context[self.output_key] = results
        return context

    def run_selected(self, data: dict, context: dict) -> list[dict]:
        results = []
        prompt = data.get("prompt")
        examples = self.memory.hypotheses.get_hypotheses_for_prompt(prompt, 3)
        merged = {**context, **{"prompt": prompt, "examples": examples}}

        if prompt:
            for name in self.templates:
                prompt_template = self.prompt_loader.from_file(name, self.cfg, merged)
                sharpened_hypothesis = self.call_llm(prompt_template, merged)
                hypothesis = data.get("response") # hypotheses result for prompt
                preferred_output, scores = self.evaluator.evaluate(prompt, hypothesis, sharpened_hypothesis)
                results.append({
                    "template": name,
                    "output": preferred_output,
                    "score": scores,
                    # "justification": justification,
                })

        return sorted(results, key=lambda x: x["score"], reverse=True)

    async def sharpen_hypotheses(self, prompt, context: dict):
        examples = self.memory.hypotheses.get_hypotheses_for_prompt(prompt, 3)
        merged = {**context, **{"prompt": prompt, "examples": examples}}
        prompt_template = self.prompt_loader.from_file("critic.txt", self.cfg, merged)

        response = self.call_llm(prompt_template, context)
        return response

    def save_predictions(self, goal, context: dict):
        for prediction in context["sharpening_predictions"]:
            self.memory.mrq.insert_sharpening_prediction(prediction, goal)
            self.logger.log("SharpeningPrediction", prediction)
