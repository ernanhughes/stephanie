from abc import ABC, abstractmethod

import dspy
from dspy import LM, Predict, Signature, InputField, OutputField, Example, BootstrapFewShot, configure

from co_ai.agents.base import BaseAgent
from co_ai.constants import GOAL

class HypothesisRefinementSignature(Signature):
    goal = InputField(desc="Scientific research objective")
    hypothesis = InputField(desc="Current hypothesis under evaluation")
    review = InputField(desc="Expert review of hypothesis")
    score = InputField(desc="Elo rating or ranking score")
    refined_hypothesis = OutputField(desc="Improved version of hypothesis")

class EvaluationResult:
    def __init__(self, score: float, reason: str):
        self.score = score
        self.reason = reason


class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, original: str, proposal: str, metadata: dict = None) -> EvaluationResult:
        pass


class DSPyEvaluator(BaseEvaluator):
    def __init__(self):
        self.program = dspy.ChainOfThought(HypothesisRefinementSignature)

    def evaluate(self, original: str, proposal: str, metadata: dict = None) -> EvaluationResult:
        result = self.program(goal=metadata["goal"], hypothesis=original, review=metadata.get("review", ""), score=metadata.get("score", 1000))
        try:
            score = float(result.score)
        except (ValueError, TypeError):
            score = 0.0
        return EvaluationResult(score=score, reason=result.explanation)



class PromptTuningAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.agent_name = cfg.get("name", "prompt_tuning")
        self.prompt_key = cfg.get("prompt_key", "default")
        self.current_version = cfg.get("version", 1)
        lm = dspy.LM(
            "ollama_chat/qwen3",
            api_base="http://localhost:11434",
            api_key="",
        )
        dspy.configure(lm=lm)

    async def run(self, context: dict) -> dict:
        goal = context.get(GOAL, "")
        seed_prompts = context.get("seed_prompts", [])

        if not seed_prompts:
            latest_prompts = self.memory.prompt.get_latest_prompts(5)
            seed_prompts = [p["prompt_text"] for p in latest_prompts]
            self.logger.log("PromptTuningSkipped", {"reason": "no_seed_prompts"})

        few_shot_data = self.memory.hypotheses.get_ranked(goal, 20)

        training_set = [
            Example(
                goal=item["goal"],
                hypothesis=item["hypotheses"],
                review=item.get("review", ""),
                score=item.get("elo_rating", 1000),
                refined_hypothesis=item.get("refined_hypothesis", "")
            ).with_inputs("goal", "hypothesis", "review", "score")
            for item in few_shot_data
        ]

        tuner = BootstrapFewShot(metric=self._exact_match_metric)
        tuned_program = tuner.compile(
            student=Predict(HypothesisRefinementSignature),
            trainset=training_set
        )

        context["refined_prompt"] = tuned_program.prompt
        return context

    def _exact_match_metric(self, example, pred, trace=None):
        return example.refined_hypothesis.lower() == pred.refined_hypothesis.lower()

    def _evaluate_prompt(self, prompt: str, test_data: list[dict]) -> float:
        total_score = 0
        for item in test_data:
            try:
                response = self.lm(prompt.format(**item))
                refined = response[0].strip()
                if refined:
                    total_score += item.get("score", 1000) * 0.1
            except Exception as e:
                self.logger.log("PromptEvaluationFailed", {"error": str(e)})
        return total_score / len(test_data) if test_data else 0