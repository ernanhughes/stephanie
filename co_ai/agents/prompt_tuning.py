from abc import ABC, abstractmethod

import dspy
from dspy import Predict, Signature, InputField, OutputField, Example, BootstrapFewShot

from co_ai.agents.base import BaseAgent
from co_ai.constants import GOAL


class PromptTuningSignature(Signature):
    goal = InputField(desc="Scientific research goal or question")
    input_prompt = InputField(desc="Original prompt used to generate hypotheses")
    hypotheses = InputField(desc="Best hypothesis generated")
    review = InputField(desc="Expert review of the hypothesis")
    score = InputField(desc="Numeric score evaluating the hypothesis quality")
    refined_prompt = OutputField(desc="Improved version of the original prompt")

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
        self.program = dspy.ChainOfThought(PromptTuningSignature)

    def evaluate(self, original: str, proposal: str, metadata: dict = None) -> EvaluationResult:
        result = self.program(
            goal=metadata["goal"],
            input_prompt=original,
            hypotheses=metadata["hypotheses"],
            review=metadata.get("review", ""),
            score=metadata.get("score", 750)
        )
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
        self.sample_size = cfg.get("sample_size", 20)
        self.generate_count = cfg.get("generate_count", 10)
        self.current_version = cfg.get("version", 1)
        lm = dspy.LM(
            "ollama_chat/qwen3",
            api_base="http://localhost:11434",
            api_key="",
        )
        dspy.configure(lm=lm)

    async def run(self, context: dict) -> dict:
        goal = context.get(GOAL, "")
        generation_count = self.sample_size + self.generate_count
        examples = self.memory.prompt.get_prompt_training_set(
            goal, generation_count
        )
        train_data = examples[:self.sample_size]
        val_data = examples[self.sample_size:]

        if not examples:
            self.logger.log(
                "PromptTuningSkipped", {"reason": "no_training_data", "goal": goal}
            )
            return context

        training_set = [
            Example(
                goal=item["goal"],
                input_prompt=item["prompt_text"],
                hypotheses=item["hypothesis_text"],
                review=item.get("review", ""),
                score=item.get("elo_rating", 1000),
                refined_prompt=item.get("refined_prompt", item["prompt_text"])  # fallbackIt's kind of like this
            ).with_inputs("goal", "input_prompt", "hypotheses", "review", "score")
            for item in train_data
        ]

        tuner = BootstrapFewShot(metric=self._exact_match_metric)
        tuned_program = tuner.compile(
            student=Predict(PromptTuningSignature), trainset=training_set
        )
        await self.generate_and_store_refined_prompts(tuned_program, goal, context, val_data)
        self.logger.log(
            "PromptTuningCompleted",
            {
                "goal": goal,
                "example_count": len(training_set),
                "refined_prompt_snippet": context["refined_prompt"][:200],
            },
        )

        return context

    def _exact_match_metric(self, example, pred, trace=None):
        return example.refined_prompt.lower() == pred.refined_prompt.lower()

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


    async def generate_and_store_refined_prompts(
        self, tuned_program, goal: str, context: dict, val_set):
        """
        Generate refined prompts using the tuned DSPy program and store them in the database.

        Args:
            tuned_program: A compiled DSPy program capable of generating refined prompts.
            goal: The scientific goal to use for all examples.
            context: The context

        """
        # Retrieve top hypothesis examples with reviews to use as inputs
        stored_count = 0
        for i, example in enumerate(val_set):
            try:
                result = tuned_program(
                    goal=example["goal"],
                    input_prompt=example["prompt_text"],
                    hypotheses=example["hypothesis_text"],
                    review=example.get("review", ""),
                    score=example.get("elo_rating", 1000),
                )

                refined_prompt = result.refined_prompt.strip()
                self.memory.prompt.save(
                    goal=example["goal"],
                    agent_name=self.name,
                    prompt_key=self.prompt_key,
                    prompt_text=refined_prompt,
                    response=None,
                    strategy="refined_via_dspy",
                    version=self.current_version + 1,
                )

                stored_count += 1

                self.logger.log(
                    "TunedPromptStored",
                    {"goal": goal, "refined_snippet": refined_prompt[:100]},
                )

            except Exception as e:
                self.logger.log(
                    "TunedPromptGenerationFailed",
                    {"error": str(e), "example_snippet": str(example)[:100]},
                )

        self.logger.log("BatchTunedPromptsComplete", {
            "goal": goal,
            "count": stored_count
        })
