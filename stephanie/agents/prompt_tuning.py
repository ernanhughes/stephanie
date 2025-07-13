# stephanie/agents/prompt_tuning.py
import re
from abc import ABC, abstractmethod

import dspy
from dspy import (BootstrapFewShot, Example, InputField, OutputField, Predict,
                  Signature)

from stephanie.agents.base_agent import BaseAgent
from stephanie.constants import GOAL


# DSPy signature for prompt refinement: defines input/output fields for tuning
class PromptTuningSignature(Signature):
    goal = InputField(desc="Scientific research goal or question")
    input_prompt = InputField(desc="Original prompt used to generate hypotheses")
    hypotheses = InputField(desc="Best hypothesis generated")
    review = InputField(desc="Expert review of the hypothesis")
    score = InputField(desc="Numeric score evaluating the hypothesis quality")
    refined_prompt = OutputField(desc="Improved version of the original prompt")


# Simple evaluation result class to return from evaluator
class EvaluationResult:
    def __init__(self, score: float, reason: str):
        self.score = score
        self.reason = reason


# Base evaluator interface (not used directly, but useful for future extensions)
class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(
        self, original: str, proposal: str, metadata: dict = None
    ) -> EvaluationResult:
        pass


# DSPy-based evaluator that can run a Chain-of-Thought program
class DSPyEvaluator(BaseEvaluator):
    def __init__(self):
        self.program = dspy.ChainOfThought(PromptTuningSignature)

    def evaluate(
        self, original: str, proposal: str, metadata: dict = None
    ) -> EvaluationResult:
        result = self.program(
            goal=metadata["goal"],
            input_prompt=original,
            hypotheses=metadata["hypotheses"],
            review=metadata.get("review", ""),
            score=metadata.get("score", 750),
        )
        try:
            score = float(result.score)
        except (ValueError, TypeError):
            score = 0.0
        return EvaluationResult(score=score, reason=result.explanation)


# Main agent class responsible for training and tuning prompts using DSPy
class PromptTuningAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.agent_name = cfg.get("name", "prompt_tuning")
        self.prompt_key = cfg.get("prompt_key", "default")
        self.sample_size = cfg.get("sample_size", 20)
        self.generate_count = cfg.get("generate_count", 10)
        self.current_version = cfg.get("version", 1)

        # Configure DSPy with local LLM (Ollama)
        lm = dspy.LM(
            "ollama_chat/qwen3",
            api_base="http://localhost:11434",
            api_key="",
        )
        dspy.configure(lm=lm)

    async def run(self, context: dict) -> dict:
        goal = self.extract_goal_text(context.get(GOAL))
        generation_count = self.sample_size + self.generate_count
        self.logger.log(
            "PromptTuningExamples",
            {"samples size": self.sample_size, "generation count": generation_count},
        )

        # Get training + validation data
        examples = self.memory.prompt.get_prompt_training_set(goal, generation_count)
        train_data = examples[: self.sample_size]
        val_data = examples[self.sample_size :]

        if not examples:
            self.logger.log(
                "PromptTuningSkipped", {"reason": "no_training_data", "goal": goal}
            )
            return context

        # Build training set for DSPy
        training_set = [
            Example(
                goal=item["goal"],
                input_prompt=item["prompt_text"],
                hypotheses=item["hypothesis_text"],
                review=item.get("review", ""),
                score=item.get("elo_rating", 1000),
            ).with_inputs("goal", "input_prompt", "hypotheses", "review", "score")
            for item in train_data
        ]

        # Wrap our scoring metric so we can inject context during tuning
        def wrapped_metric(example, pred, trace=None):
            return self._prompt_quality_metric(example, pred, context=context)

        # Train prompt-tuning program
        tuner = BootstrapFewShot(metric=wrapped_metric)
        student = Predict(PromptTuningSignature)
        tuned_program = tuner.compile(student=student, trainset=training_set)

        # Use tuned program to generate and store new refined prompt
        await self.generate_and_store_refined_prompts(
            tuned_program, goal, context, val_data
        )
        self.logger.log(
            "PromptTuningCompleted",
            {
                "goal": goal,
                "example_count": len(training_set),
                "generated_count": len(val_data),
            },
        )

        return context

    async def generate_and_store_refined_prompts(
        self, tuned_program, goal: str, context: dict, val_set
    ):
        """
        Generate refined prompts using the tuned DSPy program and store them in the database.

        Args:
            tuned_program: A compiled DSPy program capable of generating refined prompts.
            goal: The scientific goal for this run.
            context: Shared pipeline state.
            val_set: Validation examples to run through the tuned program.
        """

        stored_count = 0
        for i, example in enumerate(val_set):
            try:
                # Run DSPy program on new example
                result = tuned_program(
                    goal=example["goal"],
                    input_prompt=example["prompt_text"],
                    hypotheses=example["hypothesis_text"],
                    review=example.get("review", ""),
                    score=example.get("elo_rating", 1000),
                )

                refined_prompt = result.refined_prompt.strip()

                # Store refined prompt to the DB
                self.memory.prompt.save(
                    goal={"goal_text": example["goal"]},
                    agent_name=self.name,
                    prompt_key=self.prompt_key,
                    prompt_text=refined_prompt,
                    response=None,
                    pipeline_run_id=context.get("pipeline_run_id"),
                    strategy="refined_via_dspy",
                    version=self.current_version + 1,
                )

                stored_count += 1

                # Update context with prompt history
                self.add_to_prompt_history(
                    context, refined_prompt, {"original": example["prompt_text"]}
                )

                self.logger.log(
                    "TunedPromptStored",
                    {"goal": goal, "refined_snippet": refined_prompt[:100]},
                )

            except Exception as e:
                print(f"❌ Exception: {type(e).__name__}: {e}")
                self.logger.log(
                    "TunedPromptGenerationFailed",
                    {"error": str(e), "example_snippet": str(example)[:100]},
                )

        self.logger.log(
            "BatchTunedPromptsComplete", {"goal": goal, "count": stored_count}
        )

    def _prompt_quality_metric(self, example, pred, context: dict) -> float:
        """Run both prompts and compare results"""
        try:
            prompt_a = example.input_prompt
            prompt_b = pred.refined_prompt
            self.logger.log(
                "PromptQualityCompareStart",
                {
                    "prompt_a_snippet": prompt_a[:100],
                    "prompt_b_snippet": prompt_b[:100],
                },
            )

            hypotheses_a = self.call_llm(prompt_a, context)
            self.logger.log(
                "PromptAResponseGenerated", {"hypotheses_a_snippet": hypotheses_a[:200]}
            )

            hypotheses_b = self.call_llm(prompt_b, context)
            self.logger.log(
                "PromptBResponseGenerated", {"hypotheses_b_snippet": hypotheses_b[:200]}
            )

            # Run comparison
            merged = {
                **context,
                **{
                    "prompt_a": prompt_a,
                    "prompt_b": prompt_b,
                    "hypotheses_a": hypotheses_a,
                    "hypotheses_b": hypotheses_b,
                },
            }
            comparison_prompt = self.prompt_loader.load_prompt(self.cfg, merged)
            self.logger.log(
                "ComparisonPromptConstructed",
                {"comparison_prompt_snippet": comparison_prompt[:200]},
            )

            response = self.call_llm(comparison_prompt, context)
            self.logger.log(
                "ComparisonResponseReceived", {"response_snippet": response[:200]}
            )

            match = re.search(r"better prompt:<([AB])>", response, re.IGNORECASE)
            if match:
                choice = match.group(1).upper()
                score = 1.0 if choice == "B" else 0.5
                self.logger.log(
                    "PromptComparisonResult", {"winner": choice, "score": score}
                )
                return score
            else:
                self.logger.log("PromptComparisonNoMatch", {"response": response})
                return 0.0
        except Exception as e:
            print(f"❌ Exception: {type(e).__name__}: {e}")
            self.logger.log(
                "PromptQualityMetricError",
                {
                    "error": str(e),
                    "example_input_prompt_snippet": example.input_prompt[:100],
                    "refined_prompt_snippet": getattr(pred, "refined_prompt", "")[:100],
                },
            )
            return 0.0
