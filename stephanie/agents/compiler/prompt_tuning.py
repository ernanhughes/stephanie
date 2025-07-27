# stephanie/agents/compiler/prompt_tuning.py
from abc import ABC, abstractmethod

import dspy
from dspy import (BootstrapFewShot, Example, InputField, OutputField, Predict,
                  Signature)

from stephanie.agents.base_agent import BaseAgent
from stephanie.constants import GOAL
from stephanie.scoring.mrq_scorer import MRQScorer


# DSPy signature for prompt refinement: defines input/output fields for tuning
class PromptTuningSignature(Signature):
    goal = InputField(desc="Scientific research goal or question")
    input_prompt = InputField(
        desc="Original prompt used to generate hypotheses"
    )
    hypotheses = InputField(desc="Best hypothesis generated")
    review = InputField(desc="Expert review of the hypothesis")
    score = InputField(desc="Numeric score evaluating the hypothesis quality")
    refined_prompt = OutputField(
        desc="Improved version of the original prompt"
    )


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

        self.scorer = MRQScorer(cfg, memory, logger)
        self.scorer.load_models()

    async def run(self, context: dict) -> dict:
        goal = self.extract_goal_text(context.get(GOAL))
        step_outputs = context.get("step_outputs", [])
        if not step_outputs:
            self.logger.log(
                "PromptTuningSkipped", {"reason": "no_steps_found"}
            )
            return context

        self.logger.log(
            "StepPromptTuningStart", {"step_count": len(step_outputs)}
        )

        tuned_steps = []
        for i, step in enumerate(step_outputs):
            try:
                original_prompt = step.get("step")  # fallback
                original_output = step["output"]
                original_score = step.get("score", 1000)

                self.logger.log(
                    "TuningInputExample",
                    {
                        "goal": goal,
                        "input_prompt": original_prompt,
                        "hypotheses": original_output,
                        "score": original_score,
                    },
                )

                # Run DSPy prompt tuning program
                example = Example(
                    goal=goal,
                    input_prompt=original_prompt,
                    hypotheses=original_output,
                    review="",  # or step.get("review", "")
                    score=original_score,
                ).with_inputs(
                    "goal", "input_prompt", "hypotheses", "review", "score"
                )

                # Generate refined prompt candidates
                program = Predict(PromptTuningSignature)
                tuned_program = BootstrapFewShot(
                    metric=lambda e, p, _: 1.0
                ).compile(
                    student=program,
                    trainset=[example],  # one-shot tuning
                )

                result = tuned_program(
                    goal=goal,
                    input_prompt=original_prompt,
                    hypotheses=original_output,
                    review="",
                    score=original_score,
                )
                self.logger.log(
                    "TunedProgramResult",
                    {"step_index": i, "result": str(result)},
                )
                print(f"Refined prompt: {result}")
                refined_prompt = result.refined_prompt.strip()

                # Score both versions
                prompt_a = original_prompt
                prompt_b = refined_prompt

                score_a = self._prompt_quality_metric(
                    example=example,
                    pred=type("obj", (object,), {"refined_prompt": prompt_a}),
                    context=context,
                )
                score_b = self._prompt_quality_metric(
                    example=example,
                    pred=type("obj", (object,), {"refined_prompt": prompt_b}),
                    context=context,
                )

                # Choose best
                best_prompt = prompt_b if score_b > score_a else prompt_a
                best_score = max(score_a, score_b)

                step["refined_prompt"] = best_prompt
                step["refinement_history"] = {
                    "original": prompt_a,
                    "refined_candidate": prompt_b,
                    "score_a": score_a,
                    "score_b": score_b,
                    "selected": best_prompt,
                }
                step["score"] = best_score

                # Store in prompt memory (if needed)
                self.memory.prompt.save(
                    goal={"goal_text": goal},
                    agent_name=self.name,
                    prompt_key=f"step_{i}",
                    prompt_text=best_prompt,
                    response=None,
                    pipeline_run_id=context.get("pipeline_run_id"),
                    strategy="step_prompt_refined",
                    version=self.current_version + 1,
                )

                tuned_steps.append(step)

                self.logger.log(
                    "StepPromptRefined",
                    {
                        "step_index": i,
                        "original_snippet": prompt_a[:100],
                        "refined_snippet": prompt_b[:100],
                        "winner": "B" if score_b > score_a else "A",
                        "score_a": score_a,
                        "score_b": score_b,
                    },
                )

            except Exception as e:
                self.logger.log(
                    "StepPromptTuningFailed",
                    {
                        "step_index": i,
                        "error": str(e),
                        "step_snippet": str(step)[:100],
                    },
                )

        context["step_outputs"] = tuned_steps
        self.logger.log(
            "StepPromptTuningCompleted", {"count": len(tuned_steps)}
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

                self.logger.log(
                    "TunedProgramResult",
                    {"step_index": i, "result": str(result)},
                )
                print(f"Refined prompt: {result}")

                # Safely extract refined prompt
                if (
                    not result
                    or not hasattr(result, "refined_prompt")
                    or result.refined_prompt is None
                ):
                    raise ValueError(
                        "Refined prompt not returned from DSPy program."
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
                    context,
                    refined_prompt,
                    {"original": example["prompt_text"]},
                )

                self.logger.log(
                    "TunedPromptStored",
                    {"goal": goal, "refined_snippet": refined_prompt[:100]},
                )

            except Exception as e:
                self.logger.log(
                    "StepPromptTuningFailed",
                    {
                        "step_index": i,
                        "error": str(e),
                        "step_snippet": str(example.get("prompt_text", ""))[
                            :100
                        ],
                    },
                )
                print(f"❌ Exception: {type(e).__name__}: {e}")

        self.logger.log("StepPromptTuningCompleted", {"count": stored_count})

    def _prompt_quality_metric(self, example, pred, context: dict) -> float:
        """
        Evaluate whether the refined prompt (pred.refined_prompt) is better than the original (example.input_prompt),
        using fast MR.Q-style prediction from memory embeddings when possible.

        Returns:
            1.0 if refined is better
            0.5 if equal
            0.0 if original is better
        """
        try:
            # Extract both prompts
            prompt_a = example.input_prompt.strip()
            prompt_b = (
                pred.refined_prompt.strip() if pred.refined_prompt else ""
            )

            if not prompt_b:
                self.logger.log(
                    "StepPromptTuningFailed",
                    {
                        "step_index": context.get("step_index"),
                        "error": "Refined prompt is empty",
                        "step_snippet": str(context.get("step", ""))[:200],
                    },
                )
                return 0.0

            # Use dimension-aware scorer to get detailed scores
            dimensions = context.get(
                "dimensions", ["correctness", "clarity", "relevance"]
            )

            # Try fast MR.Q prediction via embeddings first
            try:
                score_dict_a = {
                    dim: self.scorer.predict_score_from_prompt(prompt_a, dim)
                    for dim in dimensions
                }
                score_dict_b = {
                    dim: self.scorer.predict_score_from_prompt(prompt_b, dim)
                    for dim in dimensions
                }

                # Weighted comparison
                weighted_score_a = self._score_weighted(score_dict_a)
                weighted_score_b = self._score_weighted(score_dict_b)

                mode = "predicted"

            except Exception as e:
                self.logger.log(
                    "MRQPredictionFailed",
                    {"error": str(e), "fallback": "llm_scoring"},
                )

                # Fallback: Call LLM to generate output and score it
                goal = context.get("goal", {}).get("goal_text", "")
                output_a = self.call_llm(prompt_a, context={"goal": goal})
                output_b = self.call_llm(prompt_b, context={"goal": goal})

                # Score outputs using MR.Q
                score_dict_a = self.scorer.score(
                    goal={"goal_text": goal},
                    hypothesis={"text": output_a},
                    dimensions=dimensions,
                ).to_dict()
                score_dict_b = self.scorer.score(
                    goal={"goal_text": goal},
                    hypothesis={"text": output_b},
                    dimensions=dimensions,
                ).to_dict()

                weighted_score_a = self._score_weighted(score_dict_a)
                weighted_score_b = self._score_weighted(score_dict_b)

                mode = "llm"

            # Log full evaluation result
            self.logger.log(
                "PromptQualityComparison",
                {
                    "mode": mode,
                    "prompt_a_snippet": prompt_a[:100],
                    "prompt_b_snippet": prompt_b[:100],
                    "scores_a": score_dict_a,
                    "scores_b": score_dict_b,
                    "weighted_score_a": weighted_score_a,
                    "weighted_score_b": weighted_score_b,
                    "winner": "B"
                    if weighted_score_b > weighted_score_a
                    else (
                        "A" if weighted_score_b < weighted_score_a else "Tie"
                    ),
                },
            )

            # Return binary decision signal
            if weighted_score_b > weighted_score_a:
                return 1.0
            elif weighted_score_b < weighted_score_a:
                return 0.0
            else:
                return 0.5

        except Exception as e:
            self.logger.log(
                "StepPromptTuningFailed",
                {
                    "step_index": context.get("step_index"),
                    "error": str(e),
                    "step_snippet": str(context.get("step", ""))[:200],
                },
            )
            return 0.0

    def _score_weighted(self, dim_scores: dict) -> float:
        """
        Weighted aggregation of dimensional scores.
        Can be customized per task (e.g., science vs storytelling).
        """
        weights = {
            "correctness": 0.4,
            "clarity": 0.3,
            "relevance": 0.2,
            "originality": 0.1,
        }
        return sum(dim_scores.get(k, 0) * w for k, w in weights.items())
