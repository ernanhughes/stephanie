import dspy
from dspy import (
    BootstrapFewShot,
    Example,
    InputField,
    OutputField,
    Signature,
    ChainOfThought,
)

from co_ai.agents.base_agent import BaseAgent
from co_ai.constants import GOAL
from co_ai.scoring.mrq_scorer import MRQScorer
from co_ai.agents.mixins.scoring_mixin import ScoringMixin
from co_ai.agents.mixins.memory_aware_mixin import MemoryAwareMixin


# DSPy signature for merging multiple high-quality prompts into a coherent prompt
class PromptMergeSignature(Signature):
    goal = InputField(desc="The original scientific or research goal.")
    prompts = InputField(desc="List of high-quality prompts to intelligently merge.")
    merged_prompt = OutputField(
        desc="A coherent merged prompt that integrates the best aspects of the provided prompts."
    )


# DSPy module implementing intelligent merging of prompts
class PromptMerger(dspy.Module):
    def __init__(self):
        super().__init__()
        self.merger = ChainOfThought(PromptMergeSignature)

    def forward(self, goal: str, prompts: list[str]) -> dspy.Prediction:
        prompt_text = "\n\n---\n\n".join([f"Prompt {i+1}:\n{p}" for i, p in enumerate(prompts)])
        return self.merger(goal=goal, prompts=prompt_text)


class DSPyAssemblerAgent(ScoringMixin, MemoryAwareMixin, BaseAgent):
    """
    DSPyAssembler uses DSPy to merge and refine multiple prompt variants into one optimal prompt.
    """

    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.scorer = MRQScorer(cfg, memory, logger)
        self.scorer.load_models()

        # Configure local LLM (e.g., Ollama)
        self.lm = dspy.LM(
            "ollama_chat/qwen3", api_base="http://localhost:11434", api_key=""
        )
        dspy.configure(lm=self.lm)

        # Initialize modules
        self.prompt_merger = PromptMerger()
        self.mrq_eval_metric = self._mrq_eval_metric
        self.max_included = cfg.get("max_included", 15)

    async def run(self, context: dict) -> dict:
        goal = self.extract_goal_text(context.get(GOAL))
        step_outputs = context.get("step_outputs", [])

        if not step_outputs:
            self.logger.log("PromptTuningSkipped", {"reason": "no_steps_found"})
            return context

        self.logger.log("StepPromptTuningStart", {"step_count": len(step_outputs)})

        # 1. Get top N prompts by score
        ranked_prompts = sorted(
            [step for step in step_outputs if step.get("score")],
            key=lambda x: x["score"],
            reverse=True,
        )

        top_prompts = [
            step.get("refined_prompt") or step.get("prompt")
            for step in ranked_prompts[:self.max_included]
        ]
        top_prompts = [p.strip() for p in top_prompts if p]

        # 2. Create training examples from top-performing prompts
        train_examples = [
            Example(
                goal=goal,
                prompts=top_prompts,
                merged_prompt="",  # Will be filled during tuning
            ).with_inputs("goal", "prompts")
            for _ in range(5)  # Generate 5 example sets
        ]

        # Wrap our scoring metric so we can inject context during tuning
        def wrapped_metric(example, pred, trace=None):
            return self.mrq_eval_metric(example, pred, trace, context=context)

        # 3. Compile tuned merger with BootstrapFewShot
        tuner = BootstrapFewShot(metric=wrapped_metric, max_bootstrapped_demos=4)
        compiled_merger = tuner.compile(self.prompt_merger, trainset=train_examples)

        # 4. Merge prompts using trained module
        try:
            merged_prediction = compiled_merger(goal=goal, prompts=top_prompts)
            merged_prompt = merged_prediction.merged_prompt.strip()
        except Exception as e:
            self.logger.log("PromptMergeFailed", {"error": str(e)})
            merged_prompt = top_prompts[0]  # Fallback to best known prompt

        # 5. Score merged result
        try:
            hypothesis = self.call_llm(merged_prompt, context)
            score_bundle = self.score_hypothesis(
                {"text": hypothesis}, context, metrics="compiler", scorer=self.scorer
            )
            final_score = score_bundle.aggregate()
        except Exception as e:
            self.logger.log("FinalScoreFailed", {"error": str(e)})
            final_score = 0.0

        # 6. Save to context
        context["final_merged_prompt"] = merged_prompt
        context["final_score"] = final_score

        self.logger.log(
            "PromptMergeCompleted",
            {"merged_prompt_snippet": merged_prompt[:200], "final_score": final_score},
        )

        return context

    def _mrq_eval_metric(self, example, pred, trace=None, context=None):
        """Evaluation metric using MR.Q scorer"""
        try:
            merged_prompt = pred.merged_prompt
            hypothesis = self.call_llm(merged_prompt, context=context)
            score_bundle = self.score_hypothesis(
                {"text": hypothesis}, context, metrics="compiler", scorer=self.scorer
            )
            aggregate_score = score_bundle.aggregate()
            normalized_score = aggregate_score / 100.0  # Normalize to [0, 1]
            return normalized_score
        except Exception as e:
            self.logger.log("MRQEvalMetricError", {"error": str(e)})
            return 0.0
