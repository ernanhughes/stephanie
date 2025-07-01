"""
ChainOfThoughtDSPyGeneratorAgent

This agent uses DSPy to generate a chain-of-thought reasoning trace in response to a given research goal or question.
It loads a goal from the context, uses a DSPy Module configured with a local Ollama model (e.g., qwen3), and returns
a structured reasoning output. The result is stored as a hypothesis and linked to the goal, prompt, and pipeline run
for later evaluation or training (e.g., via MR.Q or LLM-based scoring).

Key features:
- DSPy integration with structured signature (CoTGenerationSignature)
- Local model inference via Ollama (e.g., qwen3)
- Hypothesis logging and memory storage (via Hypothesis ORM)
- Prompt metadata management and linking
- Designed for use in self-improving Co AI pipelines

Intended to be paired with self-judging evaluators and symbolic optimizers.
"""

from abc import ABC, abstractmethod

import dspy
from dspy import InputField, OutputField, Signature

from stephanie.agents.base_agent import BaseAgent
from stephanie.constants import GOAL, GOAL_TEXT


# DSPy signature for generating Chains of Thought
class CoTGenerationSignature(Signature):
    question = InputField(desc="A scientific or reasoning question")
    references = InputField(desc="Optional reference material to inform the reasoning")
    preferences = InputField(desc="Optional reasoning preferences or style constraints")
    answer = OutputField(desc="Chain-of-thought reasoning that addresses the question")


class CoTGeneratorModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generator = dspy.Predict(CoTGenerationSignature)

    def forward(self, question, references="", preferences=""):
        return self.generator(
            question=question, references=references, preferences=preferences
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


# Main agent class responsible for training and tuning prompts using DSPy
class ChainOfThoughtDSPyGeneratorAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

        # Setup DSPy
        lm = dspy.LM(
            "ollama_chat/qwen3",
            api_base="http://localhost:11434",
            api_key="",
        )
        dspy.configure(lm=lm)

        self.module = CoTGeneratorModule()

    async def run(self, context: dict):
        goal = context.get(GOAL)
        references = context.get("references", "")
        preferences = context.get("preferences", "")

        result = self.module(
            question=goal.get("goal_text"), references=references, preferences=preferences
        )

        cot = result.answer.strip()
        self.logger.log("CoTGenerated", {"goal": goal, "cot": cot})

        prompt_text = goal.get(GOAL_TEXT)
        prompt = self.get_or_save_prompt(prompt_text, context)
        hyp = self.save_hypothesis(
            {
                "prompt_id": prompt.id,
                "text": cot,
                "features": {"source": "cot_dspy"},
            },
            context=context,
        )
        context[self.output_key] = [hyp.to_dict()]
        return context
