from abc import ABC, abstractmethod

import re
import dspy
from dspy import Predict, Signature, InputField, OutputField, Example, BootstrapFewShot

from co_ai.agents.base import BaseAgent
from co_ai.constants import GOAL
from co_ai.models import Hypothesis


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
            question=question,
            references=references,
            preferences=preferences
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
        self.agent_name = cfg.get("name", "cot_dspy_generator")

        # Setup DSPy
        lm = dspy.LM(
            "ollama_chat/qwen:0.5b",
            api_base="http://localhost:11434",
            api_key="",
        )
        dspy.configure(lm=lm)

        self.module = CoTGeneratorModule()

    async def run(self, context: dict):
        goal = self.extract_goal_text(context.get(GOAL))
        references = context.get("references", "")
        preferences = context.get("preferences", "")

        result = self.module(
            question=goal,
            references=references,
            preferences=preferences
        )

        cot = result.answer.strip()
        self.logger.log("CoTGenerated", {"goal": goal, "cot": cot})

        hyp = Hypothesis(goal=goal, goal_type=context.get(GOAL).get("goal_type"), text=cot, features={"source": "cot_dspy"},
                         prompt=goal)
        self.memory.hypotheses.store(hyp)

        context[self.output_key] = [cot]
        return context
