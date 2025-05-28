from abc import ABC, abstractmethod

import dspy
from dspy import InputField, OutputField, Signature

from co_ai.agents.base import BaseAgent
from co_ai.constants import GOAL, PIPELINE, PROMPT_PATH, STRATEGY
from co_ai.models import HypothesisORM


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

        prompt_text = goal.goal_text
        prompt = self.memory.prompt.get_from_text(prompt_text)
        if prompt is None:
            self.memory.prompt.save(
                context.get("goal"),
                agent_name=self.name,
                prompt_key=self.cfg.get(PROMPT_PATH, ""),
                prompt_text=prompt,
                strategy=self.cfg.get(STRATEGY, ""),
                version=self.cfg.get("version", 1),
            )


        hyp = HypothesisORM(
            goal_id=goal.id,
            goal_type=context.get(GOAL).get("goal_type"),
            text=cot,
            features={"source": "cot_dspy"},
            prompt=prompt_text,
            pipeline_signature=context.get(PIPELINE),
        )
        self.memory.hypotheses.insert(hyp)

        context[self.output_key] = [cot]
        return context
