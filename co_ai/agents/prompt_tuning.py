# co_ai/agents/prompt_tuning.py
from co_ai.agents.base import BaseAgent
from dspy import LM, BootstrapFewShot, InputField, OutputField, Predict, Signature, configure
import requests
import dspy
from ollama import chat
from abc import ABC, abstractmethod
import re
from co_ai.constants import GOAL

class HypothesisRefinementSignature(Signature):
    goal = InputField(desc="Scientific research objective")
    hypothesis = InputField(desc="Current hypothesis under evaluation")
    review = InputField(desc="Expert review of hypothesis")
    score = InputField(desc="Elo rating or ranking score")
    refined_hypothesis = OutputField(desc="Improved version of hypothesis")

class EvaluationResult:
    def __init__(self, score: int, reason: str):
        self.score = score
        self.reason = reason

class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, original: str, proposal: str, metadata: dict = None) -> EvaluationResult:
        pass

class EvaluateEnhancement(dspy.Signature):
    original_section = dspy.InputField()
    proposal = dspy.InputField()
    score = dspy.OutputField(desc="Score from 0.0 (worse) to 1.0 (excellent)")
    explanation = dspy.OutputField(desc="Why this proposal improves (or worsens) the section")

class DSPyEvaluator(BaseEvaluator):
    def __init__(self):
        self.program = dspy.ChainOfThought(EvaluateEnhancement)

    def evaluate(self, original: str, proposal: str, metadata: dict = None) -> EvaluationResult:
        result = self.program(original_section=original, proposal=proposal)
        try:
            score = float(result.score)
        except (ValueError, TypeError):
            score = 0.0
        return EvaluationResult(score=score, reason=result.explanation)

class OllamaEvaluator(BaseEvaluator):
    def __init__(self, model: str = "qwen2.5"):
        self.model = model

    def evaluate(self, original: str, proposal: str, metadata: dict = None) -> EvaluationResult:
        prompt = f"""
You are evaluating a proposed rewrite of a section in an AI book.

Original:
---
{original}
---

Proposal:
---
{proposal}
---

Does this proposal meaningfully improve the original? Give a score from 1–10 and explain why.
""".strip()

        response = chat(model=self.model, messages=[{'role': 'user', 'content': prompt}])
        output = response.message.content.strip()

        score = self._extract_score(output)
        return EvaluationResult(score=score, reason=output)

    def _extract_score(self, text: str) -> int:
        match = re.search(r'([1-9]|10)', text)
        return int(match.group(1)) if match else 0


class PromptTuningAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.agent_name = cfg.get("name", "prompt_tuning")
        self.prompt_key = cfg.get("prompt_key", "default")
        self.current_version = cfg.get("version", 1)

        ollama_model = cfg.get("llm_model", "ollama/qwen3")
        ollama_api_base = cfg.get("api_base", "http://localhost:11434")

        class LocalLLM(LM):
            def __init__(self):
                super().__init__(model=ollama_model, api_base=ollama_api_base)

            def _generate(self, prompt: str, max_tokens: int = 2048):
                payload = {
                    "model": self.model.split("/")[-1],
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": 0.7,
                    "stream": False
                }
                try:
                    response = requests.post(self.api_base, json=payload)
                    return [response.json().get("response", "").strip()]
                except Exception as e:
                    print(f"[LLM] Call failed: {str(e)}")
                    return [""]

        self.lm = LocalLLM()
        configure(lm=self.lm)

    async def run(self, context: dict) -> dict:
        goal = context.get(GOAL, "")
        seed_prompts = context.get("seed_prompts", [])
        if not seed_prompts:
            latest_prompts = self.memory.prompt.get_latest_prompts(5)
            seed_prompts = [
                p["prompt_text"] for p in latest_prompts
            ]
            self.logger.log("PromptTuningSkipped", {"reason": "no_seed_prompts"})

        ranked = self.memory.hypotheses.get_ranked(goal,20)

        few_shot_data = self.memory.hypotheses.get_ranked(goal,20)
        training_set = [
            {
                "goal": item["goal"],
                "hypotheses": item["hypotheses"],
                "review": item.get("review", ""),
                "score": item.get("elo_rating", 1000)
            } for item in few_shot_data
        ]

        tuner = BootstrapFewShot(metric=self._exact_match_metric)
        program = Predict(HypothesisRefinementSignature)
        # tuned_program = tuner.compile(
        #     student=Predict(HypothesisRefinementSignature),
        #     trainset=training_set
        # )
        # refined_prompt = tuned_program.prompt
        #
        # old_prompt = seed_prompts[0]
        # old_score = self._evaluate_prompt(old_prompt, few_shot_data)
        # new_score = self._evaluate_prompt(refined_prompt, few_shot_data)
        #
        # if new_score > old_score:
        #     self.logger.log("PromptRefinedAndImproved", {
        #         "agent": self.agent_name,
        #         "prompt_key": self.prompt_key,
        #         "old_score": old_score,
        #         "new_score": new_score
        #     })
        #
        #     self.memory.hypotheses.store_prompt_version(
        #         agent_name=self.agent_name,
        #         prompt_key=self.prompt_key,
        #         prompt_text=refined_prompt,
        #         input_keys=["goal", "literature", "preferences"],
        #         output_key="hypotheses",
        #         extraction_regex=r"Hypothesis 1:\n(.+?)\n\nHypothesis 2:",
        #         source="dsp_refinement",
        #         version=self.current_version + 1,
        #         is_current=True,
        #         metadata={"few_shot_count": len(few_shot_data), "improvement_score": new_score - old_score}
        #     )
        # else:
        #     self.logger.log("PromptRefiningNoImprovement", {
        #         "agent": self.agent_name,
        #         "prompt_key": self.prompt_key,
        #         "old_score": old_score,
        #         "new_score": new_score
        #     })
        #
        # context["refined_prompt"] = refined_prompt
        return context


    def evaluate_prompt_version(self, original: str, proposal: str, use_dspy: bool = True):
        evaluator = DSPyEvaluator() if use_dspy else OllamaEvaluator()
        result = evaluator.evaluate(original, proposal)
        print(f"Score: {result.score}")
        print(f"Reason: {result.reason}")

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
