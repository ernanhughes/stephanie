# stephanie/agents/cot_generator.py
from __future__ import annotations

from stephanie.agents.base_agent import BaseAgent
from stephanie.analysis.rubric_classifier import RubricClassifierMixin
from stephanie.constants import GOAL
from stephanie.evaluator.llm_judge_evaluator import LLMJudgeEvaluator
from stephanie.scoring.scorable import ScorableFactory, ScorableType
from stephanie.scoring.scorer.contrastive_ranker_scorer import \
    ContrastiveRankerScorer
from stephanie.scoring.scorer.ebt_scorer import EBTScorer
from stephanie.scoring.scorer.hrm_scorer import HRMScorer
from stephanie.scoring.scorer.mrq_scorer import MRQScorer
from stephanie.scoring.scorer.sicql_scorer import SICQLScorer
from stephanie.scoring.scorer.svm_scorer import SVMScorer


class ChainOfThoughtGeneratorAgent(BaseAgent, RubricClassifierMixin):
    """
    Generates chain-of-thought candidates and evaluates them
    using a configurable evaluator/scorer.
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.logger.log("AgentInit", {"agent": "ChainOfThoughtGeneratorAgent"})
        self.num_candidates = cfg.get("num_candidates", 2)
        self.evaluator = self._init_evaluator(cfg)

    async def run(self, context: dict):
        goal = context.get(GOAL)
        context["__llm"] = self.call_llm
        self.logger.log("AgentRunStarted", {"goal": goal})

        # 1. Generate candidates
        merged = {
            **context,
            "gen": lambda slot: ChainOfThoughtGeneratorAgent.gen(
                slot, context
            ),
        }

        prompt_text = self.prompt_loader.load_prompt(self.cfg, merged)
        candidates = [
            self.call_llm(prompt_text, context)
            for _ in range(self.num_candidates)
        ]
        self.logger.log(
            "GenerationCompleted",
            {"candidates": [c[:100] for c in candidates]},
        )
        self.report(
            {
                "event": "candidates_generated",
                "count": len(candidates),
                "examples": [c[:120] for c in candidates],
            }
        )

        # 2. Evaluate candidates pairwise
        best = candidates[0]
        scores = {}
        for candidate in candidates[1:]:
            if isinstance(self.evaluator, LLMJudgeEvaluator):
                best, scores = self.evaluator.judge(
                    prompt=prompt_text,
                    output_a=best,
                    output_b=candidate,
                    context=context,
                )
            else:
                scorable_a = ScorableFactory.from_dict(
                    {"id": "a", "text": best}, ScorableType.HYPOTHESIS
                )
                scorable_b = ScorableFactory.from_dict(
                    {"id": "b", "text": candidate}, ScorableType.HYPOTHESIS
                )

                score_a = (
                    self.evaluator.score(context, scorable_a, ["value"])
                    .results["value"]
                    .score
                )
                score_b = (
                    self.evaluator.score(context, scorable_b, ["value"])
                    .results["value"]
                    .score
                )
                scores = {"value_a": score_a, "value_b": score_b}
                best = best if score_a >= score_b else candidate

            # Report after each comparison
            self.report(
                {
                    "event": "pairwise_eval",
                    "scores": scores,
                    "best_snippet": best[:120],
                }
            )

        self.logger.log(
            "EvaluationCompleted", {"best_output": best[:100], **scores}
        )
        self.report(
            {
                "event": "evaluation_done",
                "final_choice_snippet": best[:200],
                "scores": scores,
            }
        )

        # 3. Save hypothesis
        confidence = max(scores.get("value_a", 0), scores.get("value_b", 0))
        features = {
            "prompt": prompt_text,
            "best_output": best,
            "candidates": candidates,
        }

        prompt = self.get_or_save_prompt(prompt_text, context)
        best_orm = self.save_hypothesis(
            {
                "text": best,
                "confidence": confidence,
                "features": features,
                "prompt_id": prompt.id,
            },
            context,
        )
        self.memory.hypotheses.insert(best_orm)

        self.report(
            {
                "event": "hypothesis_saved",
                "hypothesis_id": best_orm.id,
                "confidence": confidence,
            }
        )

        # 4. Classify patterns
        self.classify_and_store_patterns(
            hypothesis=best_orm.to_dict(),
            context=context,
            prompt_loader=self.prompt_loader,
            cfg=self.cfg,
            memory=self.memory,
            logger=self.logger,
            agent_name=self.name,
            score=confidence,
        )
        self.report(
            {"event": "patterns_classified", "hypothesis_id": best_orm.id}
        )

        context[self.output_key] = [best_orm.to_dict()]
        self.logger.log("AgentRunCompleted", {"output_key": self.output_key})

        # Final report entry
        self.report(
            {
                "event": "completed",
                "message": f"CoT generation completed. Best hypothesis {best_orm.id} saved.",
            }
        )

        return context

    def _init_evaluator(self, cfg):
        """Pick evaluator based on config"""
        strategy = cfg.get("evaluator", "mrq")

        if strategy == "llm":
            return LLMJudgeEvaluator(
                cfg,
                cfg.get("evaluator_model", cfg.get("model")),
                cfg.get("evaluator_prompt_file", "evaluation.txt"),
                self.call_llm,
                self.logger,
            )

        scorers = {
            "svm": SVMScorer,
            "mrq": MRQScorer,
            "sicql": SICQLScorer,
            "ebt": EBTScorer,
            "hrm": HRMScorer,
            "contrastive_ranker": ContrastiveRankerScorer,
        }
        scorer_cls = scorers.get(strategy, MRQScorer)
        return scorer_cls(cfg, memory=self.memory, logger=self.logger)

    @staticmethod
    def gen(slot: str, context: dict = None) -> str:
        """Generate slot-specific text using context LLM."""
        slot_prompts = {
            "step_1_understanding": "Explain the problem in your own words.",
            "step_2_concepts": "List key concepts and methods that could help.",
            "step_3_application": "Apply these concepts to solving the problem.",
            "step_4_verification": "Check if each step is logically correct.",
            "step_5_reflection": "Reflect: is the reasoning complete and creative?",
            "final_answer": "Give the final answer in a clear, concise form.",
        }
        instruction = slot_prompts.get(slot, f"Expand reasoning for {slot}.")
        llm = context.get("__llm") if context else None
        goal_text = (
            context.get("goal", {}).get("goal_text", "") if context else ""
        )

        query = f"Problem: {goal_text}\n\nTask: {instruction}"
        print("GeneratingPrompt", {"slot": slot, "prompt": query})

        return llm(query, context) if llm else f"[No LLM available for {slot}]"
