# stephanie/agents/cot_generator.py
from stephanie.agents.base_agent import BaseAgent
from stephanie.analysis.rubric_classifier import RubricClassifierMixin
from stephanie.constants import GOAL
from stephanie.evaluator.llm_judge_evaluator import LLMJudgeEvaluator
from stephanie.scoring.contrastive_ranker_scorer import ContrastiveRankerScorer
from stephanie.scoring.ebt_scorer import EBTScorer
from stephanie.scoring.hrm_scorer import HRMScorer
from stephanie.scoring.mrq_scorer import MRQScorer
from stephanie.scoring.scorable_factory import ScorableFactory, TargetType
from stephanie.scoring.sicql_scorer import SICQLScorer
from stephanie.scoring.svm_scorer import SVMScorer


class ChainOfThoughtGeneratorAgent(BaseAgent, RubricClassifierMixin):
    """
    Generates chain-of-thought candidates and evaluates them
    using a configurable evaluator/scorer.
    """

    def __init__(self, cfg, memory, logger):
        super().__init__(cfg, memory, logger)
        self.logger.log("AgentInit", {"agent": "ChainOfThoughtGeneratorAgent"})
        self.num_candidates = cfg.get("num_candidates", 2)
        self.evaluator = self._init_evaluator(cfg)
        

    async def run(self, context: dict):
        goal = context.get(GOAL)
        # __ will flag as non serializable
        context["__llm"] = self.call_llm
        self.logger.log("AgentRunStarted", {"goal": goal})

        # 1. Generate candidates
        prompt_text = self.prompt_loader.load_prompt(self.cfg, context)
        candidates = [
            self.call_llm(prompt_text, context) for _ in range(self.num_candidates)
        ]
        self.logger.log("GenerationCompleted", {"candidates": [c[:100] for c in candidates]})

        # 2. Evaluate candidates pairwise
        best = candidates[0]
        scores = {}

        for candidate in candidates[1:]:
            if isinstance(self.evaluator, LLMJudgeEvaluator):
                # LLM judge handles prompts directly
                best, scores = self.evaluator.judge(
                    prompt=prompt_text, output_a=best, output_b=candidate, context=context
                )
            else:
                # All scorers follow standard Scorable interface
                scorable_a = ScorableFactory.from_dict(
                    {"id": "a", "text": best}, TargetType.HYPOTHESIS
                )
                scorable_b = ScorableFactory.from_dict(
                    {"id": "b", "text": candidate}, TargetType.HYPOTHESIS
                )

                score_a = self.evaluator.score(context, scorable_a, ["value"]).results["value"].score
                score_b = self.evaluator.score(context, scorable_b, ["value"]).results["value"].score

                scores = {"value_a": score_a, "value_b": score_b}
                best = best if score_a >= score_b else candidate

        self.logger.log("EvaluationCompleted", {"best_output": best[:100], **scores})

        # 3. Save hypothesis
        confidence = max(scores.get("value_a", 0), scores.get("value_b", 0))
        features = {"prompt": prompt_text, "best_output": best, "candidates": candidates}

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

        context[self.output_key] = [best_orm.to_dict()]
        self.logger.log("AgentRunCompleted", {"output_key": self.output_key})
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
        # Compose a slot-specific instruction
        slot_prompts = {
            "step_1_understanding": "Explain the problem in your own words.",
            "step_2_concepts": "List key concepts and methods that could help.",
            "step_3_application": "Apply these concepts to solving the problem.",
            "step_4_verification": "Check if each step is logically correct.",
            "step_5_reflection": "Reflect: is the reasoning complete and creative?",
            "final_answer": "Give the final answer in a clear, concise form."
        }
        instruction = slot_prompts.get(slot, f"Expand reasoning for {slot}.")
        llm = context.get("__llm")
        goal_text = context.get("goal", {}).get("goal_text", "")
        query = f"Problem: {goal_text}\n\nTask: {instruction}"
        print("GeneratingPrompt", {"slot": slot, "prompt": query})
        response = llm(query, context)
        return response
