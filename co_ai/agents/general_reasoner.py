from itertools import combinations
from typing import Optional

from co_ai.agents.base import BaseAgent
from co_ai.analysis.rubric_classifier import RubricClassifierMixin
from co_ai.constants import GOAL, GOAL_TEXT, PIPELINE, PIPELINE_RUN_ID
from co_ai.evaluator import LLMJudgeEvaluator, MRQSelfEvaluator
from co_ai.models import HypothesisORM, ScoreORM
from co_ai.prompts import PromptLoader


class GeneralReasonerAgent(BaseAgent, RubricClassifierMixin):
    def __init__(self, cfg, memory, logger):
        super().__init__(cfg, memory, logger)
        self.logger.log("AgentInit", {"agent": "GeneralReasonerAgent"})
        self.judge = self._init_judge()
        self.prompt_loader = PromptLoader(self.cfg, self.logger)

    async def run(self, context: dict) -> dict:
        goal = context.get(GOAL)

        self.logger.log("AgentRunStarted", {"goal": goal})

        # Generate hypotheses (if needed)
        if self.cfg.get("thinking_mode") == "generate_and_judge":
            hypotheses = self.generate_hypotheses(context)
        else:
            hypotheses = self.get_hypotheses(context)

        context["hypotheses"] = hypotheses
        win_counts = {h.get("id"): 0 for h in hypotheses}
        evaluations = []

        prompt_loader = PromptLoader(self.memory, logger=self.logger)
        judging_prompt_template = self.cfg.get(
            "evaluator_prompt_file", "judge_pairwise_comparison.txt"
        )

        context["scoring"] = []

        # Evaluate all hypothesis pairs
        for hyp_a, hyp_b in combinations(hypotheses, 2):
            judge_context = {
                "goal": goal,
                "hypothesis_a": hyp_a.get("text"),
                "hypothesis_b": hyp_b.get("text")
            }

            prompt_text = prompt_loader.from_file(
                judging_prompt_template, self.cfg, judge_context
            )

            preferred, score = self.judge.judge(prompt_text, goal, hyp_a.get("text"), hyp_b.get("text"))

            # Save scores
            goal_id = self.get_goal_id(goal)
            judge_name = self.cfg.get("judge", "mrq")

            s_a = self._create_score_orm(
                goal_id=goal_id,
                hypothesis=hyp_a,
                preferred="A",
                score_data=score,
                model_name=self.model_name,
                judge_name=judge_name,
                pipeline_run_id=context.get(PIPELINE_RUN_ID),
            )
            s_b = self._create_score_orm(
                goal_id=goal_id,
                hypothesis=hyp_b,
                preferred="B",
                score_data=score,
                model_name=self.model_name,
                judge_name=judge_name,
                pipeline_run_id=context.get(PIPELINE_RUN_ID),
            )

            self.memory.scores.insert(s_a)
            self.memory.scores.insert(s_b)

            evaluations.append(score)
            winner_id = hyp_a.get("id") if score["winner"] == "A" else hyp_b.get("id")
            win_counts[winner_id] += 1

            self.logger.log("GeneralReasoningJudgement", {
                "event": "JudgedPair",
                "goal": goal,
                "hypothesis_a": hyp_a.get("text")[:100],
                "hypothesis_b": hyp_b.get("text")[:100],
                "winner": score["winner"],
                "score_a": score["score_a"],
                "score_b": score["score_b"],
                "reason": score["reason"],
                "evaluator": judge_name
            })

            context["scoring"].append({
                "hypothesis_a": hyp_a.get("text"),
                "hypothesis_b": hyp_b.get("text"),
                "winner": score["winner"],
                "score_a": score["score_a"],
                "score_b": score["score_b"],
                "reason": score["reason"],
                "preferred": preferred,
                "evaluator": judge_name
            })

        # Select best hypothesis by win count
        best_id = max(win_counts, key=win_counts.get)
        best_hypothesis = next(h for h in hypotheses if h.get("id") == best_id)
        best_hypothesis["evaluation"] = {
            "wins": win_counts[best_id],
            "judged_pairs": len(evaluations),
        }

        # Classify with rubrics and store pattern stats
        pattern = self.classify_with_rubrics(
            hypothesis=best_hypothesis,
            context=context,
            prompt_loader=self.prompt_loader,
            cfg=self.cfg,
            logger=self.logger
        )

        summarized = self._summarize_pattern(pattern)
        context["pattern"] = summarized

        goal_id, hypothesis_id, pattern_stats = self.generate_pattern_stats(
            goal=goal,
            hypothesis=best_hypothesis,
            pattern_dict=summarized,
            memory=self.memory,
            cfg=self.cfg,
            agent_name=self.name,
            confidence_score=best_hypothesis.get("confidence")
        )

        self.memory.pattern_stats.insert(pattern_stats)
        context["pattern_stats"] = summarized

        context[self.output_key] = best_hypothesis
        return context

    def _create_score_orm(self, *, goal_id: int, hypothesis: HypothesisORM,
                         preferred: str, score_data: dict, model_name: str,
                         judge_name: str, pipeline_run_id: Optional[int] = None) -> ScoreORM:
        """
        Creates a ScoreORM object from hypothesis and score data.
        """
        score_type = "pairwise_comparison"
        print(f"Score data: {score_data}")
        value = score_data[f"score_{preferred.lower()}"]

        score_obj = ScoreORM(
            goal_id=goal_id,
            hypothesis_id=hypothesis.get("id"),
            agent_name=self.name,
            model_name=model_name,
            evaluator_name=judge_name,
            score_type=score_type,
            pipeline_run_id=pipeline_run_id,
            reasoning_strategy=hypothesis.get("strategy"),
        )

        if isinstance(value, (float, int)):
            score_obj.score = float(value)
        elif isinstance(value, str):
            score_obj.score_text = value
        else:
            raise ValueError(f"Unexpected score type: {type(value)}")

        return score_obj

    def generate_hypotheses(self, context: dict) -> list[dict]:
        """Generates multiple hypotheses using different strategies"""
        goal = context.get(GOAL)
        question = goal.get(GOAL_TEXT)

        strategies = self.cfg.get("generation_strategy_list", ["cot"])
        merged = {**context, "question": question}

        hypotheses = []
        goal_id = self.get_goal_id(goal)
        for strategy in strategies:
            prompt = self.prompt_loader.from_file(
                f"strategy_{strategy}.txt", self.cfg, merged
            )
            response = self.call_llm(prompt, merged)
            hypothesis = HypothesisORM(
                text=response,
                goal_id=goal_id,
                strategy=strategy,
                features={"strategy": strategy},
                source=self.name,
                pipeline_signature=context.get(PIPELINE),
            )
            self.memory.hypotheses.insert(hypothesis)
            hypotheses.append(hypothesis.to_dict())

        return hypotheses

    def _init_judge(self):
        judge_strategy = self.cfg.get("judge", "mrq")
        if judge_strategy == "llm":
            llm = self.cfg.get("judge_model", self.cfg.get("model"))
            prompt_file = self.cfg.get("judge_prompt_file", "judge_pairwise_comparison.txt")
            self.logger.log("EvaluatorInit", {"strategy": "LLM", "prompt_file": prompt_file})
            return LLMJudgeEvaluator(self.cfg, llm, prompt_file, self.call_llm, self.logger)
        else:
            self.logger.log("EvaluatorInit", {"strategy": "MRQ"})
            return MRQSelfEvaluator(self.memory, self.logger)

    def _summarize_pattern(self, pattern: dict):
        stats = {}
        for dimension, label in pattern.items():
            stats[label] = stats.get(label, 0) + 1
        return stats
