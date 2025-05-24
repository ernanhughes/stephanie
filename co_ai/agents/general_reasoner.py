from dataclasses import asdict
from itertools import combinations

from co_ai.agents.base import BaseAgent
from co_ai.analysis.rubric_classifier import RubricClassifierMixin
from co_ai.constants import GOAL, PIPELINE
from co_ai.evaluator import LLMJudgeEvaluator, MRQSelfEvaluator
from co_ai.models import Hypothesis, Score
from co_ai.models.pattern_stat import generate_pattern_stats
from co_ai.prompts import PromptLoader


class GeneralReasonerAgent(BaseAgent, RubricClassifierMixin):
    def __init__(self, cfg, memory, logger):
        super().__init__(cfg, memory, logger)
        self.logger.log("AgentInit", {"agent": "GeneralReasonerAgent"})
        self.judge = self._init_judge()
        self.prompt_loader = PromptLoader(self.cfg, self.logger)

    async def run(self, context: dict):
        goal_struct = context.get(GOAL)
        goal = self.extract_goal_text(goal_struct)
        self.logger.log("AgentRunStarted", {"goal": goal})

        if self.cfg.get("thinking_mode") == "generate_and_judge":
            hypotheses = self.generate_hypotheses(goal, context)
        else:
            hypotheses = self.get_hypotheses(context)

        context["hypotheses"] = [asdict(h) for h in hypotheses]

        win_counts = {h.id: 0 for h in hypotheses}
        evaluations = []

        prompt_loader = PromptLoader(None, self.logger)
        judging_prompt_template = self.cfg.get("evaluator_prompt_file", "judge_pairwise_comparison.txt")

        context["scoring"] = []
        for hyp_a, hyp_b in combinations(hypotheses, 2):
            judge_context = {
                "goal": goal,
                "hypothesis_a": hyp_a.text,
                "hypothesis_b": hyp_b.text,
            }
            prompt_text = prompt_loader.from_file(judging_prompt_template, self.cfg, judge_context)

            preferred, score = self.judge.judge(prompt_text, goal, hyp_a.text, hyp_b.text)



            s_a = Score.build(goal, hyp_a.text, self.cfg)
            s_a.set_score(score["score_a"])
            s_a.reasoning_strategy = hyp_a.strategy_used
            self.memory.scores.insert(s_a)

            s_b = Score.build(goal, hyp_b.text, self.cfg)
            s_b.set_score(score["score_b"])
            s_b.reasoning_strategy = hyp_b.strategy_used
            self.memory.scores.insert(s_b)

            evaluations.append(score)
            winner_id = hyp_a.id if score["winner"] == "A" else hyp_b.id
            win_counts[winner_id] += 1

            self.logger.log("GeneralReasoningJudgement", {
                "event": "JudgedPair",
                "goal": goal,
                "hypothesis_a": hyp_a.text[:100],
                "hypothesis_b": hyp_b.text[:100],
                "winner": score["winner"],
                "score_a": score["score_a"],
                "score_b": score["score_b"],
                "reason": score["reason"],
                "evaluator": self.cfg.get("judge", "llm"),
            })
            context["scoring"].append({
                "hypothesis_a": hyp_a.text,
                "hypothesis_b": hyp_b.text,
                "winner": score["winner"],
                "score_a": score["score_a"],
                "score_b": score["score_b"],
                "reason": score["reason"],
                "preferred": preferred,
                "evaluator": self.cfg.get("judge", "llm"),
            })

        best_id = max(win_counts, key=win_counts.get)
        best_hypothesis = next(h for h in hypotheses if h.id == best_id)
        best_hypothesis.evaluation = {
            "wins": win_counts[best_id],
            "judged_pairs": len(hypotheses) - 1,
        }

        # Add rubric classification and pattern stat storage
        pattern = self.classify_with_rubrics(
            hypothesis=best_hypothesis,
            context=context,
            prompt_loader=self.prompt_loader,
            cfg=self.cfg,
            logger=self.logger
        )

        summarized = self._summarize_pattern(pattern)
        context["pattern"] = summarized

        goal_id, hypothesis_id, pattern_stats = generate_pattern_stats(
            self.extract_goal_text(context.get(GOAL)), best_hypothesis.text, summarized, self.memory, self.cfg, self.name, win_counts[best_id]
        )
        self.memory.hypotheses.store_pattern_stats(goal_id, hypothesis_id, pattern_stats)
        context["pattern_stats"] = summarized

        context[self.output_key] = asdict(best_hypothesis)
        return context

    def generate_hypotheses(self, question, context):
        strategies = self.cfg.get("generation_strategy_list", ["cot"])
        merged = {**context, **{"question": question}}
        hypotheses = []
        for strategy in strategies:
            prompt = self.prompt_loader.from_file(f"strategy_{strategy}.txt", self.cfg, merged)
            response = self.call_llm(prompt, merged)
            hypothesis = Hypothesis(
                text=response,
                goal=self.extract_goal_text(context.get(GOAL)),
                goal_type = "",
                strategy_used=strategy,
                features={"strategy": strategy},
                source=self.name,
                pipeline_signature=context.get(PIPELINE)
            )
            self.memory.hypotheses.insert(hypothesis)
            hypotheses.append(hypothesis)
        return hypotheses

    def _init_judge(self):
        if self.cfg.get("judge", "mrq") == "llm":
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
            if label not in stats:
                stats[label] = 0
            stats[label] += 1
        return stats
