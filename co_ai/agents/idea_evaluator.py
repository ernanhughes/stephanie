# co_ai/agents/idea_evaluator.py
from co_ai.agents.base import BaseAgent
from co_ai.constants import GOAL
from co_ai.evaluator.llm_judge_evaluator import LLMJudgeEvaluator
from co_ai.evaluator.mrq_self_evaluator import MRQSelfEvaluator


class IdeaEvaluatorAgent(BaseAgent):
    """
    Evaluates research ideas and hypotheses using multiple strategies:

    - LLM-based pairwise comparison (like DPO)
    - Preference learning via MR.Q Self Evaluator
    """

    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.strategy = cfg.get("strategy", "llm")  # llm | mrq
        self.evaluator = self._init_evaluator()
        self.top_k = cfg.get("top_k", 5)

    async def run(self, context: dict) -> dict:
        hypotheses = self.get_hypotheses(context)
        goal = context.get(GOAL)
        baseline = context.get("baseline_hypotheses", {}).get("text")

        if not hypotheses:
            self.logger.log("NoHypothesesToEvaluate", {})
            context["scored_hypotheses"] = []
            return context

        scored_results = []
        for hyp in hypotheses:
            hyp_text = hyp["text"]
            preferred, scores = self.evaluator.judge(
                prompt=hyp_text,
                output_a=baseline or hyp_text,
                output_b=hyp_text,
                context=context,
            )
            scored_results.append(
                {
                    "text": hyp_text,
                    "preferred": preferred,
                    "scores": scores,
                    "source": "llm-judge",
                    "score": scores.get("score_b", 0),
                    "reasoning": scores.get("reason", ""),
                }
            )

        scored_results.sort(key=lambda x: x["score"], reverse=True)
        context["scored_hypotheses"] = scored_results
        context["top_hypothesis"] = scored_results[0]
        return context

    def get_top_k(self, context: dict, k: int = 5):
        return sorted(
            context.get("scored_hypotheses", []), key=lambda x: x["score"], reverse=True
        )[:k]

    def _init_evaluator(self):
        if self.cfg.get("evaluator", "llm") == "llm":
            llm_model = self.cfg.get("evaluator_model", self.cfg.get("model"))
            prompt_file = self.cfg.get("evaluator_prompt_file", "evaluator.txt")
            return LLMJudgeEvaluator(
                self.cfg,
                llm_cfg=llm_model,
                prompt_file=prompt_file,
                llm=self.call_llm,
                logger=self.logger,
            )
        else:
            return MRQSelfEvaluator(
                memory=self.memory,
                logger=self.logger,
                device=self.cfg.get("device", "cpu"),
            )
