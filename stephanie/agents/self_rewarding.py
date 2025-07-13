# stephanie/agents/self_rewarding.py

from dataclasses import dataclass
from typing import Dict, List, Optional

from stephanie.agents.base_agent import BaseAgent
from stephanie.constants import PIPELINE_RUN_ID
from stephanie.evaluator.base import BaseEvaluator
from stephanie.memory import SymbolicRuleStore
from stephanie.models import EvaluationORM, ScoreORM
from stephanie.prompts import PromptLoader


@dataclass
class SelfRewardingConfig:
    inner_agent: str  # e.g., "ChainOfThoughtAgent"
    scorer_agent: str = "MRQScoringAgent"
    min_score_threshold: float = 5.0
    use_symbolic_filter: bool = True
    log_to_db: bool = True
    update_rules: bool = False
    rule_store: Optional[SymbolicRuleStore] = None


class SelfRewardingAgent(BaseAgent):
    """
    A self-evaluating agent that wraps another agent,
    scores its own outputs using structured evaluators,
    and optionally updates symbolic rules or training data.
    """

    def __init__(self, cfg: SelfRewardingConfig, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.cfg = cfg
        self.prompt_loader = PromptLoader(cfg.get("prompt_dir", "prompts"))
        self.scorer = self._init_scorer()
        self.inner_agent = self._init_inner_agent()

    def _init_scorer(self) -> BaseEvaluator:
        scorer_type = self.cfg.scorer_agent
        if scorer_type == "MRQScoringAgent":
            from stephanie.evaluator import MRQSelfEvaluator

            return MRQSelfEvaluator(
                cfg=self.cfg, memory=self.memory, logger=self.logger
            )
        elif scorer_type == "LLMJudgeEvaluator":
            from stephanie.evaluator import LLMJudgeEvaluator

            return LLMJudgeEvaluator(
                cfg=self.cfg, memory=self.memory, logger=self.logger
            )
        else:
            raise ValueError(f"Unsupported scorer type: {scorer_type}")

    def _init_inner_agent(self):
        agent_class = self.cfg.inner_agent
        try:
            module_name, class_name = agent_class.rsplit(".", 1)
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            return cls(cfg=self.cfg, memory=self.memory, logger=self.logger)
        except Exception as e:
            raise ImportError(f"Could not load agent {agent_class}: {e}")

    async def run(self, context: dict) -> dict:
        """Run inner agent, evaluate result, and store score."""
        goal = context.get("goal")
        self.logger.log("SelfRewardingStart", {"goal": goal})

        # Step 1: Run inner agent to generate hypothesis
        hypothesis = await self.inner_agent.run(context)

        # Step 2: Evaluate hypothesis
        scores = self.scorer.evaluate(hypothesis, context=context)

        # Step 3: Store evaluation in DB
        if self.cfg.log_to_db:
            eval_id = self._log_evaluation(hypothesis, scores, context)

        # Step 4: Rule tuning (optional)
        if self.cfg.update_rules and self.cfg.rule_store:
            self.cfg.rule_store.update_rules_from_scores(scores)

        # Step 5: Decide which hypothesis to return
        best_hypothesis = self._select_best_hypothesis([hypothesis], scores)

        context["best_hypothesis"] = best_hypothesis
        context["scores"] = scores
        self.logger.log("SelfRewardingDone", {"score": scores.get("total", 0)})
        return context

    def _log_evaluation(self, hypothesis, scores, context: dict) -> int:
        """Log hypothesis and scores to database"""
        evaluation = EvaluationORM(
            goal_id=hypothesis.get("goal_id"),
            target_type="hypothesis",
            hypothesis_id=hypothesis.get("id"),
            agent_name=self.name,
            model_name=self.model_name,
            evaluator_name=self.name,
            pipeline_run_id=context.get(PIPELINE_RUN_ID),
        )
        self.memory.session.add(evaluation)
        self.memory.session.commit()

        for dim, score in scores.items():
            score_orm = ScoreORM(
                evaluation_id=evaluation.id,
                dimension=dim,
                score=score,
                source="self_rewarding",
            )
            self.memory.session.add(score_orm)
        self.memory.session.commit()
        return evaluation.id

    def _select_best_hypothesis(
        self, hypotheses: List[dict], scores: dict[str, float]
    ) -> Dict:
        """Select hypothesis with highest composite score"""
        # For simplicity, just return the one scored above
        return hypotheses[0]
