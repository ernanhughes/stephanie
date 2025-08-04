# stephanie/agents/knowledge/epistemic_trace_reward_scorer.py

import json
import os
from typing import Any, Dict, List

from stephanie.agents.base_agent import BaseAgent
from stephanie.data.plan_trace import PlanTrace
from stephanie.scoring.ep_hrm_scorer import EpistemicPlanHRMScorer
from stephanie.scoring.scorable_factory import ScorableFactory
from stephanie.data.score_bundle import ScoreBundle
from stephanie.scoring.scoring_manager import ScoringManager
from stephanie.utils.trace_utils import load_plan_traces_from_export_dir


class EpistemicTraceHRMInferenceAgent(BaseAgent):
    """
    Uses the EpistemicPlanHRMScorer to score reasoning traces.
    Can load traces from context or from export directory if missing.
    Stores score results in memory and context.
    """

    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.dimensions = cfg.get("dimensions", [])
        self.export_dir = cfg.get("export_dir", "reports/epistemic_plan_executor")

        self.scorer = EpistemicPlanHRMScorer(cfg.get("hrm", {}), memory=memory, logger=logger)

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.log("EpistemicTraceRewardScoringStarted", {
            "dimensions": self.dimensions
        })

        # --- 1. Load traces from context or disk ---
        raw_traces_data = context.get("plan_traces", [])
        if not raw_traces_data:
            self.logger.log("NoTracesFoundInContext", {
                "message": "No traces in context; loading from disk.",
                "path": self.export_dir
            })
            traces = load_plan_traces_from_export_dir(self.export_dir)
        else:
            traces = [PlanTrace.from_dict(t) for t in raw_traces_data]

        if not traces:
            self.logger.log("EpistemicTraceRewardScorerNoData", {
                "message": "No traces found to score."
            })
            return context

        results = []
        for trace in traces:
            score_bundle: ScoreBundle = self.scorer.score(trace, self.dimensions)

            scorable = ScorableFactory.from_plan_trace(trace, mode="default")
            # Save to memory
            ScoringManager.save_score_to_memory(
                bundle=score_bundle,
                scorable=scorable,
                context=context,
                cfg=self.cfg,
                memory=self.memory,
                logger=self.logger,
                source=self.scorer.model_type,
                model_name=self.scorer.get_model_name(),
            )

            results.append({
                "trace_id": trace.trace_id,
                "scores": score_bundle.to_dict()
            })

        context[self.output_key] = results
        return context

