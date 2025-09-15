# stephanie/agents/inference/et_hrm_inference.py
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from stephanie.agents.base_agent import BaseAgent
from stephanie.data.plan_trace import PlanTrace
from stephanie.data.score_bundle import ScoreBundle
from stephanie.scoring.scorable_factory import ScorableFactory
from stephanie.scoring.scorer.ep_hrm_scorer import EpistemicPlanHRMScorer
from stephanie.scoring.scoring_manager import ScoringManager
from stephanie.utils.trace_utils import load_plan_traces_from_export_dir


class EpistemicTraceHRMInferenceAgent(BaseAgent):
    """
    Uses the EpistemicPlanHRMScorer to score reasoning traces.
    Can load traces from context or from export directory if missing.
    Stores score results in memory and context.
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.dimensions = cfg.get("dimensions", [])
        self.export_dir = cfg.get("export_dir", "reports/epistemic_plan_executor")

        self.scorer = EpistemicPlanHRMScorer(cfg.get("hrm", {}), memory=memory, container=container, logger=logger)

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
        goal_text = context.get("goal", {}).get("goal_text", "")
        for trace in traces:
            score_bundle: ScoreBundle = self.scorer.score(trace, self.dimensions)

            scorable = ScorableFactory.from_plan_trace(trace, goal_text=goal_text)
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
                evaluator_name=self.scorer.name,
            )

            results.append({
                "trace_id": trace.trace_id,
                "scores": score_bundle.to_dict()
            })

        context[self.output_key] = results
        return context

