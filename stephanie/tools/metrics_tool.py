# stephanie/tools/metrics_tool.py
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Tuple

from stephanie.scoring.metrics.metric_observer import MetricObserver
from stephanie.scoring.scorable import Scorable

log = logging.getLogger(__name__)


class MetricsTool:
    """
    Tool responsible for:
      - Running multiple scorers (SICQL, HRM, TinyCritic, SVM...)
      - Flattening bundles
      - Producing canonical metrics_vector / columns / values
      - Observing metric vectors
    """

    def __init__(self, cfg, memory, container, logger):
        self.cfg = cfg or {}
        self.memory = memory
        self.container = container
        self.logger = logger

        # Scoring engines + settings
        self.scorers: List[str] = self.cfg.get("scorers", ["sicql", "hrm", "tiny", "svm"])
        self.dimensions: List[str] = self.cfg.get(
            "dimensions",
            ["coverage", "reasoning", "knowledge", "clarity", "faithfulness"],
        )
        self.persist: bool = bool(self.cfg.get("persist_scores", False))
        self.attach_scores: bool = bool(self.cfg.get("attach_scores", True))

        # Services
        self.scoring = container.get("scoring")
        metric_observer_cfg = self.cfg.get("metric_observer", {})
        self.metric_observer = MetricObserver(
            enabled=bool(metric_observer_cfg.get("enabled", True)),
            snapshot_path=metric_observer_cfg.get("snapshot_path", "runs/metrics/metric_universe.json")
        )
 
    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------
    async def apply(self, scorable, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return a dict with:
          metrics_vector: Dict[str, float]
          metrics_columns: List[str]
          metrics_values: List[float]
        """
        if not self.attach_scores or not self.scoring:
            return {
                "metrics_vector": {},
                "metrics_columns": [],
                "metrics_values": [],
            }

        goal_text = Scorable.get_goal_text(scorable, context=context)
        run_id = context.get("pipeline_run_id")
        ctx = {"goal": {"goal_text": goal_text}, "pipeline_run_id": run_id}

        vector: Dict[str, float] = {}
        for scorer_name in self.scorers:
            bundle = (
                self.scoring.score_and_persist
                if self.persist
                else self.scoring.score
            )(
                scorer_name=scorer_name,
                scorable=scorable,
                context=ctx,
                dimensions=self.dimensions,
            )

            alias = self.scoring.get_model_name(scorer_name)
            agg = float(bundle.aggregate())

            flat = bundle.flatten(
                include_scores=True,
                include_attributes=True,
                numeric_only=True,
            )

            for k, v in flat.items():
                vector[f"{alias}.{k}"] = float(v)

            vector[f"{alias}.aggregate"] = agg

            await asyncio.sleep(0)  # maintain async fairness

        # ordering
        metrics_columns = sorted(vector.keys())
        metrics_values = [vector[col] for col in metrics_columns]

        # observer hook
        self._observe(scorable, context, metrics_columns, metrics_values)

        return {
            "metrics_vector": vector,
            "metrics_columns": metrics_columns,
            "metrics_values": metrics_values,
        }

    # --------------------------------------------------------
    def _observe(self, scorable, context, cols, vals):
        """Forward to MetricObserver, if enabled."""
        if not self.metric_observer:
            return
        run_id = context.get("run_id", "unknown_run")
        cohort = context.get("cohort", "default")
        is_correct = getattr(scorable, "meta", {}).get("is_correct", None)

        metric_dict = dict(zip(cols, vals))

        self.metric_observer.observe(
            metrics=metric_dict,
            run_id=run_id,
            cohort=cohort,
            is_correct=is_correct,
        )
