# stephanie/tools/frontier_lens_group_tool.py
from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

from stephanie.tools.frontier_lens_tool import FrontierLensTool

log = logging.getLogger(__name__)


class FrontierLensGroupTool:
    """
    Convenience wrapper to run FrontierLens over a batch of rows
    outside the ScorableProcessor (e.g., in Nexus / Critic agents).

    It:
      - calls FrontierLensTool.apply_rows()
      - attaches a shared frontier_lens_report + feature_names to each row
      - leaves features as a single global vector (not per-row).
    """

    name = "frontier_lens_group"

    def __init__(self, cfg: Dict, memory, container, logger):
        self.cfg = cfg or {}
        self.memory = memory
        self.container = container
        self.logger = logger
        self.vc = FrontierLensTool(cfg, memory, container, logger)

    async def apply(self, rows: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not rows:
            return rows

        episode_id = context.get("pipeline_run_id", "frontier_lens:default")

        out = self.vc.apply_rows(
            episode_id=episode_id,
            rows=rows,
            meta={"n_rows": len(rows)},
        )

        report = out.get("report")
        feat_names = out.get("feature_names") or []
        feats = out.get("features")

        if report is None:
            log.warning("[FrontierLensGroupTool] FrontierLens returned no report; leaving rows untouched.")
            return rows

        rep = report.to_dict() if hasattr(report, "to_dict") else (report or {})

        # Global feature vector (3M+3) – not per-row. Expose once as a list.
        if isinstance(feats, np.ndarray):
            global_feats = feats.astype(float).tolist()
        else:
            global_feats = None

        for r in rows:
            r.setdefault("frontier_lens_feature_names", feat_names)
            r.setdefault("frontier_lens_features", global_feats)
            r.setdefault("frontier_lens_report", rep)

        # out["vpm"] is cohort-level; callers can persist it via ZeroModel if desired
        return rows
