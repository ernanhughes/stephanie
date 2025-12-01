# stephanie/scoring/metrics/feature/frontier_lens_group_feature.py
from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

from stephanie.scoring.metrics.feature.base_group_feature import \
    BaseGroupFeature
from stephanie.tools.frontier_lens_tool import FrontierLensTool

log = logging.getLogger(__name__)


class FrontierLensGroupFeature(BaseGroupFeature):
    """
    Batch FrontierLens feature.

    - Consumes rows with metrics_columns / metrics_values
    - Runs FrontierLensTool.apply_rows() once per batch
    - Attaches a shared 'frontier_lens_report' and (optionally) the
      global FrontierLens feature vector to each row.
    """

    name = "frontier_lens_group"
    requires = ["metric_filter"]  # keep the dependency ordering

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.tool = FrontierLensTool(cfg, memory, container, logger)
        self.enabled = bool(self.cfg.get("enabled", True))
        self.episode_id = self.cfg.get("episode_id", "frontier_lens:default")

        # Optional: attach the same global features to each row
        self.store_per_row_features = bool(self.cfg.get("store_per_row_features", False))

        # debug/telemetry fields so .report() never crashes
        self._quality: float | None = None
        self._kept_cols: int = 0
        self._rows_in: int = 0
        self._rows_used: int = 0
        self._error: str | None = None

    async def apply(self, rows: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self.enabled:
            return rows

        self._error = None
        self._rows_in = len(rows)

        if not rows:
            return rows

        # Use pipeline_run_id when available, fall back to static
        episode_id = context.get("pipeline_run_id") or self.episode_id

        try:
            out = self.tool.apply_rows(
                episode_id=episode_id,
                rows=rows,
                meta={"n_rows": len(rows)},
            )
        except Exception as e:
            # Fail-closed: keep rows untouched and record why
            self._quality = None
            self._kept_cols = 0
            self._rows_used = 0
            self._error = f"{type(e).__name__}: {e}"
            log.warning("[FrontierLensGroupFeature] skipped: %s", self._error)
            return rows

        report = out.get("report")
        vpm = out.get("vpm")
        feats = out.get("features")
        names = out.get("feature_names") or []

        q = out.get("quality")
        self._quality = float(q) if isinstance(q, (int, float)) else None
        self._kept_cols = len(names)
        self._rows_used = int(vpm.shape[0]) if isinstance(vpm, np.ndarray) else 0

        # attach per-row artifacts (lightweight)
        rep_dict = report.to_dict() if hasattr(report, "to_dict") else (report or {})

        # If we decide to expose features, it’s one global vector (3M+3),
        # not per-row. Attach as-is or leave None.
        if isinstance(feats, np.ndarray):
            global_feats = feats.astype(float).tolist()
        else:
            global_feats = None

        for r in rows:
            r.setdefault("frontier_lens_report", rep_dict)
            r.setdefault("frontier_lens_feature_names", names)
            if self.store_per_row_features:
                r.setdefault("frontier_lens_features", global_feats)
            else:
                r.setdefault("frontier_lens_features", None)

        return rows

    def report(self) -> Dict[str, Any]:
        ok = (self._error is None) and (self._quality is None or self._quality >= 0.5)
        return {
            "feature": self.name,
            "ok": ok,
            "quality": self._quality,
            "rows_in": self._rows_in,
            "rows_used": self._rows_used,
            "kept_metric_cols": self._kept_cols,
            "error": self._error,
            "summary": (
                f"rows={self._rows_used}/{self._rows_in}; "
                f"kept_cols={self._kept_cols}; quality={self._quality}"
            ),
        }
