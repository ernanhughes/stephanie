# stephanie/scoring/metrics/visicalc_group_feature.py
from __future__ import annotations
import logging
from typing import Any, Dict, List
import numpy as np

from stephanie.scoring.metrics.base_group_feature import BaseGroupFeature
from stephanie.tools.visicalc_tool import VisiCalcTool

log = logging.getLogger(__name__)

class VisiCalcGroupFeature(BaseGroupFeature):
    name = "visicalc_group"
    requires = ["metric_filter"]  # keep the dependency ordering

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.tool = VisiCalcTool(cfg, memory, container, logger)
        self.enabled = bool(self.cfg.get("enabled", True))
        self.episode_id = self.cfg.get("episode_id", "visicalc:default")
        # init debug/telemetry fields so .report() never crashes
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
        try:
            out = self.tool.apply_rows(
                episode_id=self.episode_id,
                rows=rows,
                meta={"n_rows": len(rows)},
            )
        except Exception as e:
            # Fail-closed: keep rows untouched and record why
            self._quality = None
            self._kept_cols = 0
            self._rows_used = 0
            self._error = f"{type(e).__name__}: {e}"
            log.warning("[VisiCalcGroupFeature] skipped: %s", self._error)
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
        for r in rows:
            r.setdefault("visicalc_report", rep_dict)
            r.setdefault("visicalc_features", None)
            r.setdefault("visicalc_feature_names", names)

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