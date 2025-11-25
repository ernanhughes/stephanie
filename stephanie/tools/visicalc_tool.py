# stephanie/tools/visicalc_tool.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence, Optional

import numpy as np

from stephanie.scoring.metrics.metric_mapping import MetricMapper
from stephanie.scoring.metrics.visicalc import VisiCalc
from stephanie.tools.base_tool import BaseTool

log = logging.getLogger(__name__)


class VisiCalcTool(BaseTool):
    """
    Row-compatible VisiCalc wrapper.

    Modes:
      - apply_row()    → per-row feature enrichment (light)
      - apply_batch()  → real VisiCalc analysis (heavy)
    """

    name = "visicalc"

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)

        # config
        self.frontier_metric     = cfg.get("frontier_metric", "HRM.aggregate")
        self.row_region_splits   = cfg.get("row_region_splits", 3)
        self.frontier_low        = cfg.get("frontier_low", 0.25)
        self.frontier_high       = cfg.get("frontier_high", 0.75)
        self.per_metric_normalize = cfg.get("per_metric_normalize", True)

        # optional ordering
        self.visicalc_metric_keys = cfg.get("metric_keys", [])

        self.mapper = MetricMapper.from_config(cfg)

        # output settings
        self.persist_png   = cfg.get("vpm_png", {}).get("enabled", False)
        self.png_mode      = cfg.get("vpm_png", {}).get("mode", "L")
        self.png_target    = cfg.get("vpm_png", {}).get("target_file")
        self.png_baseline  = cfg.get("vpm_png", {}).get("baseline_file")

    # =====================================================================
    # 1) PER-ROW API  (called by ScorableProcessor Feature)
    # =====================================================================
    async def apply(self, scorable, acc: Dict[str, Any], context: Dict[str, Any]):
        """
        Lightweight per-row enrichment:
            • attach visicalc_report (if metrics exist)
            • optionally save tiny per-row VPM preview
        """
        cols = acc.get("metrics_columns") or []
        vals = acc.get("metrics_values") or []

        if not cols or not vals:
            log.debug("[VisiCalcTool] no metrics, skipping row-level visi")
            return acc

        # Build mapping for this row
        metric_map = dict(zip(cols, vals))

        # Simple summary: frontier score, region tags, etc.
        # (We can expand this later)
        report = {
            "frontier_metric": self.frontier_metric,
            "frontier_value": metric_map.get(self.frontier_metric),
            "num_metrics": len(cols),
        }

        acc["visicalc_report"] = report
        return acc

    # =====================================================================
    # 2) BATCH API  (used by critic/nexus)
    # =====================================================================
    def apply_rows(
        self,
        *,
        episode_id: str,
        rows: List[Dict[str, Any]],
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Full VisiCalc batch analysis.
        Produces:
            - report
            - features
            - feature_names
            - vpm (uint8)
            - quality
        """
        vpm, metric_names, item_ids = self._matrix_for_rows(rows)
        if vpm.size == 0 or vpm.ndim != 2 or vpm.shape[1] == 0:
            log.warning("VisiCalcTool.apply_rows: empty matrix after selection; skipping VisiCalc.")
            return {
                "report": None,
                "features": np.empty((len(rows), 0), dtype=float),
                "feature_names": [],
                "vpm": np.empty((0, 0), dtype=np.uint8),
                "quality": None,  # ← was {}. Make it None so callers can treat “no signal” cleanly
            }

        vc = VisiCalc.from_matrix(
            episode_id      = episode_id,
            scores          = vpm,
            metric_names    = metric_names,
            item_ids        = item_ids,
            frontier_metric = self.frontier_metric,
            row_region_splits = self.row_region_splits,
            frontier_low    = self.frontier_low,
            frontier_high   = self.frontier_high,
            meta            = meta,
        )

        vpm_uint8 = vc.to_vpm_array(per_metric_normalize=self.per_metric_normalize)

        return {
            "report": vc.report,
            "features": vc.features,
            "feature_names": vc.feature_names,
            "vpm": vpm_uint8,
            "quality": vc.quality(),
        }

    # =====================================================================
    # Internal helper
    # =====================================================================

    def _matrix_for_rows(
        self,
        rows: List[Dict[str, Any]],
    ) -> tuple[np.ndarray, List[str], List[str]]:
        if not rows:
            raise ValueError("VisiCalcTool: no rows given")

        # UNION of all metric columns across rows
        union_cols, seen = [], set()
        non_empty = 0
        for r in rows:
            cols = r.get("metrics_columns") or []
            if cols:
                non_empty += 1
            for c in cols:
                if c not in seen:
                    seen.add(c)
                    union_cols.append(c)

        if not union_cols:
            raise ValueError("VisiCalcTool: no metric columns found on any row")

        metric_names = self.mapper.select_columns(union_cols) or union_cols

        if self.visicalc_metric_keys:
            preferred = [k for k in self.visicalc_metric_keys if k in metric_names]
            rest = [k for k in metric_names if k not in preferred]
            metric_names = preferred + rest

        matrix_rows, item_ids = [], []
        skipped = 0
        for r in rows:
            cols = r.get("metrics_columns") or []
            vals = r.get("metrics_values") or []
            if not cols or not vals:
                skipped += 1
                continue

            mapping = dict(zip(cols, vals))
            vec = [float(mapping.get(name, 0.0)) for name in metric_names]
            matrix_rows.append(vec)
            item_ids.append(str(r.get("scorable_id", "unknown")))

        if not matrix_rows:
            raise ValueError(
                "VisiCalcTool: no rows had usable metrics after mapping "
                f"(non_empty_rows={non_empty}, union_cols={len(union_cols)}, kept={len(metric_names)}, skipped={skipped})"
            )

        vpm = np.asarray(matrix_rows, dtype=np.float32)
        return vpm, metric_names, item_ids
