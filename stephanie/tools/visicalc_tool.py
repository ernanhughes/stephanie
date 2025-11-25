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
    def apply_batch(
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
        """
        Build:
            scores matrix
            metric_names
            item_ids
        """
        # Collect all metric names from first row
        all_cols = rows[0].get("metrics_columns", [])

        metric_names = self.mapper.select_columns(all_cols) or all_cols

        # Apply optional preferred ordering
        if self.visicalc_metric_keys:
            preferred = [k for k in self.visicalc_metric_keys if k in metric_names]
            rest      = [k for k in metric_names if k not in preferred]
            metric_names = preferred + rest

        # Build matrix
        matrix_rows = []
        item_ids    = []

        for r in rows:
            cols = r.get("metrics_columns") or []
            vals = r.get("metrics_values") or []
            if not cols:
                continue

            mapping = dict(zip(cols, vals))
            vector  = [float(mapping.get(name, 0.0)) for name in metric_names]

            matrix_rows.append(vector)
            item_ids.append(str(r["scorable_id"]))

        vpm = np.asarray(matrix_rows, dtype=np.float32)
        return vpm, metric_names, item_ids
