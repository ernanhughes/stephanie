# stephanie/scoring/metrics/metric_filter_group_feature.py
from __future__ import annotations
import logging
from typing import Any, Dict, List, Sequence

import numpy as np

from stephanie.scoring.metrics.base_group_feature import BaseGroupFeature
from stephanie.scoring.metrics.metric_filter import MetricFilter

log = logging.getLogger(__name__)

class MetricFilterGroupFeature(BaseGroupFeature):
    name = "metric_filter"
    requires: list[str] = []  # declare upstream deps if you want ordering enforced

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.filter = MetricFilter(
            k=int(cfg.get("top_k", 100)),
            dup_threshold=float(cfg.get("dup_threshold", 0.995)),
            min_variance=float(cfg.get("min_variance", 1e-8)),
            normalize=bool(cfg.get("normalize", True)),
            include_patterns=list(cfg.get("include", []) or []),
            exclude_patterns=list(cfg.get("exclude", []) or []),
            alias_strip=bool(cfg.get("alias_strip", True)),
        )

    async def apply(self, rows: list[dict], context: dict) -> list[dict]:
        """
        Batch-level metric selection + cleanup.
        - Builds a consistent column universe across rows
        - Runs MetricFilter.select()
        - Rewrites each row's metrics to the selected column set
        - Persists a short summary in MetricStore (if available)
        """
        if not self.enabled:
            return rows

        # 0) Collect union of metric columns across all rows
        all_cols: list[str] = []
        for r in rows:
            cols = r.get("metrics_columns") or []
            if cols:
                all_cols.extend(cols)

        if not all_cols:
            # Nothing to filter; emit a small debug summary and return as-is
            summary = {
                "status": "no_cols",
                "reason": "no metric columns present on any row",
                "total_rows": len(rows),
            }
            context["metric_filter_summary"] = summary
            # best-effort store
            try:
                self.memory.metrics.upsert_group_meta(
                    run_id=context.get("pipeline_run_id", "unknown"),
                    patch={"metric_filter_summary": summary},
                )
            except Exception as e:
                self._warn(f"[MetricFilterGroupFeature] persist skipped: {e}")

            return rows

        # 1) Build a dense matrix X (n_rows x n_cols) aligned to all_cols
        #    We do this via per-row mapping to tolerate missing metrics.
        uniq_cols = list(dict.fromkeys(all_cols))  # stable de-dup while preserving order
        name_to_idx = {name: i for i, name in enumerate(uniq_cols)}

        X_rows: list[list[float]] = []
        for r in rows:
            cols = r.get("metrics_columns") or []
            vals = r.get("metrics_values") or []
            mapping = dict(zip(cols, vals)) if cols and vals else {}
            vec = [float(mapping.get(name, 0.0)) for name in uniq_cols]
            X_rows.append(vec)

        X = np.asarray(X_rows, dtype=np.float32)

        # 2) Run selection
        keep_mask, selected_names = self.filter.select(uniq_cols, X, labels=context.get("labels"))

        # Safety: if selection ends up empty, keep originals to avoid breaking downstream steps.
        if not selected_names:
            self._warn("[MetricFilterGroupFeature] selection empty; keeping original columns.")
            selected_names = list(uniq_cols)

        # 3) Rewrite each row's metrics to the filtered column set
        for r in rows:
            cols = r.get("metrics_columns") or []
            vals = r.get("metrics_values") or []
            mapping = dict(zip(cols, vals)) if cols and vals else {}

            r["metrics_columns"] = list(selected_names)
            r["metrics_values"]  = [float(mapping.get(name, 0.0)) for name in selected_names]
            # keep r["metrics_vector"] if you want; otherwise it can be rebuilt lazily when needed

        # 4) Build + stash summary for debugging / observability
        summary = {
            "status": "ok",
            "kept_count": len(selected_names),
            "total_raw": len(uniq_cols),
            "kept_sample": selected_names[:20],
        }
        context["metric_filter_summary"] = summary

        # 5) Persist summary (best-effort)
        try:
            self.memory.metrics.upsert_group_meta(
                run_id=context.get("pipeline_run_id", "unknown"),
                patch={"metric_filter_summary": summary},
            )
        except Exception as e:
            self._warn(f"[MetricFilterGroupFeature] persist skipped: {e}")

        return rows
