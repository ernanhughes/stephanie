# stephanie/scoring/metrics/metric_filter_group_feature.py

from __future__ import annotations
import logging
from typing import Any, Dict, List

import numpy as np

from stephanie.scoring.metrics.base_group_feature import BaseGroupFeature
from stephanie.scoring.metrics.metric_filter import MetricFilter
from stephanie.scoring.metrics.feature_report import FeatureReport  # assuming you have this (same used elsewhere)
from pathlib import Path

log = logging.getLogger(__name__)

class MetricFilterGroupFeature(BaseGroupFeature):
    name = "metric_filter"
    requires: list[str] = []  # upstream deps if you ever want enforced ordering

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
        self._last_summary: Dict[str, Any] | None = None
        self._last_selected: List[str] | None = None

    async def apply(self, rows: list[dict], context: dict) -> list[dict]:
        """
        Batch-level metric selection + cleanup.
        - Builds a consistent column universe across rows
        - Runs MetricFilter.select()
        - Rewrites each row's metrics to the selected column set
        - Persists a short summary in MetricStore (if available)
        """
        if not self.enabled or not rows:
            return rows

        # 0) Collect union of metric columns across all rows
        all_cols: list[str] = []
        for r in rows:
            cols = r.get("metrics_columns") or []
            if cols:
                all_cols.extend(cols)

        if not all_cols:
            summary = {
                "status": "no_cols",
                "reason": "no metric columns present on any row",
                "total_rows": len(rows),
            }
            context["metric_filter_summary"] = summary
            self._last_summary = summary
            self._persist_summary(context, summary)
            return rows

        # 1) Dense matrix X (n_rows x n_cols) aligned to uniq_cols
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

        run_id = context.get("pipeline_run_id")
        run_dir  = Path(context.get("run_dir") or f"runs/critic/{run_id}")
        run_dir.mkdir(parents=True, exist_ok=True)

        # A) Write a lock list file for downstream training
        lock_path = run_dir / "kept_features.txt"
        lock_path.write_text("\n".join(selected_names), encoding="utf-8")
        # Persist for reporters / downstream tools
        metric_store = self.memory.metrics
        try:
            metric_store.upsert_group_meta(
                run_id=run_id,
                patch={
                    "metric_filter": {
                        "kept_columns": selected_names,  # <-- reporter should read this exact key
                        "n_kept": len(selected_names),
                        "total_raw": len(uniq_cols),
                    }
                },
            )
        except Exception as e:
            log.warning("[MetricFilterGroupFeature] persist kept_columns failed: %s", e)
        # Safety fallback
        if not selected_names:
            self._warn("[MetricFilterGroupFeature] selection empty; keeping original columns.")
            selected_names = list(uniq_cols)
            keep_mask = np.ones((len(uniq_cols),), dtype=bool)

        self._last_selected = list(selected_names)

        # 3) Rewrite each row's metrics to the filtered column set (and sync metrics_vector)
        for r in rows:
            cols = r.get("metrics_columns") or []
            vals = r.get("metrics_values") or []
            mapping = dict(zip(cols, vals)) if cols and vals else {}

            r["metrics_columns"] = list(selected_names)
            filtered_vals = [float(mapping.get(name, 0.0)) for name in selected_names]
            r["metrics_values"] = filtered_vals
            r["metrics_vector"] = {name: val for name, val in zip(selected_names, filtered_vals)}

        # 4) Build + stash summary for debugging / observability
        summary = {
            "status": "ok",
            "kept_count": len(selected_names),
            "total_raw": len(uniq_cols),
            "kept_sample": selected_names[:20],
        }
        context["metric_filter_summary"] = summary
        context["filtered_metric_names"] = list(selected_names)
        self._last_summary = summary

        log.info(
            "[MetricFilterGroupFeature] kept %d of %d metrics",
            summary["kept_count"], summary["total_raw"]
        )

        # 5) Persist summary (best-effort)
        self._persist_summary(context, summary)

        return rows

    def _persist_summary(self, context: dict, summary: dict) -> None:
        try:
            self.memory.metrics.upsert_group_meta(
                run_id=context.get("pipeline_run_id", "unknown"),
                patch={"metric_filter_summary": summary},
            )
        except Exception as e:
            self._warn(f"[MetricFilterGroupFeature] persist skipped: {e}")

    # ---------- Feature report hook ----------
    def report(self) -> FeatureReport:
        if not self._last_summary:
            return FeatureReport(
                name=self.name,
                kind="group",
                ok=True,
                quality=None,
                summary="no-op",
                details={},
                warnings=[],
            )
        kept = self._last_summary.get("kept_count")
        total = self._last_summary.get("total_raw")
        ok = kept is not None and total is not None and kept > 0
        quality = float(kept / max(total, 1)) if (kept is not None and total) else None
        return FeatureReport(
            name=self.name,
            kind="group",
            ok=ok,
            quality=quality,
            summary=f"kept {kept} of {total} metrics",
            details={
                "summary": self._last_summary,
                "selected_names_sample": (self._last_selected or [])[:20],
            },
            warnings=[],
        )
