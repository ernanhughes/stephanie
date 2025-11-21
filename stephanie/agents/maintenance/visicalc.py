# stephanie/agents/maintenance/visicalc.py
from __future__ import annotations

from dataclasses import asdict
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

import numpy as np

from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.scorable_processor import ScorableProcessor
from stephanie.zeromodel.visicalc_report import (
    compute_visicalc_report,
    format_visicalc_report,
    validate_visicalc_report,
    save_visicalc_report_json,
    save_visicalc_report_csv,
)

log = logging.getLogger(__name__)


def _get_role(obj: Any) -> Optional[str]:
    """Best-effort role access for Scorable | dict."""
    if obj is None:
        return None
    # Scorable-like
    if hasattr(obj, "role"):
        return getattr(obj, "role", None)
    # dict-like
    if isinstance(obj, dict):
        meta = obj.get("meta") or {}
        if isinstance(meta, dict) and "role" in meta:
            return meta.get("role")
        return obj.get("role")
    return None


class VisiCalcAgent(BaseAgent):
    """
    Maintenance / analysis agent that:

      1. Uses ScorableProcessor to build canonical feature rows
      2. Builds an (N, D) metrics matrix from those rows
      3. Runs VisiCalc cohort analytics (compute_visicalc_report)
      4. Logs + saves JSON/CSV reports for offline analysis

    Inputs:
      context[self.input_key] = List[Scorable | dict]

    Outputs:
      context[self.output_key]         = List[dict] (canonical feature rows)
      context["visicalc_report"]       = dataclass-as-dict
      context["visicalc_metric_keys"]  = List[str]
      context["visicalc_report_text"]  = pretty-printed summary
      context["visicalc_json_file"]    = path to JSON report
      context["visicalc_csv_file"]     = path to CSV report
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)

        # Behavior knobs (sane defaults)
        self.progress_enabled: bool = bool(cfg.get("progress", True))
        self.filter_role: bool = bool(cfg.get("filter_role", False))
        self.scorable_role: str = cfg.get("scorable_role", "assistant")

        # Batch + scoring options
        self.batch_size: int = int(cfg.get("batch_size", 64))
        self.attach_scores: bool = bool(cfg.get("attach_scores", False))
        self.scoring_dims: Optional[List[str]] = cfg.get("scoring_dims")

        # progress / concurrency knobs
        self.max_concurrency: int = int(cfg.get("max_concurrency", 8))
        self.progress_log_every: int = int(cfg.get("progress_log_every", 25))
        self.progress_leave: bool = bool(cfg.get("progress_leave", True))
        self.progress_position: int = int(cfg.get("progress_position", 0))

        # -----------------------------
        # VisiCalc cohort analysis knobs
        # -----------------------------
        vis_cfg = cfg.get("visicalc", {}) or {}
        self.visicalc_enabled: bool = bool(vis_cfg.get("enabled", True))

        # If None, we’ll auto-infer numeric columns from the first row
        self.visicalc_metric_keys: Optional[List[str]] = vis_cfg.get("metric_keys")

        # Which metric to treat as the "frontier" (can be None → first column)
        self.visicalc_frontier_metric: Optional[str] = vis_cfg.get(
            "frontier_metric",
            "sicql.clarity.score",
        )

        # How many row regions (coarse bands) to split into
        self.visicalc_row_region_splits: int = int(
            vis_cfg.get("row_region_splits", 4)
        )

        # Output directory + file names
        self.out_dir: Path = Path(vis_cfg.get("out_dir", "debug/visicalc"))
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # These can be overridden via config if needed
        self.visicalc_json_file: Path = Path(
            vis_cfg.get("json_file", self.out_dir / "visicalc_report.json")
        )
        self.visicalc_csv_file: Path = Path(
            vis_cfg.get("csv_file", self.out_dir / "visicalc_report.csv")
        )

        self.scorable_processor: ScorableProcessor = ScorableProcessor(
            self.cfg.get("processor", {}),
            memory,
            container,
            logger,
        )

    # ---------- Public entry point ----------

    async def run(self, context: dict) -> dict:
        """
        Expects context[self.input_key] = List[Scorable | dict].
        Produces:
          - context[self.output_key] = List[dict] (canonical features rows)
          - VisiCalc cohort report in JSON/CSV + context fields.
        """
        scorables = list(context.get(self.input_key) or [])

        if self.filter_role:
            before = len(scorables)
            scorables = [
                s for s in scorables if _get_role(s) == self.scorable_role
            ]
            log.info(
                "VisiCalcAgent: filtered scorables by role=%r: %d → %d",
                self.scorable_role,
                before,
                len(scorables),
            )

        if not scorables:
            log.warning("VisiCalcAgent: no scorables to process after filtering")
            context[self.output_key] = []
            return context

        # 1) Canonical feature rows via ScorableProcessor
        rows = await self.scorable_processor.process_many(
            scorables,
            context=context,
        )
        context[self.output_key] = rows

        # 2) Cohort-level VisiCalc analytics
        if self.visicalc_enabled and rows:
            try:
                report, used_metric_keys = self._compute_visicalc_for_rows(rows)

                # Persist / expose for downstream tools
                context["visicalc_report"] = asdict(report)
                context["visicalc_metric_keys"] = used_metric_keys

                pretty = format_visicalc_report(report)
                context["visicalc_report_text"] = pretty

                # Validate invariants
                validate_visicalc_report(report)

                # Save to disk for offline analysis
                save_visicalc_report_json(report, self.visicalc_json_file)
                save_visicalc_report_csv(report, self.visicalc_csv_file)

                context["visicalc_json_file"] = str(self.visicalc_json_file)
                context["visicalc_csv_file"] = str(self.visicalc_csv_file)

                log.info("VisiCalc cohort summary:\n%s", pretty)
                log.info(
                    "VisiCalcAgent: saved JSON=%s, CSV=%s",
                    self.visicalc_json_file,
                    self.visicalc_csv_file,
                )

            except Exception:
                log.exception("VisiCalc cohort analysis failed")

        return context

    # ---------- Internal helpers ----------

    def _compute_visicalc_for_rows(
        self,
        rows: List[Dict[str, Any]],
    ):
        """
        Build an (N, D) matrix from canonical feature rows and run VisiCalc.

        Expected row schema (from ScorableProcessor):
          - 'metrics_columns': List[str]
          - 'metrics_values':  List[float]

        Returns:
            (VisiCalcReport, used_metric_keys)
        """
        if not rows:
            raise ValueError("VisiCalc: no rows to analyze")

        # -------------------------
        # 1) Establish canonical metric_names
        # -------------------------
        first = rows[0]
        metric_names: List[str] = list(first.get("metrics_columns") or [])

        if not metric_names:
            raise ValueError(
                "VisiCalc: first row has no 'metrics_columns'; "
                "cannot build metric matrix"
            )

        num_metrics = len(metric_names)
        log.debug(
            "VisiCalc: using %d metrics from metrics_columns[0], example=%r",
            num_metrics,
            metric_names[:8],
        )

        # Optional override: restrict to subset of metric_keys if provided
        if self.visicalc_metric_keys:
            # keep only those that exist in metric_names, preserve order
            filtered = [k for k in self.visicalc_metric_keys if k in metric_names]
            if filtered:
                metric_names = filtered
                num_metrics = len(metric_names)
                log.info(
                    "VisiCalc: restricted to %d metrics via config.metric_keys; "
                    "example=%r",
                    num_metrics,
                    metric_names[:8],
                )
            else:
                log.warning(
                    "VisiCalc: visicalc.metric_keys configured but none found "
                    "in metrics_columns; using all %d metrics instead",
                    len(first.get("metrics_columns") or []),
                )

        # -------------------------
        # 2) Build matrix X (N, D) from metrics_values
        # -------------------------
        matrix_rows: List[List[float]] = []
        skipped = 0

        for idx, row in enumerate(rows):
            cols = row.get("metrics_columns") or metric_names
            vals = row.get("metrics_values")

            if vals is None:
                skipped += 1
                log.debug(
                    "VisiCalc: row %d (scorable_id=%r) has no 'metrics_values'; skipping",
                    idx,
                    row.get("scorable_id"),
                )
                continue

            # Align by column name
            mapping = {k: float(v) for k, v in zip(cols, vals)}
            vec = [
                float(mapping.get(name, 0.0))
                for name in metric_names
            ]
            matrix_rows.append(vec)

        if not matrix_rows:
            raise ValueError(
                "VisiCalc: all rows were missing metrics_values; "
                "cannot build matrix"
            )

        X = np.asarray(matrix_rows, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(
                f"VisiCalc: expected 2D matrix (rows x metrics), got {X.shape!r}"
            )

        log.info(
            "VisiCalc: built matrix with shape=%s from %d rows (skipped=%d)",
            X.shape,
            len(matrix_rows),
            skipped,
        )

        # -------------------------
        # 3) Choose frontier metric
        # -------------------------
        frontier_metric = self.visicalc_frontier_metric

        if frontier_metric and frontier_metric not in metric_names:
            log.warning(
                "VisiCalc: requested frontier_metric=%r not in metric_names; "
                "falling back to first metric=%r",
                frontier_metric,
                metric_names[0],
            )
            frontier_metric = metric_names[0]
        elif not frontier_metric:
            frontier_metric = metric_names[0]

        # -------------------------
        # 4) Run pure VisiCalc over this cohort
        # -------------------------
        report = compute_visicalc_report(
            vpm=X,
            metric_names=metric_names,
            frontier_metric=frontier_metric,
            row_region_splits=self.visicalc_row_region_splits,
        )

        log.info(
            "VisiCalc: report frontier_metric=%r on matrix shape=%s "
            "(metrics=%d, rows=%d)",
            report.frontier_metric,
            X.shape,
            len(metric_names),
            X.shape[0],
        )

        return report, metric_names
