# stephanie/agents/maintenance/visicalc.py
from __future__ import annotations

from dataclasses import asdict
import logging
from pathlib import Path
from typing import List, Optional

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


class VisiCalcAgent(BaseAgent):
    """
    Agent that places the ScorableProcessor at the front of the pipeline.
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)

        # Behavior knobs (sane defaults)
        self.progress_enabled: bool = bool(cfg.get("progress", True))
        self.filter_role: bool = bool(cfg.get("filter_role", False))
        self.scorable_role: str = cfg.get("scorable_role", "candidate")


        # Batch + scoring options
        self.batch_size: int = int(cfg.get("batch_size", 64))
        self.attach_scores: bool = bool(cfg.get("attach_scores", True))
        self.scoring_dims: Optional[List[str]] = cfg.get("scoring_dims")

        # progress/concurrency knobs
        self.max_concurrency: int = int(cfg.get("max_concurrency", 8))
        self.progress_log_every: int = int(cfg.get("progress_log_every", 25))
        self.progress_leave: bool = bool(cfg.get("progress_leave", True))
        self.progress_position: int = int(cfg.get("progress_position", 0))

        # -----------------------------
        # VisiCalc cohort analysis knobs
        # -----------------------------
        vis_cfg = cfg.get("visicalc", {})
        self.visicalc_enabled: bool = bool(vis_cfg.get("enabled", True))
        # If None, we’ll auto-infer numeric columns from the first row
        self.visicalc_metric_keys: Optional[List[str]] = vis_cfg.get("metric_keys")
        self.visicalc_frontier_metric: Optional[str] = vis_cfg.get("frontier_metric", 'sicql.clarity.score') 
        self.visicalc_row_region_splits: int = int(
            vis_cfg.get("row_region_splits", 4)
        )
        self.out_dir: Path = Path(vis_cfg.get("out_dir", "debug/visicalc"))
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.visicalc_json_file: str = str(self.out_dir / "visicalc_report.json")
        self.visicalc_csv_file: str = str(self.out_dir / "visicalc_report.csv")

        self.scorable_processor: ScorableProcessor = ScorableProcessor(
            self.cfg.get("processor", {}),
            memory,
            container,
            logger
        )

    # ---------- Public entry point ----------

    async def run(self, context: dict) -> dict:
        """
        Expects context['scorables'] = List[dict|Scorable].
        Produces:
          - context['scorable_features'] = List[dict] (canonical features rows)
          - updates each scorable.meta with short-form 'domains' and 'ner'
          - context['scorable_annotation_summary'] with stats
        """
        scorables = list(context.get(self.input_key) or [])
        rows = await self.scorable_processor.process_many(scorables, context=context)
    
        if self.visicalc_enabled and rows:
            try:
                report, used_metric_keys = self._compute_visicalc_for_rows(rows)
                context["visicalc_report"] = asdict(report)
                context["visicalc_metric_keys"] = used_metric_keys

                pretty = format_visicalc_report(report)

                context["visicalc_report_text"] = pretty
                log.info("VisiCalc cohort summary:\n%s", pretty)

                validate_visicalc_report(report)
            except Exception:
                log.exception("VisiCalc cohort analysis failed")


        context[self.output_key] = rows

        return context

    def _compute_visicalc_for_rows(
        self,
        rows: List[dict],
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
        metric_names = list(first.get("metrics_columns") or [])

        if not metric_names:
            raise ValueError(
                "VisiCalc: first row has no 'metrics_columns'; "
                "cannot build metric matrix"
            )

        num_metrics = len(metric_names)
        log.debug(
            "VisiCalc: using %d metrics from metrics_columns[0], "
            "example=%r",
            num_metrics,
            metric_names[:8],
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

            # If column order/length matches, fast path
            if cols == metric_names and len(vals) == num_metrics:
                vec = [
                    float(v) if np.isfinite(v) else 0.0
                    for v in vals
                ]
            else:
                # Align by name via dict
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

        log.debug(
            "VisiCalc: built matrix with shape=%s from %d rows (skipped=%d)",
            X.shape,
            len(matrix_rows),
            skipped,
        )

        # -------------------------
        # 3) Choose frontier metric
        # -------------------------
        frontier_metric = getattr(self, "visicalc_frontier_metric", None)

        if frontier_metric and frontier_metric not in metric_names:
            log.warning(
                "VisiCalc: requested frontier_metric=%r not in metric_names; "
                "falling back to first metric=%r",
                frontier_metric,
                metric_names[0],
            )
            frontier_metric = None

        if not frontier_metric:
            frontier_metric = metric_names[0]

        # -------------------------
        # 4) Run pure VisiCalc over this cohort
        # -------------------------
        report = compute_visicalc_report(
            vpm=X,
            metric_names=metric_names,
            frontier_metric=frontier_metric,
            row_region_splits=getattr(self, "visicalc_row_region_splits", 4),
        )

        save_visicalc_report_json(report, self.visicalc_json_file)
        save_visicalc_report_csv(report, self.visicalc_csv_file)

        log.info(
            "VisiCalc: report frontier_metric=%r on matrix shape=%s "
            "(metrics=%d, rows=%d)",
            report.frontier_metric,
            X.shape,
            len(metric_names),
            X.shape[0],
        )

        return report, metric_names
