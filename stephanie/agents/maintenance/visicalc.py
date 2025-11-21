# stephanie/agents/maintenance/visicalc.py
from __future__ import annotations

from dataclasses import asdict
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.scorable_processor import ScorableProcessor
from stephanie.utils.json_sanitize import dumps_safe
from dataclasses import asdict  # <- you can actually drop this now if unused
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.scorable_processor import ScorableProcessor
from stephanie.utils.json_sanitize import dumps_safe
from stephanie.zeromodel.visicalc import (
    VisiCalc,
    graph_quality_from_report,
)

log = logging.getLogger(__name__)


class VisiCalcAgent(BaseAgent):
    """
    Maintenance / analysis agent that:

      1. Uses ScorableProcessor to build canonical feature rows
      2. Builds an (N, D) metrics matrix from those rows
      3. Runs VisiCalc cohort analytics (VisiCalc.from_rows)
      4. Logs + saves JSON/CSV reports for offline analysis
      5. Optionally compares baseline vs targeted cohorts + TopLeft diff

    ----------
    Inputs
    ----------
    Context keys expected:

      - context[self.input_key]              = List[Scorable | dict]
          Base cohort of scorables to process.

      - context["scorables_targeted"]        = Optional[List[dict]]
      - context["scorables_baseline"]        = Optional[List[dict]]
          Optional A/B cohorts. If both are present and non-empty:
            * targeted  → "improved" / experimental cohort
            * baseline  → comparison cohort

    ----------
    Outputs
    ----------
    Core outputs:

      - context[self.output_key]             = List[dict]
          Canonical feature rows from ScorableProcessor.

      - context["visicalc_report"]           = dict (single-cohort case)
      - context["visicalc_report_text"]      = pretty JSON string
      - context["visicalc_metric_keys"]      = List[str]

      - context["visicalc_targeted_report"]  = dict (A/B case)
      - context["visicalc_baseline_report"]  = dict
      - context["visicalc_targeted_text"]    = pretty JSON string
      - context["visicalc_baseline_text"]    = pretty JSON string
      - context["visicalc_target_quality"]   = float
      - context["visicalc_baseline_quality"] = float
      - context["visicalc_ab_diff"]          = dict (VisiCalc.diff result)
      - context["visicalc_ab_topleft"]       = dict (TopLeft comparison)

      - Optional:
        - context["visicalc_features"]       = List[float]   # single-cohort episode features
        - context["visicalc_feature_names"]  = List[str]

    ----------
    Config schema (Hydra-style)
    ----------
    Top-level agent config:

      visicalc_agent:
        _target_: stephanie.agents.maintenance.visicalc.VisiCalcAgent

        # BaseAgent wiring
        input_key: scorables            # where to read scorables from context
        output_key: scorable_features   # where to write canonical rows

        # Progress / role filtering
        progress: true                  # enable/disable progress logging
        filter_role: false              # if true, restrict scorables by role
        scorable_role: assistant        # role to keep when filter_role=true

        # Scoring / batching
        batch_size: 64                  # ScorableProcessor batch size
        attach_scores: false            # whether ScorableProcessor attaches scores
        scoring_dims: null              # optional: limit scoring to these dimensions

        # Concurrency & progress bar behavior
        max_concurrency: 8
        progress_log_every: 25
        progress_leave: true
        progress_position: 0

        # ScorableProcessor config (passed through unmodified)
        processor:
          offload_mode: inline          # or "rpc"/"async" etc.
          # ... other ScorableProcessor options ...

        # VisiCalc-specific options
        visicalc:
          enabled: true                 # turn VisiCalc on/off

          # Optional: restrict to these metric keys (subset of metrics_columns)
          # If null, use whatever metrics_columns ScorableProcessor emits.
          metric_keys: null             # e.g. ["HRM.aggregate", "sicql.clarity.score"]

          # Metric to treat as the "frontier" dimension
          # If not present in metrics_columns, falls back to the first metric.
          frontier_metric: "HRM.aggregate"

          # How many row bands to split the episode into
          row_region_splits: 4

          # Output locations (per run_id)
          out_dir: "runs/visicalc"      # final path is out_dir / run_id

          # These are relative to out_dir / run_id unless absolute paths are given
          json_file: "visicalc_report.json"
          csv_file: "visicalc_report.csv"

    Notes:
      - `processor` is forwarded directly into `ScorableProcessor`; it controls
        how scorables are annotated, scored, and offloaded to the bus.
      - `visicalc.metric_keys` is not required; by default the agent uses whatever
        `metrics_columns` the first row exposes.
      - `frontier_metric` should usually be a high-level quality channel like
        "HRM.aggregate" or "sicql.overall.score".
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

        # Which metric to treat as the "frontier" (can be None → first column)             "sicql.clarity.score",

        self.visicalc_frontier_metric: Optional[str] = vis_cfg.get(
            "frontier_metric",
            "HRM.aggregate",
        )

        # How many row regions (coarse bands) to split into
        self.visicalc_row_region_splits: int = int(
            vis_cfg.get("row_region_splits", 4)
        )

        # Output directory + file names
        self.out_dir: Path = Path(vis_cfg.get("out_dir", "runs/visicalc")) / self.run_id
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # These can be overridden via config if needed
        self.visicalc_json_file: Path = Path(
            vis_cfg.get("json_file", self.out_dir / "visicalc_report.json")
        ) 
        self.visicalc_csv_file: Path = Path(
            vis_cfg.get("csv_file", self.out_dir / "visicalc_report.csv")
        )

        self.scorable_processor: ScorableProcessor = ScorableProcessor(
            self.cfg.get("processor", {"offload_mode": "inline"}),
            memory,
            container,
            logger,
        )

    async def run(self, context: dict) -> dict:
        """
        Expects:
          - context['scorables']            = List[dict|Scorable] (union)
          - optional: context['scorables_targeted'], ['scorables_baseline']

        Produces:
          - context['scorable_features']         = List[dict] (canonical rows)
          - context['visicalc_*'] entries with cohort reports and deltas
        """
        scorables_all = list(context.get(self.input_key) or [])


        # 1) Canonical rows for all scorables (we'll slice into cohorts by id)
        rows = await self.scorable_processor.process_many(
            scorables_all,
            context=context,
        )
        context[self.output_key] = rows

        if not self.visicalc_enabled or not rows:
            log.warning("VisiCalcAgent: skipping VisiCalc analysis")
            return context

        # Build id → row map once
        id_to_row = {
            r.get("scorable_id"): r
            for r in rows
            if r.get("scorable_id") is not None
        }

        scorables_targeted = list(context.get("scorables_targeted") or [])
        scorables_baseline = list(context.get("scorables_baseline") or [])

        # Helper to slice rows by a cohort of scorables
        def _rows_for_cohort(scorables: List[dict]) -> List[dict]:
            out = []
            for s in scorables:
                sid = str(s.get("scorable_id"))
                row = id_to_row.get(sid)
                if row is not None:
                    out.append(row)
            return out

        try:
            # ----------------------
            # Case A: A/B cohorts
            # ----------------------
            # ----------------------
            # Case A: A/B cohorts
            # ----------------------
            vpm_tgt = None
            vpm_base = None
            if scorables_targeted and scorables_baseline:
                rows_tgt = _rows_for_cohort(scorables_targeted)
                rows_base = _rows_for_cohort(scorables_baseline)

                if rows_tgt:
                    vc_tgt, metric_keys_tgt = self._compute_visicalc_for_rows(
                        rows_tgt,
                        cohort_label="targeted",
                    )
                    context["visicalc_targeted_report"] = vc_tgt.report.to_dict()
                    context["visicalc_targeted_metric_keys"] = metric_keys_tgt

                    pretty_tgt = vc_tgt.pretty()
                    context["visicalc_targeted_text"] = pretty_tgt
                    log.info("VisiCalc TARGETED cohort:\n%s", pretty_tgt)

                    # Save reports
                    vc_tgt.report.save_json(self.out_dir / "visicalc_targeted.json")
                    vc_tgt.report.save_csv(self.out_dir / "visicalc_targeted.csv")

                    # Scalar quality
                    target_q = graph_quality_from_report(vc_tgt.report)
                    log.info("VisiCalc TARGETED quality=%.4f", target_q)
                    context["visicalc_target_quality"] = target_q

                    vpm_tgt = vc_tgt.scores  # for TopLeft later

                if rows_base:
                    vc_base, metric_keys_base = self._compute_visicalc_for_rows(
                        rows_base,
                        cohort_label="baseline",
                    )
                    context["visicalc_baseline_report"] = vc_base.report.to_dict()
                    context["visicalc_baseline_metric_keys"] = metric_keys_base

                    pretty_base = vc_base.pretty()
                    context["visicalc_baseline_text"] = pretty_base
                    log.info("VisiCalc BASELINE cohort:\n%s", pretty_base)

                    vc_base.report.save_json(self.out_dir / "visicalc_baseline.json")
                    vc_base.report.save_csv(self.out_dir / "visicalc_baseline.csv")

                    base_q = graph_quality_from_report(vc_base.report)
                    log.info("VisiCalc BASELINE quality=%.4f", base_q)
                    context["visicalc_baseline_quality"] = base_q

                    vpm_base = vc_base.scores

                # If we got both, compute delta (using VisiCalc.diff)
                if rows_tgt and rows_base and vpm_tgt is not None and vpm_base is not None:
                    diff = vc_tgt.diff(vc_base)
                    context["visicalc_ab_diff"] = diff
                    log.info(dumps_safe(diff, indent=2))

                    topleft_res = self.compare_vpms_with_topleft(
                        vpm_base,
                        vpm_tgt,
                    )
                    log.info(
                        "VisiCalc A/B TopLeft diff: gain=%.4f loss=%.4f improvement_ratio=%.4f",
                        topleft_res["gain"],
                        topleft_res["loss"],
                        topleft_res["improvement_ratio"],
                    )
                    context["visicalc_ab_topleft"] = topleft_res

                    log.info(
                        "VisiCalc A/B delta: frontier_metric=%r "
                        "global_mean_delta=%.4f frontier_frac_delta=%.4f",
                        diff["frontier_metric"],
                        diff["global_delta"]["mean"],
                        diff["global_delta"]["frontier_frac"],
                    )

            # ----------------------
            # Case B: single cohort (no A/B)
            # ----------------------
            else:
                vc, used_metric_keys = self._compute_visicalc_for_rows(
                    rows,
                    cohort_label="cohort",
                )
                context["visicalc_report"] = vc.report.to_dict()
                context["visicalc_metric_keys"] = used_metric_keys

                pretty = vc.pretty()
                context["visicalc_report_text"] = pretty
                log.info("VisiCalc cohort summary:\n%s", pretty)

                vc.report.save_json(self.out_dir / "visicalc_report.json")
                vc.report.save_csv(self.out_dir / "visicalc_report.csv")

                # Optional: expose features in context if you want downstream use
                context["visicalc_features"] = vc.features.tolist()
                context["visicalc_feature_names"] = vc.feature_names

        except Exception:
            log.exception("VisiCalc cohort analysis failed")

        return context

    def _compute_visicalc_for_rows(
        self,
        rows: List[dict],
        cohort_label: str,
    ):
        """
        Build a VisiCalc instance from canonical feature rows.

        Expected row schema (from ScorableProcessor):
          - 'metrics_columns': List[str]
          - 'metrics_values':  List[float]

        Returns:
            (VisiCalc, used_metric_keys)
        """
        if not rows:
            raise ValueError("VisiCalc: no rows to analyze")

        # 1) Establish canonical metric_names
        first = rows[0]
        metrics_columns = list(first.get("metrics_columns") or [])

        if not metrics_columns:
            raise ValueError(
                "VisiCalc: first row has no 'metrics_columns'; "
                "cannot build metric matrix"
            )

        # 2) Choose frontier metric
        frontier_metric = self.visicalc_frontier_metric
        if frontier_metric and frontier_metric not in metrics_columns:
            log.warning(
                "VisiCalc: requested frontier_metric=%r not in metric_names; "
                "falling back to first metric=%r",
                frontier_metric,
                metrics_columns[0],
            )
            frontier_metric = None

        if not frontier_metric:
            frontier_metric = metrics_columns[0]

        # 3) Let VisiCalc.from_rows handle matrix + item_ids
        vc = VisiCalc.from_rows(
            episode_id=cohort_label,
            rows=rows,
            frontier_metric=frontier_metric,
            row_region_splits=self.visicalc_row_region_splits,
            frontier_low=0.25,   # or make configurable later
            frontier_high=0.75,
            meta={"cohort": cohort_label},
        )

        log.info(
            "VisiCalc: built episode=%r frontier_metric=%r matrix_shape=%s "
            "(metrics=%d, rows=%d)",
            cohort_label,
            vc.frontier_metric,
            vc.scores.shape,
            len(vc.metric_names),
            vc.scores.shape[0],
        )

        return vc, metrics_columns
 
    def compare_vpms_with_topleft(
        self,
        vpm_base: np.ndarray,
        vpm_tgt: np.ndarray,
        *,
        metric_mode: str = "luminance",
        iterations: int = 5,
        push_corner: str = "tl",
    ) -> dict:
        """
        Run TopLeft on baseline and targeted VPMs with identical config,
        then compute a simple visual-diff metric.
        """
        from zeromodel.pipeline.organizer.top_left import TopLeft
        stage = TopLeft(
            metric_mode=metric_mode,
            iterations=iterations,
            push_corner=push_corner,
            monotone_push=True,
            stretch=True,
        )

        # 1) Canonicalize both VPMs
        tl_base, meta_base = stage.process(vpm_base)
        tl_tgt, meta_tgt = stage.process(vpm_tgt)

        # 2) Ensure same shape / type
        tl_base = tl_base.astype(np.float32)
        tl_tgt = tl_tgt.astype(np.float32)
        assert tl_base.shape == tl_tgt.shape, "Base/Target shapes must match after TopLeft"

        # 3) Visual difference: positive = target > base
        diff = tl_tgt - tl_base

        # 4) Aggregate into simple scores
        gain = float(np.sum(np.clip(diff, 0.0, None)))           # total positive mass
        loss = float(np.sum(np.clip(-diff, 0.0, None)))          # total negative mass
        total = gain + loss + 1e-8

        # fraction of mass that's an improvement (0.0–1.0)
        improvement_ratio = gain / total

        return {
            "topleft_base": tl_base,
            "topleft_tgt": tl_tgt,
            "diff": diff,
            "gain": gain,
            "loss": loss,
            "improvement_ratio": improvement_ratio,
            "meta_base": meta_base,
            "meta_tgt": meta_tgt,
        }
