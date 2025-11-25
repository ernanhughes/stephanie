# stephanie/components/critic/agents/visicalc.py
from __future__ import annotations

import csv
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.metrics.metric_importance import (
    compute_metric_importance, save_metric_importance_json)
from stephanie.scoring.metrics.metric_mapping import MetricMapper
from stephanie.scoring.metrics.scorable_processor import ScorableProcessor
from stephanie.scoring.metrics.visicalc import (VisiCalc,
                                                graph_quality_from_report)
from stephanie.utils.json_sanitize import dumps_safe

log = logging.getLogger(__name__)


class CriticCohortAgent(BaseAgent):
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
        _target_: stephanie.agents.maintenance.critic_cohort.CriticCohortAgent

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

    def __init__(self, cfg, memory, container, logger, run_id: Optional[str] = None):
        super().__init__(cfg, memory, container, logger)

        if run_id is not None:
            self.run_id: str = run_id
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
        vis_cfg = cfg.get("visicalc", {}) or cfg
        self.visicalc_enabled: bool = bool(vis_cfg.get("enabled", True))

        # If None, we’ll auto-infer numeric columns from the first row
        self.visicalc_metric_keys: Optional[List[str]] = vis_cfg.get("metric_keys")

        # Which metric to treat as the "frontier" (can be None → first column)             "sicql.clarity.score",

        self.visicalc_frontier_metric: Optional[str] = vis_cfg.get(
            "frontier_metric",
            "HRM.aggregate",
        )
        self.visicalc_frontier_low: float = float(vis_cfg.get("frontier_low", 0.25))
        self.visicalc_frontier_high: float = float(vis_cfg.get("frontier_high", 0.75))

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
        self.metric_mapper = MetricMapper.from_config(vis_cfg)
        self.vpm_png_cfg: Dict[str, Any] = vis_cfg.get("vpm_png", {}) or {}
        self.vpm_png_enabled: bool = bool(self.vpm_png_cfg.get("enabled", True))
        self.vpm_png_mode: str = str(self.vpm_png_cfg.get("mode", "L"))
        self.vpm_png_per_metric: bool = bool(
            self.vpm_png_cfg.get("per_metric_normalize", True)
        )
        self.vpm_png_target_name: str = str(
            self.vpm_png_cfg.get("target_file", "visicalc_targeted_vpm.png")
        )
        self.vpm_png_baseline_name: str = str(
            self.vpm_png_cfg.get("baseline_file", "visicalc_baseline_vpm.png")
        )
        self.visicalc_top_k_metrics = vis_cfg.get("top_k_metrics", 100)  # e.g. 50
        self.visicalc_min_effect = float(vis_cfg.get("min_effect", 0.1))  # e.g. 0.25

        # --- Importance-based metric subset (online + offline) ---
        imp_cfg = vis_cfg.get("importance_filter") or {}

        # 1) Per-run *save* location for metric_importance.json
        #    (used by VisiCalc runs, and later by OfflineImportanceReducer)
        self.importance_save_path: Path = self.out_dir / "metric_importance.json"

        # 2) Where to *load* important metric names from for filtering
        self.importance_enabled: bool = bool(imp_cfg.get("enabled", False))

        # If config specifies a path, we use that (e.g. "config/core_metrics.json").
        # Otherwise, we default to this run's metric_importance.json.
        importance_path_cfg = imp_cfg.get("path")
        if importance_path_cfg:
            self.importance_load_path: Optional[Path] = Path(str(importance_path_cfg))
        else:
            self.importance_load_path: Optional[Path] = self.importance_save_path

        # 0 / None → no limit, use all metrics from the file that are present
        self.importance_top_k: Optional[int] = imp_cfg.get("top_k")
        if self.importance_top_k is not None:
            try:
                self.importance_top_k = int(self.importance_top_k)
                if self.importance_top_k <= 0:
                    self.importance_top_k = None
            except Exception:
                self.importance_top_k = None

        # cache for loaded names
        self._important_metric_names: Optional[List[str]] = None

    async def run(self, context: dict) -> dict:
        """
        Expects:
          - context[self.input_key]              = List[dict|Scorable]
          - optional: context['scorables_targeted'], context['scorables_baseline']

        Produces:
          - context[self.output_key]                   = List[dict] (canonical rows)
          - context['visicalc_*'] entries with cohort reports, qualities and deltas
          - optional: context['metric_importance'], context['visicalc_metric_importance']
        """
        # ------------------------------------------------------------------
        # 0) Load scorables and apply optional role filter
        # ------------------------------------------------------------------
        scorables_all = list(context.get(self.input_key) or [])

        if self.filter_role:
            filtered = []
            for s in scorables_all:
                role = None
                if isinstance(s, dict):
                    role = s.get("role")
                else:
                    role = getattr(s, "role", None)
                if role == self.scorable_role:
                    filtered.append(s)

            log.info(
                "CriticCohortAgent: filter_role=True scorable_role=%r → kept %d/%d scorables",
                self.scorable_role,
                len(filtered),
                len(scorables_all),
            )
            scorables_all = filtered

        # ------------------------------------------------------------------
        # 1) Canonical feature rows for *all* scorables
        # ------------------------------------------------------------------
        rows = await self.scorable_processor.process_many(
            scorables_all,
            context=context,
        )
        context[self.output_key] = rows

        if not self.visicalc_enabled or not rows:
            log.warning(
                "CriticCohortAgent: skipping VisiCalc analysis (enabled=%s, rows=%d)",
                self.visicalc_enabled,
                len(rows),
            )
            return context

        # Build id → row map once (all scorables)
        id_to_row = {
            str(r.get("scorable_id")): r
            for r in rows
            if r.get("scorable_id") is not None
        }

        # Helper to slice rows by a cohort of scorables
        def _rows_for_cohort(scorables: List[Any]) -> List[dict]:
            out: List[dict] = []
            for s in scorables:
                # Robustly get an id from either dict or Scorable
                if isinstance(s, dict):
                    sid = s.get("scorable_id") or s.get("id")
                else:
                    sid = getattr(s, "scorable_id", None) or getattr(s, "id", None)

                if sid is None:
                    continue

                row = id_to_row.get(str(sid))
                if row is not None:
                    out.append(row)
            return out

        scorables_targeted = list(context.get("scorables_targeted") or [])
        scorables_baseline = list(context.get("scorables_baseline") or [])

        try:
            # ------------------------------------------------------------------
            # Case A: A/B cohorts present → targeted vs baseline comparison
            # ------------------------------------------------------------------
            vc_tgt = None
            vc_base = None
            vpm_tgt = None
            vpm_base = None

            if scorables_targeted and scorables_baseline:
                rows_tgt = _rows_for_cohort(scorables_targeted)
                rows_base = _rows_for_cohort(scorables_baseline)

                # ---- Targeted cohort ----
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

                    # Scalar quality for summary dashboards
                    target_q = graph_quality_from_report(vc_tgt.report)
                    log.info("VisiCalc TARGETED quality=%.4f", target_q)
                    context["visicalc_target_quality"] = target_q

                    vpm_tgt = vc_tgt.scores

                # ---- Baseline cohort ----
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

                # ---- A/B comparison, visuals + importance ----
                if (
                    rows_tgt
                    and rows_base
                    and vpm_tgt is not None
                    and vpm_base is not None
                ):
                    # 1) VisiCalc's own diff (frontier deltas)
                    diff = vc_tgt.diff(vc_base)
                    context["visicalc_ab_diff"] = diff
                    log.info(dumps_safe(diff, indent=2))

                    # 2) TopLeft visual diff metrics (matrix-level gain/loss)
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

                    # 2A-1) Save full matrices as CSV
                    self._save_vpm_matrix_csv(vpm_tgt,  list(vc_tgt.metric_names),  list(vc_tgt.item_ids),  self.out_dir / "visicalc_targeted_matrix.csv")
                    self._save_vpm_matrix_csv(vpm_base, list(vc_base.metric_names), list(vc_base.item_ids), self.out_dir / "visicalc_baseline_matrix.csv")

                    # 2A-2) Save combined NPZ for tiny_critic trainer (labels inside)
                    # NOTE: metric_names must match (you already ensure same D before importance/diff)
                    self._save_ab_npz_dataset(vpm_base, vpm_tgt, list(vc_tgt.metric_names), self.out_dir / "visicalc_ab_dataset.npz")

                    # 3) VPM PNGs (full matrices)
                    if self.vpm_png_enabled:
                        try:
                            if vc_tgt is not None:
                                tgt_png = self.out_dir / self.vpm_png_target_name
                                vc_tgt.save_vpm_png(
                                    tgt_png,
                                    per_metric_normalize=self.vpm_png_per_metric,
                                    mode=self.vpm_png_mode,
                                )

                            if vc_base is not None:
                                base_png = self.out_dir / self.vpm_png_baseline_name
                                vc_base.save_vpm_png(
                                    base_png,
                                    per_metric_normalize=self.vpm_png_per_metric,
                                    mode=self.vpm_png_mode,
                                )

                            log.info(
                                "CriticCohortAgent: saved VPM PNGs → target=%s baseline=%s",
                                tgt_png,
                                base_png,
                            )
                        except Exception as e:
                            log.warning(
                                "CriticCohortAgent: failed to save VPM PNGs: %s",
                                e,
                                exc_info=True,
                            )

                    # 4) Top-left reordered VPM heatmaps (visual proof)
                    try:
                        self._render_ab_topleft_heatmaps(
                            vpm_target=vpm_tgt,
                            vpm_baseline=vpm_base,
                            out_dir=self.out_dir,
                        )
                    except Exception:
                        log.exception("VisiCalc: failed to render TopLeft VPM images")

                    # 5) GAP-style metric importance (writes metric_importance.json)
                    if vc_tgt.scores.shape[1] == vc_base.scores.shape[1]:
                        importance = self._compute_and_save_metric_importance(
                            vc_tgt,
                            vc_base,
                        )
                        if importance is not None:
                            context["metric_importance"] = [
                                m.to_dict() for m in importance
                            ]
                    else:
                        log.warning(
                            "CriticCohortAgent: cannot compute metric importance: "
                            "target.D=%d != baseline.D=%d",
                            vc_tgt.scores.shape[1],
                            vc_base.scores.shape[1],
                        )

                    # 6) Detailed separability stats (Cohen's d, entropy, etc.)
                    try:
                        metric_importance_report = self._compute_metric_separability(
                            vpm_base=vc_base.scores,
                            vpm_tgt=vc_tgt.scores,
                            metric_names=list(vc_tgt.metric_names),
                        )
                        context["visicalc_metric_importance"] = metric_importance_report

                        report_path = self.out_dir / "visicalc_metric_importance.json"
                        with report_path.open("w", encoding="utf-8") as f:
                            f.write(dumps_safe(metric_importance_report, indent=2))

                        log.info(
                            "VisiCalc: wrote metric separability report → %s",
                            report_path,
                        )
                    except Exception:
                        log.exception(
                            "CriticCohortAgent: _compute_metric_separability failed"
                        )


            # ------------------------------------------------------------------
            # Case B: single cohort only (no A/B comparison)
            # ------------------------------------------------------------------
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

                # Optional: expose episode features for downstream analysis
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

        # 1) Matrix + metric names via MetricMapper / visicalc_metric_keys
        vpm, metric_names, item_ids = self._build_vpm_and_metric_names(rows)

        # 2) Optionally filter by importance
        vpm, metric_names = self._maybe_filter_by_importance(vpm, metric_names)

        # 3) Choose frontier metric
        frontier_metric = self.visicalc_frontier_metric
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

        # 3) Build VisiCalc episode + report
        episode_id = f"{self.name}:{cohort_label}"

        vc = VisiCalc.from_matrix(
            episode_id=episode_id,
            scores=vpm,
            metric_names=metric_names,
            item_ids=item_ids,
            frontier_metric=frontier_metric,
            row_region_splits=self.visicalc_row_region_splits,
            frontier_low=self.visicalc_frontier_low,
            frontier_high=self.visicalc_frontier_high,
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

        return vc, metric_names
 
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

        
    def _build_vpm_and_metric_names(self, rows: List[dict]) -> tuple[np.ndarray, List[str], List[str]]:
        if not rows:
            raise ValueError("CriticCohortAgent: no rows provided")

        # UNION of metric columns across *all* rows
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
            raise ValueError("CriticCohortAgent: no metric columns found on any row")


        metric_names = self.metric_mapper.select_columns(union_cols) or union_cols

        # Optional preferred ordering
        kept = self.memory.metrics.get_kept_columns(self.run_id)
        if kept:
            kept_set = set(kept)
            # keep the DB order, but only those that actually exist in metric_names
            metric_names = [m for m in kept if m in set(metric_names)]
            # if the intersection is tiny (e.g., mis-match), fall back gracefully
            if len(metric_names) < 3:
                metric_names = self.metric_mapper.select_columns(union_cols) or union_cols

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
                "CriticCohortAgent: no rows with metrics_values after mapping "
                f"(non_empty={non_empty}, skipped={skipped}, "
                f"union_cols={len(union_cols)}, kept={len(metric_names)})"
            )

        X = np.asarray(matrix_rows, dtype=np.float32)
        return X, metric_names, item_ids

    # ------------------------------------------------------------------
    #  Top-left image helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize01(arr: np.ndarray) -> np.ndarray:
        """
        Safe [0, 1] normalization for visualization.
        Flat columns become 0.5 so they still show up neutrally.
        """
        arr = np.asarray(arr, dtype=np.float32)
        if arr.size == 0:
            return arr

        mn = float(arr.min())
        mx = float(arr.max())
        if mx <= mn:
            return np.full_like(arr, 0.5, dtype=np.float32)
        return (arr - mn) / (mx - mn)

    @staticmethod
    def _top_left_order_pair(
        tgt: np.ndarray,
        base: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute a *shared* row/col ordering that pushes the *difference*
        (targeted - baseline) into the top-left.

        Returns:
            (row_order, col_order) as index arrays.
        """
        tgt = np.asarray(tgt, dtype=np.float32)
        base = np.asarray(base, dtype=np.float32)
        if tgt.shape != base.shape:
            raise ValueError(
                f"_top_left_order_pair: shape mismatch tgt={tgt.shape}, base={base.shape}"
            )

        diff = tgt - base  # positive = where targeted beats baseline

        # Aggregate per-row / per-column "advantage"
        row_scores = diff.sum(axis=1)  # [N]
        col_scores = diff.sum(axis=0)  # [D]

        # Larger advantage → earlier (closer to top-left)
        row_order = np.argsort(-row_scores)
        col_order = np.argsort(-col_scores)

        return row_order, col_order

    def _save_vpm_heatmap(
        self,
        matrix: np.ndarray,
        out_path: Path,
        title: str,
        *,
        vmin: float = 0.0,
        vmax: float = 1.0,
        cmap: str = "magma",
    ) -> None:
        """
        Generic utility: save a VPM-like matrix as a heatmap PNG.
        """
        m = self._normalize01(matrix)

        out_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(m, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("metric index")
        ax.set_ylabel("item index")
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_path.as_posix(), dpi=160)
        plt.close(fig)

        log.info("VisiCalc: wrote VPM heatmap → %s", out_path)

    def _save_topleft_pair(
        self,
        vpm_tgt: np.ndarray,
        vpm_base: np.ndarray,
        run_dir: Path,
        *,
        prefix: str = "visicalc",
    ) -> None:
        """
        Given *normalized* VPM matrices for targeted and baseline cohorts,
        produce three images:

          - {prefix}_targeted_topleft.png
          - {prefix}_baseline_topleft.png
          - {prefix}_delta_topleft.png

        All three use the *same* row/column ordering derived from
        (targeted - baseline), so the top-left is literally:
           'where targeted wins hardest over baseline'.
        """
        if vpm_tgt.size == 0 or vpm_base.size == 0:
            log.warning("VisiCalc: _save_topleft_pair called with empty matrices; skipping")
            return

        # 0) Normalize each matrix for fair visual comparison
        tgt_norm = self._normalize01(vpm_tgt)
        base_norm = self._normalize01(vpm_base)

        if tgt_norm.shape != base_norm.shape:
            raise ValueError(
                f"_save_topleft_pair: shape mismatch tgt={tgt_norm.shape}, base={base_norm.shape}"
            )

        # 1) Shared top-left ordering based on (tgt - base)
        row_order, col_order = self._top_left_order_pair(tgt_norm, base_norm)

        tgt_tl = tgt_norm[row_order][:, col_order]
        base_tl = base_norm[row_order][:, col_order]
        diff_tl = tgt_tl - base_tl  # [-1, 1] ish after normalization

        # 2) Optional numeric summary: how much mass is in the top-left patch?
        h, w = tgt_tl.shape
        top_rows = max(1, int(0.25 * h))
        left_cols = max(1, int(0.25 * w))

        tgt_mass = float(tgt_tl[:top_rows, :left_cols].sum())
        base_mass = float(base_tl[:top_rows, :left_cols].sum())
        log.info(
            "VisiCalc TopLeft mass: targeted=%.4f baseline=%.4f delta=%.4f (rows=%d, cols=%d)",
            tgt_mass,
            base_mass,
            tgt_mass - base_mass,
            top_rows,
            left_cols,
        )

        # 3) Save images
        tgt_path = run_dir / f"{prefix}_targeted_topleft.png"
        base_path = run_dir / f"{prefix}_baseline_topleft.png"
        delta_path = run_dir / f"{prefix}_delta_topleft.png"

        self._save_vpm_heatmap(
            tgt_tl,
            tgt_path,
            title="Targeted (TopLeft-ordered)",
            vmin=0.0,
            vmax=1.0,
            cmap="magma",
        )
        self._save_vpm_heatmap(
            base_tl,
            base_path,
            title="Baseline (TopLeft-ordered)",
            vmin=0.0,
            vmax=1.0,
            cmap="magma",
        )

        # Δ image with diverging colormap
        # Note: we re-normalize to [-1, 1]-ish by clipping.
        diff_clip = np.clip(diff_tl, -1.0, 1.0)
        self._save_vpm_heatmap(
            diff_clip,
            delta_path,
            title="Targeted − Baseline (TopLeft-ordered)",
            vmin=-1.0,
            vmax=1.0,
            cmap="seismic",
        )


    def _render_ab_topleft_heatmaps(
        self,
        vpm_target: np.ndarray,
        vpm_baseline: np.ndarray,
        out_dir: Path,
        *,
        clip_percent: float = 0.01,
        corner_frac: float = 0.35,
    ) -> dict:
        """
        Save three heatmaps in `out_dir`:

          - visicalc_baseline_topleft.png
          - visicalc_targeted_topleft.png
          - visicalc_delta_topleft.png

        using a shared TopLeft ordering and shared intensity scaling.

        Returns a small dict with scalar stats (gain/loss/etc.).
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        T = np.asarray(vpm_target, dtype=np.float32)
        B = np.asarray(vpm_baseline, dtype=np.float32)

        # 1) Align shapes (rows) — we already have same cols
        n_rows = min(T.shape[0], B.shape[0])
        T = T[:n_rows]
        B = B[:n_rows]

        n_rows, n_cols = T.shape

        # 2) Shared scaling across BOTH matrices (this is the big fix)
        stacked = np.concatenate([T.reshape(-1), B.reshape(-1)])
        if stacked.size == 0:
            log.warning("VisiCalc TopLeft: empty matrices, skipping render")
            return {"status": "empty"}

        # Robust global min/max with optional percentile clipping
        lo, hi = np.quantile(stacked, [clip_percent, 1.0 - clip_percent])
        if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
            lo = float(stacked.min())
            hi = float(stacked.max())
        scale = max(hi - lo, 1e-8)

        Tn = np.clip((T - lo) / scale, 0.0, 1.0)
        Bn = np.clip((B - lo) / scale, 0.0, 1.0)

        # 3) TopLeft ordering based on **combined** energy
        row_score = Tn.sum(axis=1) + Bn.sum(axis=1)
        col_score = Tn.sum(axis=0) + Bn.sum(axis=0)

        row_order = np.argsort(row_score)[::-1]   # high → top
        col_order = np.argsort(col_score)[::-1]   # high → left

        T_sorted = Tn[row_order][:, col_order]
        B_sorted = Bn[row_order][:, col_order]

        # 4) Define "TopLeft" window for gain/loss stats
        tl_rows = max(1, int(n_rows * corner_frac))
        tl_cols = max(1, int(n_cols * corner_frac))

        T_tl = T_sorted[:tl_rows, :tl_cols]
        B_tl = B_sorted[:tl_rows, :tl_cols]

        delta_tl = T_tl - B_tl
        gain = float(np.maximum(delta_tl, 0.0).sum())
        loss = float(np.maximum(-delta_tl, 0.0).sum())
        total = gain + loss + 1e-8
        improvement_ratio = gain / total

        # 5) Full delta matrix (IMPORTANT: keep sign, no extra [0,1] renorm)
        delta = T_sorted - B_sorted
        max_abs = float(np.max(np.abs(delta))) or 1e-6
        delta_vis = np.clip(delta, -max_abs, max_abs)

        # 6) Baseline / Target heatmaps (shared [0,1] scale)
        plt.figure(figsize=(10, 6))
        plt.imshow(B_sorted, cmap="magma", aspect="auto", vmin=0.0, vmax=1.0)
        plt.colorbar()
        plt.title("Baseline (TopLeft-ordered)")
        plt.xlabel("metric index")
        plt.ylabel("item index")
        baseline_png = out_dir / "visicalc_baseline_topleft.png"
        plt.tight_layout()
        plt.savefig(baseline_png, dpi=160)
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.imshow(T_sorted, cmap="magma", aspect="auto", vmin=0.0, vmax=1.0)
        plt.colorbar()
        plt.title("Targeted (TopLeft-ordered)")
        plt.xlabel("metric index")
        plt.ylabel("item index")
        targeted_png = out_dir / "visicalc_targeted_topleft.png"
        plt.tight_layout()
        plt.savefig(targeted_png, dpi=160)
        plt.close()

        # 7) Delta heatmap with symmetric color range (no more all-red slab)
        plt.figure(figsize=(10, 6))
        plt.imshow(
            delta_vis,
            cmap="seismic",
            aspect="auto",
            vmin=-max_abs,
            vmax=+max_abs,
        )
        plt.colorbar()
        plt.title("Targeted − Baseline (TopLeft-ordered)")
        plt.xlabel("metric index")
        plt.ylabel("item index")
        delta_png = out_dir / "visicalc_delta_topleft.png"
        plt.tight_layout()
        plt.savefig(delta_png, dpi=160)
        plt.close()

        log.info(
            "VisiCalc A/B TopLeft: gain=%.4f loss=%.4f improvement_ratio=%.4f "
            "(top-left window %d×%d, lo=%.4g hi=%.4g)",
            gain,
            loss,
            improvement_ratio,
            tl_rows,
            tl_cols,
            lo,
            hi,
        )

        return {
            "status": "ok",
            "rows": int(n_rows),
            "cols": int(n_cols),
            "clip_percent": float(clip_percent),
            "corner_frac": float(corner_frac),
            "gain": gain,
            "loss": loss,
            "improvement_ratio": improvement_ratio,
            "top_left_rows": tl_rows,
            "top_left_cols": tl_cols,
            "scale_lo": float(lo),
            "scale_hi": float(hi),
            "paths": {
                "baseline_topleft_png": str(baseline_png),
                "targeted_topleft_png": str(targeted_png),
                "delta_topleft_png": str(delta_png),
            },
        }

    def _compute_metric_separability(
        self,
        vpm_base: np.ndarray,
        vpm_tgt: np.ndarray,
        metric_names: List[str],
    ) -> Dict[str, Any]:
        """
        Computes separability metrics (Cohen's d, mean diff) for each metric column
        between the targeted and baseline VPMs.
        """
        if vpm_base.shape != vpm_tgt.shape:
             # This should have been caught upstream, but is a safe check
            log.warning("VisiCalc: Cannot compute separability due to shape mismatch.")
            return {}

        n_rows_base, n_metrics = vpm_base.shape
        n_rows_tgt, _ = vpm_tgt.shape

        import math

        import numpy as np
        from scipy.stats import differential_entropy

        results = {}

        eps_var = 1e-8  # variance threshold to treat as "flat"

        for i, metric_name in enumerate(metric_names):
            base_col = np.asarray(vpm_base[:, i], dtype=np.float32)
            tgt_col = np.asarray(vpm_tgt[:, i], dtype=np.float32)

            # 1. Basic Stats
            mean_base = float(base_col.mean())
            mean_tgt = float(tgt_col.mean())
            std_base = float(base_col.std())
            std_tgt = float(tgt_col.std())

            # 2. Mean Difference (Targeted - Baseline)
            delta_mean = mean_tgt - mean_base

            # 3. Cohen's d (Effect Size)
            if n_rows_base + n_rows_tgt - 2 > 0:
                s_pooled = math.sqrt(
                    (
                        ((n_rows_base - 1) * std_base**2)
                        + ((n_rows_tgt - 1) * std_tgt**2)
                    )
                    / (n_rows_base + n_rows_tgt - 2)
                )
            else:
                s_pooled = 1e-8

            cohens_d = delta_mean / s_pooled if s_pooled != 0 else 0.0

            # 4. Robust differential entropy (skip flat / near-flat columns)
            var_base = float(base_col.var())
            var_tgt = float(tgt_col.var())

            if var_base < eps_var:
                entropy_base = float("nan")
            else:
                with np.errstate(divide="ignore", invalid="ignore"):
                    try:
                        entropy_base = float(differential_entropy(base_col))
                    except Exception:
                        entropy_base = float("nan")

            if var_tgt < eps_var:
                entropy_tgt = float("nan")
            else:
                with np.errstate(divide="ignore", invalid="ignore"):
                    try:
                        entropy_tgt = float(differential_entropy(tgt_col))
                    except Exception:
                        entropy_tgt = float("nan")

            # 5. Compile results for the metric
            results[metric_name] = {
                "mean_base": mean_base,
                "mean_tgt": mean_tgt,
                "delta_mean": delta_mean,
                "cohens_d": cohens_d,
                "std_base": std_base,
                "std_tgt": std_tgt,
                "entropy_base": entropy_base,
                "entropy_tgt": entropy_tgt,
                "delta_entropy": (
                    entropy_tgt - entropy_base
                    if not (math.isnan(entropy_tgt) or math.isnan(entropy_base))
                    else float("nan")
                ),
            }

        # Structure the output for easy consumption/saving
        metric_rankings = sorted(
            results.items(),
            key=lambda item: abs(item[1]["cohens_d"]),
            reverse=True,
        )

        return {
            "n_base": n_rows_base,
            "n_tgt": n_rows_tgt,
            "metrics_by_cohens_d": [
                {"metric": name, "stats": stats} for name, stats in metric_rankings
            ],
            "metrics_all": results,
        }
    
    # ------------------------------------------------------------------
    # Importance-based metric subset
    # ------------------------------------------------------------------
    def _load_important_metric_names(self) -> List[str]:
        """
        Load ordered metric names from the importance file.

        We assume the JSON is either:
          - a list of entries (string or dict with 'metric'/'name'/...)
          - OR a dict with one of:
                - 'metrics': [...]
                - 'metric_importance': [...]
        The *list order* is treated as importance ranking.
        """
        if self._important_metric_names is not None:
            return self._important_metric_names

        if not self.importance_enabled:
            self._important_metric_names = []
            return self._important_metric_names

        path = getattr(self, "importance_load_path", None)
        if not path:
            self._important_metric_names = []
            return self._important_metric_names

        path = Path(path)

        if not path.exists():
            self.logger.log(
                "VisiCalcImportanceFileMissing",
                {"path": str(path)},
            )
            self._important_metric_names = []
            return self._important_metric_names

        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            self.logger.log(
                "VisiCalcImportanceFileLoadError",
                {"path": str(path), "error": str(e)},
            )
            self._important_metric_names = []
            return self._important_metric_names

        # Normalize to a list (same as you already have)
        if isinstance(data, dict):
            if isinstance(data.get("metrics"), list):
                entries = data["metrics"]
            elif isinstance(data.get("metric_importance"), list):
                entries = data["metric_importance"]
            else:
                entries = list(data.values())
        elif isinstance(data, list):
            entries = data
        else:
            entries = []

        names: List[str] = []
        for entry in entries:
            if isinstance(entry, str):
                names.append(entry)
                continue
            if isinstance(entry, dict):
                name = (
                    entry.get("metric")
                    or entry.get("name")
                    or entry.get("metric_name")
                    or entry.get("key")
                )
                if name:
                    names.append(str(name))

        # Deduplicate while preserving order
        seen = set()
        ordered_unique: List[str] = []
        for n in names:
            if n in seen:
                continue
            seen.add(n)
            ordered_unique.append(n)

        # Apply top_k limit if configured
        if self.importance_top_k is not None:
            ordered_unique = ordered_unique[: self.importance_top_k]

        self._important_metric_names = ordered_unique

        self.logger.log(
            "VisiCalcImportanceLoaded",
            {
                "path": str(path),
                "enabled": self.importance_enabled,
                "top_k": self.importance_top_k,
                "num_loaded": len(ordered_unique),
                "examples": ordered_unique[:10],
            },
        )
        return self._important_metric_names

    def _maybe_filter_by_importance(
        self,
        vpm: np.ndarray,
        metric_names: List[str],
    ) -> tuple[np.ndarray, List[str]]:
        """
        Restrict VPM columns to the important metrics if importance_filter is enabled.

        - Keeps the column *order* consistent with the importance file.
        - Only retains metrics that exist in metric_names.
        - If the intersection is tiny (< 3), we fall back to the full set.
        """
        if not self.importance_enabled:
            return vpm, metric_names

        important = self._load_important_metric_names()
        if not important:
            # nothing to filter by
            return vpm, metric_names

        name_to_idx = {name: i for i, name in enumerate(metric_names)}
        selected_indices: List[int] = []
        selected_names: List[str] = []

        for name in important:
            idx = name_to_idx.get(name)
            if idx is None:
                continue
            selected_indices.append(idx)
            selected_names.append(name)

        if not selected_indices:
            self.logger.log(
                "VisiCalcImportanceNoOverlap",
                {
                    "num_metrics": len(metric_names),
                    "num_important": len(important),
                },
            )
            return vpm, metric_names

        # If we end up with something extremely small, it's safer to keep full matrix
        if len(selected_indices) < 3:
            self.logger.log(
                "VisiCalcImportanceTooFew",
                {
                    "num_selected": len(selected_indices),
                    "num_metrics": len(metric_names),
                },
            )
            return vpm, metric_names

        vpm_reduced = vpm[:, selected_indices]

        self.logger.log(
            "VisiCalcImportanceApplied",
            {
                "num_original_metrics": len(metric_names),
                "num_selected_metrics": len(selected_names),
                "selected_example": selected_names[:10],
            },
        )

        return vpm_reduced, selected_names

    # ------------------------------------------------------------------
    # Metric importance: compute once per A/B run and persist to JSON
    # ------------------------------------------------------------------
    def _compute_and_save_metric_importance(
        self,
        vc_tgt,
        vc_base,
    ) -> Optional[list]:
        """
        GAP-style metric importance:
          - uses *current* VPM matrices (vc_tgt.scores / vc_base.scores)
          - writes a single JSON file at self.importance_path
          - returns the importance list (or None on failure)
        """
        if vc_tgt.scores.shape[1] != vc_base.scores.shape[1]:
            log.warning(
                "CriticCohortAgent: cannot compute metric importance: target.D=%d != baseline.D=%d",
                vc_tgt.scores.shape[1],
                vc_base.scores.shape[1],
            )
            return None

        X_t = vc_tgt.scores
        X_b = vc_base.scores
        metric_names = list(vc_tgt.metric_names)

        try:
            importance = compute_metric_importance(
                X_target=X_t,
                X_baseline=X_b,
                metric_names=metric_names,
                top_k=self.visicalc_top_k_metrics or None,
                min_effect=self.visicalc_min_effect or 0.0,
            )

            # Single canonical importance file
            save_metric_importance_json(importance, self.importance_save_path)

            top_show = importance[:10]
            log.info(
                "VisiCalc metric importance (top %d): %s",
                len(top_show),
                ", ".join(
                    f"{m.name}[d={m.cohen_d:+.3f}, auc={m.auc:.3f}]"
                    for m in top_show
                ),
            )

            # Optional: stash in context if the caller wants it
            return importance

        except Exception as e:
            log.exception("CriticCohortAgent: metric importance computation failed: %s", e)
            return None

    def _save_vpm_matrix_csv(self, matrix: np.ndarray, metric_names: list[str], item_ids: list[str], out_path: Path) -> None:
        """
        Write a dense CSV:
            scorable_id, <metric_0>, <metric_1>, ...
            <id_0>,     <val>,      <val>, ...
        """
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["scorable_id", *metric_names])
            for ridx in range(matrix.shape[0]):
                w.writerow([item_ids[ridx], *[float(x) for x in matrix[ridx]]])
        self.logger.log("VisiCalcMatrixCSVSaved", {"path": str(out_path), "rows": int(matrix.shape[0]), "cols": int(matrix.shape[1])})

    def _save_ab_npz_dataset(self, vpm_base: np.ndarray, vpm_tgt: np.ndarray, metric_names: list[str], out_path: Path) -> None:
        """
        Concatenate baseline & targeted matrices → X, and make binary labels y.
        baseline → 0, targeted → 1
        """
        X = np.concatenate([vpm_base, vpm_tgt], axis=0).astype(np.float32, copy=False)
        y = np.concatenate([np.zeros((vpm_base.shape[0],), dtype=np.int64),
                            np.ones((vpm_tgt.shape[0],), dtype=np.int64)], axis=0)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Also stash names so training can reconstruct feature names if desired
        np.savez(out_path.as_posix(), X=X, y=y, metric_names=np.array(metric_names, dtype=object))
        self.logger.log("VisiCalcABDatasetSaved", {"path": str(out_path), "X_shape": list(X.shape), "y_len": int(y.shape[0])})

def project_to_kept(X, metric_names, kept):
    """
    Project matrix X (n, d_in) with columns 'metric_names' to the order in 'kept'.
    Missing columns fill with 0.0; extra columns are dropped.
    Returns: X_proj (n, len(kept)), kept (same list back for convenience)
    """
    import numpy as np
    if not kept:
        return X, metric_names
    idx = {n: i for i, n in enumerate(metric_names)}
    out = np.zeros((X.shape[0], len(kept)), dtype=X.dtype)
    for j, name in enumerate(kept):
        i = idx.get(name)
        if i is not None:
            out[:, j] = X[:, i]
    return out, list(kept)
