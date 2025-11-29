# stephanie/components/critic/agents/critic_cohort.py
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from stephanie.agents.base_agent import BaseAgent
from stephanie.components.critic.reports.frontier_reporter import FrontierReporter
from stephanie.components.critic.services.frontier_intelligence import FrontierIntelligence
from stephanie.scoring.metrics.metric_importance import (
    compute_metric_importance, save_metric_importance_json)
from stephanie.scoring.metrics.metric_mapping import MetricMapper
from stephanie.scoring.metrics.scorable_processor import ScorableProcessor
from stephanie.scoring.metrics.frontier_lens import (FrontierLens, normalize_scores,
                                                graph_quality_from_report)

from stephanie.utils.json_sanitize import dumps_safe
from stephanie.components.critic.reports.frontier_lens_viz import (
    render_frontier_lens_figure,
)

from stephanie.utils.vpm_utils import ensure_chw_u8

from stephanie.components.critic.reports.cohort_reporter import (
    CriticCohortReporter,
    normalize01,
    save_topleft_vpm_triptych,
    top_left_order_pair,
    save_vpm_matrix_csv,
    save_ab_npz_dataset,
    save_ab_episode_features,
    save_vpm_heatmap,
    render_ab_topleft_heatmaps,
    compute_metric_separability,
)

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
            "Tiny.faithfulness.attr.scm.aggregate01",
        )
        self.visicalc_frontier_low: float = float(vis_cfg.get("frontier_low", 0.25))
        self.visicalc_frontier_high: float = float(vis_cfg.get("frontier_high", 0.75))

        # How many row regions (coarse bands) to split into
        self.visicalc_row_region_splits: int = int(
            vis_cfg.get("row_region_splits", 4)
        )

        # Output directory + file names
        self.out_dir: Path = Path(vis_cfg.get("out_dir", "runs/critic")) / self.run_id
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
        fi_cfg = cfg.get("frontier_intelligence", {})  # or whatever your hydra path is
        if fi_cfg.get("enabled", True):
            self.frontier_intel = FrontierIntelligence(
                cfg=fi_cfg,
                memory=self.memory,
                container=self.container,
                logger=self.logger,
                run_id=self.run_id,
            )

        self.selected_frontier_metric: Optional[str] = None
        self.reporter = FrontierReporter()

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
        self.visicalc_per_metric_normalize = bool(vis_cfg.get("per_metric_normalize", True))

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

                frontier_metric = self.visicalc_frontier_metric  # default from config
                if self.frontier_intel is not None and rows_tgt and rows_base:
                    try:
                        X, metric_names_fi, y = self._build_frontier_intel_dataset(
                            rows_base=rows_base,
                            rows_tgt=rows_tgt,
                        )
                        if len(np.unique(y)) >= 2 and X.shape[1] > 0:
                            frontier_metric = self.frontier_intel.select_frontier_metric(
                                metric_matrix=X,
                                metric_names=metric_names_fi,
                                y=y,
                                run_id=self.run_id,
                                fallback=frontier_metric or metric_names_fi[0],
                            )
                            self.logger.log(
                                "FrontierMetricSelected",
                                {"run_id": self.run_id, "frontier_metric": frontier_metric},
                            )
                            svm_val = None
                            svm_cfg = self.cfg.get("svm_validation", {})
                            if self.frontier_intel is not None and svm_cfg.get("enabled", False):
                                try:
                                    svm_val = self.frontier_intel.run_svm_frontier_validation(
                                        metric_matrix=X,         # (N, M) baseline + target
                                        y=y,                     # 0/1 labels
                                        metric_names=metric_names_fi,
                                        C=float(svm_cfg.get("C", 1.0)),
                                    )
                                # Stash under the same sub-context the reporter reads
                                except Exception as e:
                                    log.exception("CriticCohortAgent: SVM frontier validation failed %s", str(e))
                                    svm_val = {"enabled": False, "reason": "exception"}
                                context["svm_frontier_validation"] = svm_val

                    except Exception:
                        log.exception("CriticCohortAgent: FrontierIntelligence selection failed; using configured frontier_metric")


                # ---- Targeted cohort ----
                if rows_tgt:
                    vc_tgt, metric_keys_tgt = self._compute_frontier_lens_for_rows(
                        rows_tgt,
                        cohort_label="targeted",
                        frontier_metric=frontier_metric,
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
                    vc_base, metric_keys_base = self._compute_frontier_lens_for_rows(
                        rows_base,
                        cohort_label="baseline",
                        frontier_metric=frontier_metric,
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
                    context["visicalc_ab_topleft"] = topleft_res

                    topleft_png = self.out_dir / "visicalc_ab_topleft.png"
                    save_topleft_vpm_triptych(
                        vpm_base=vpm_base,
                        vpm_tgt=vpm_tgt,
                        out_path=topleft_png,
                    )
                    context["save_topleft_vpm_triptych"] = topleft_png

                    log.info(
                        "VisiCalc A/B TopLeft diff: gain=%.4f loss=%.4f improvement_ratio=%.4f",
                        topleft_res["gain"],
                        topleft_res["loss"],
                        topleft_res["improvement_ratio"],
                    )

                    log.info(
                        "VisiCalc A/B delta: frontier_metric=%r "
                        "global_mean_delta=%.4f frontier_frac_delta=%.4f",
                        diff["frontier_metric"],
                        diff["global_delta"]["mean"],
                        diff["global_delta"]["frontier_frac"],
                    )

                    # 2A-1) Save full matrices as CSV
                    save_vpm_matrix_csv(vpm_tgt,  list(vc_tgt.metric_names),  list(vc_tgt.item_ids),  self.out_dir / "visicalc_targeted_matrix.csv")
                    save_vpm_matrix_csv(vpm_base, list(vc_base.metric_names), list(vc_base.item_ids), self.out_dir / "visicalc_baseline_matrix.csv")

                    # 2A-2) Save combined NPZ for tiny_critic trainer (labels inside)
                    # NOTE: metric_names must match (you already ensure same D before importance/diff)
                    save_ab_npz_dataset(vpm_base, vpm_tgt, list(vc_tgt.metric_names), self.out_dir / "visicalc_ab_dataset.npz")

                    # 2A-3) Episode-level features for tiny_critic
                    save_ab_episode_features(
                        vc_base,
                        vc_tgt,
                        out_path=self.out_dir / "visicalc_ab_episode_features.npz",
                    )


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
                        render_ab_topleft_heatmaps(
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
                        metric_importance_report = compute_metric_separability(
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
                vc, used_metric_keys = self._compute_frontier_lens_for_rows(
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
            log.exception("Critic cohort analysis failed")

        # ------------------------------------------------------------------
        # Final reporting pass: build a compact critic cohort report
        # ------------------------------------------------------------------
        try:
            reporter = CriticCohortReporter(
                run_id=self.run_id,
                out_dir=self.out_dir,
                input_key=self.input_key,
                output_key=self.output_key,
                logger=log,
                top_k_metrics=10,
            )
            reporter(context)
        except Exception as e:
            # Do not fail the whole pipeline if reporting has issues
            log.exception("[CriticCohortAgent] reporter failed: %s", e)

        return context



    def _compute_frontier_lens_for_rows(
        self,
        rows: List[dict],
        cohort_label: str,
        frontier_metric: Optional[str] = None,
    ):
        """
        Build a FrontierLens instance from canonical feature rows.

        Expected row schema (from ScorableProcessor):
        - 'scorable_id'
        - 'metrics_columns': List[str]
        - 'metrics_values':  List[float]

        Returns:
            (FrontierLens, used_metric_keys)
        """
        if not rows:
            raise ValueError("FrontierLens: no rows to analyze")

        # 1) Build raw matrix + metric names + item ids
        first = rows[0]
        metric_names = list(first.get("metrics_columns") or [])
        if not metric_names:
            raise ValueError("FrontierLens: first row missing 'metrics_columns'")

        matrix_rows: List[List[float]] = []
        item_ids: List[str] = []

        for row in rows:
            vals = row.get("metrics_values")
            if not vals:
                continue

            cols = row.get("metrics_columns") or metric_names
            mapping = {k: float(v) for k, v in zip(cols, vals)}
            vec = [float(mapping.get(name, 0.0)) for name in metric_names]

            matrix_rows.append(vec)
            item_ids.append(str(row.get("scorable_id", "unknown")))

        if not matrix_rows:
            raise ValueError("FrontierLens: no valid rows with metrics_values")

        vpm = np.asarray(matrix_rows, dtype=np.float32)

        # 2) Optional per-metric normalization
        if self.visicalc_per_metric_normalize:
            vpm = normalize_scores(vpm).astype(np.float32)

        # 3) Frontier metric
        frontier_metric = frontier_metric or self.visicalc_frontier_metric
        if frontier_metric and frontier_metric not in metric_names:
            log.warning(
                "FrontierLens: requested frontier_metric=%r not in metric_names; "
                "falling back to first metric=%r",
                frontier_metric,
                metric_names[0],
            )
            frontier_metric = None

        if not frontier_metric:
            frontier_metric = metric_names[0]

        # 4) Build episode id + meta
        episode_id = f"{self.name}:{cohort_label}"

        vc = FrontierLens.from_matrix(
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

        img_path = self.out_dir / f"frontier_lens_{cohort_label or 'cohort'}.png"
        render_frontier_lens_figure(vc, img_path)
        # optionally stash path in context/report
        log.info("FrontierLens: saved figure → %s", img_path)

        log.info(
            "FrontierLens: built episode=%r frontier_metric=%r matrix_shape=%s "
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
        channel: int = 0,
        top_frac: float = 0.5,
        left_frac: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Compare baseline vs targeted VPMs in the *TopLeft* high-intensity region.

        - Robust to different base/target shapes.
        - Uses the minimum overlapping region so shapes always match.
        - Returns gain/loss and improvement_ratio suitable for reporting.
        """
        base = ensure_chw_u8(vpm_base)  # (C, H, W), uint8
        tgt  = ensure_chw_u8(vpm_tgt)

        if base.ndim != 3 or tgt.ndim != 3:
            raise ValueError(
                f"VPMs must be CHW images; got base={base.shape}, tgt={tgt.shape}"
            )

        c_b, h_b, w_b = base.shape
        c_t, h_t, w_t = tgt.shape

        if channel < 0 or channel >= min(c_b, c_t):
            raise ValueError(
                f"channel={channel} out of range for base={base.shape}, tgt={tgt.shape}"
            )

        # --- Align shapes by overlapping region ---
        H = min(h_b, h_t)
        W = min(w_b, w_t)

        # Make sure we always keep at least 1 pixel
        top_frac  = float(top_frac)
        left_frac = float(left_frac)
        h_cut = max(1, int(H * top_frac))
        w_cut = max(1, int(W * left_frac))

        # --- Extract TopLeft regions on the chosen channel ---
        tl_base = base[channel, :h_cut, :w_cut].astype(np.float32) / 255.0
        tl_tgt  = tgt[channel,  :h_cut, :w_cut].astype(np.float32) / 255.0

        assert tl_base.shape == tl_tgt.shape, (
            f"TopLeft shapes still mismatch base={tl_base.shape}, tgt={tl_tgt.shape}"
        )

        # --- Compute gain/loss in that region ---
        delta = tl_tgt - tl_base
        gain  = float(delta[delta > 0].sum())
        loss  = float(-delta[delta < 0].sum())  # make positive

        denom = gain + loss
        improvement_ratio = gain / denom if denom > 0 else 0.5

        status = "ok"
        if denom == 0:
            status = "flat"
        elif improvement_ratio < 0.5:
            status = "worse"

        return {
            "gain": gain,
            "loss": loss,
            "improvement_ratio": improvement_ratio,
            "status": status,
            "shape": {
                "height": h_cut,
                "width": w_cut,
            },
            "channel": channel,
            "top_frac": top_frac,
            "left_frac": left_frac,
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
        tgt_norm = normalize01(vpm_tgt)
        base_norm = normalize01(vpm_base)

        if tgt_norm.shape != base_norm.shape:
            raise ValueError(
                f"_save_topleft_pair: shape mismatch tgt={tgt_norm.shape}, base={base_norm.shape}"
            )

        # 1) Shared top-left ordering based on (tgt - base)
        row_order, col_order = top_left_order_pair(tgt_norm, base_norm)

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

        save_vpm_heatmap(
            tgt_tl,
            tgt_path,
            title="Targeted (TopLeft-ordered)",
            vmin=0.0,
            vmax=1.0,
            cmap="magma",
        )
        save_vpm_heatmap(
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
        save_vpm_heatmap(
            diff_clip,
            delta_path,
            title="Targeted − Baseline (TopLeft-ordered)",
            vmin=-1.0,
            vmax=1.0,
            cmap="seismic",
        )



    
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
            log.log(
                "VisiCalcImportanceFileMissing",
                {"path": str(path)},
            )
            self._important_metric_names = []
            return self._important_metric_names

        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            log.log(
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

        log.log(
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
            log.log(
                "VisiCalcImportanceNoOverlap",
                {
                    "num_metrics": len(metric_names),
                    "num_important": len(important),
                },
            )
            return vpm, metric_names

        # If we end up with something extremely small, it's safer to keep full matrix
        if len(selected_indices) < 3:
            log.log(
                "VisiCalcImportanceTooFew",
                {
                    "num_selected": len(selected_indices),
                    "num_metrics": len(metric_names),
                },
            )
            return vpm, metric_names

        vpm_reduced = vpm[:, selected_indices]

        log.log(
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

            self._update_core_metrics_from_importance(importance)

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



    def _update_core_metrics_from_importance(self, importance) -> None:
        """
        Given a list[MetricImportance], extract the top-N names and
        store them via MetricStore so future runs can focus on them.
        """
        if not importance:
            return

        # Use the same top_k that drove the visual importance
        k = self.visicalc_top_k_metrics or len(importance)
        top = importance[:k]

        core_names = [m.name for m in top]

        try:
            # Persist to MetricStore so _build_vpm_and_metric_names can pick them up
            self.memory.metrics.set_kept_columns(self.run_id, core_names)
            log.info(f"run_id: {self.run_id}, count: {len(core_names)}")
        except Exception as e:
            log.exception("CriticCohortAgent: failed to persist core metrics to MetricStore: %s", e)

    def _select_frontier_and_bands(
        self,
        metric_names: List[str],
        vpm: np.ndarray,
    ) -> tuple[str, float, float]:
        """
        Central hook for frontier selection.
        For now: static config values.
        Later: call FrontierIntelligence + dynamic band computation.
        """
        frontier_metric = self.visicalc_frontier_metric
        low = self.visicalc_frontier_low
        high = self.visicalc_frontier_high
        return frontier_metric, low, high

    def _rows_to_matrix(
        self,
        rows: List[dict],
        metric_names: List[str],
    ) -> np.ndarray:
        """Project canonical rows into a dense (N, M) matrix using metric_names order."""
        matrix_rows: List[List[float]] = []

        for row in rows:
            vals = row.get("metrics_values")
            if not vals:
                continue

            cols = row.get("metrics_columns") or metric_names
            mapping = {k: float(v) for k, v in zip(cols, vals)}
            vec = [float(mapping.get(name, 0.0)) for name in metric_names]
            matrix_rows.append(vec)

        if not matrix_rows:
            raise ValueError("FrontierIntelligence: no valid rows with metrics_values")

        return np.asarray(matrix_rows, dtype=np.float32)

    def _build_frontier_intel_dataset(
        self,
        rows_base: List[dict],
        rows_tgt: List[dict],
    ) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """
        Build (X, metric_names, y) for FrontierIntelligence:

        - X: (N, M) scores for baseline + target
        - metric_names: common metric ordering
        - y: 0 for baseline, 1 for target
        """
        # 1) Infer global metric_names from both cohorts
        names_set = set()
        for r in rows_base + rows_tgt:
            cols = r.get("metrics_columns") or []
            names_set.update(cols)
        if not names_set:
            raise ValueError("FrontierIntelligence: no metric names found in rows")

        metric_names = sorted(names_set)

        # 2) Project into common matrix
        X_base = self._rows_to_matrix(rows_base, metric_names)
        X_tgt  = self._rows_to_matrix(rows_tgt, metric_names)

        # 3) Build labels
        y_base = np.zeros(X_base.shape[0], dtype=np.int8)
        y_tgt  = np.ones(X_tgt.shape[0], dtype=np.int8)

        X = np.vstack([X_base, X_tgt])
        y = np.concatenate([y_base, y_tgt])

        return X, metric_names, y


