# stephanie/components/gap/orchestrator.py
"""
GAP Analysis Orchestrator - Core pipeline coordinator for model comparison.

Two modes:
1) A/B-from-Nexus mode (preferred when available):
   - Resolve baseline/targeted Nexus run directories
   - Load their manifests and extract per-item metrics tables
   - Treat Baseline vs Targeted as two 'models' and run Analysis→Significance→Calibration→Report

2) Legacy full pipeline:
   - Data → Scoring → Analysis → Significance → Calibration → Report
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")  # safe on headless
import matplotlib.pyplot as plt
from PIL import Image

from stephanie.components.gap.io.data_retriever import (DataRetriever,
                                                        RetrieverConfig)
from stephanie.components.gap.models import GapConfig
from stephanie.components.gap.processors.ab_frontier import (
    ABFrontierConfig, ABFrontierProcessor)
from stephanie.components.gap.processors.analysis import AnalysisProcessor
from stephanie.components.gap.processors.calibration import \
    CalibrationProcessor
from stephanie.components.gap.processors.report import ReportBuilder
from stephanie.components.gap.processors.scoring import ScoringProcessor
from stephanie.components.gap.processors.significance import (
    SignificanceConfig, SignificanceProcessor)
from stephanie.core.manifest import ManifestManager
from stephanie.services.eg_visual_service import EGVisualService
from stephanie.services.epistemic_guard_service import EpistemicGuardService
from stephanie.services.risk_predictor_service import RiskPredictorService
from stephanie.services.scm_service import SCMService
from stephanie.services.storage_service import StorageService
from stephanie.utils.progress_mixin import ProgressMixin

log = logging.getLogger(__name__)


# ---------- small utils ----------
def _safe_json_load(p: Path) -> Optional[dict]:
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log.warning("Failed to read %s: %s", p, e)
        return None


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


class GapAnalysisOrchestrator(ProgressMixin):
    """
    Main orchestrator for GAP analysis pipeline execution.
    """

    def __init__(self, cfg: GapConfig, container, logger, memory=None):
        self.cfg = cfg
        self.container = container
        self.logger = logger
        self.memory = memory

        # ---- Processor Initialization ----
        self.scoring_processor = ScoringProcessor(self.cfg, container, logger)
        self.analysis_processor = AnalysisProcessor(self.cfg, container, logger)
        self.calibration_processor = CalibrationProcessor(self.cfg, container, logger)

        self.base_dir = getattr(self.cfg, "base_dir", Path("./data/gap_runs"))

        # Register services (idempotent)
        try:
            self.container.register(
                name="storage",
                factory=lambda: StorageService(),
                dependencies=[],
                init_args={"base_dir": str(self.cfg.base_dir), "logger": self.logger},
            )

            container.register(
                name="scm_service",
                factory=lambda: SCMService(),
                dependencies=[],
                init_args={"config": cfg.scm, "logger": logger},
            )

            container.register(
                name="ep_guard",
                factory=lambda: EpistemicGuardService(self.container),
                dependencies=[],
                init_args={
                    "config": {
                        "out_dir": str(cfg.base_dir / "eg"),
                        "thresholds": (0.2, 0.6),
                    },
                    "logger": logger,
                },
            )
            container.register(
                name="eg_visual",
                factory=lambda: EGVisualService(),
                dependencies=[],
                init_args={"config": {"out_dir": str(cfg.base_dir / "eg" / "img")}, "logger": logger},
            )

            self.container.register(
                name="risk_predictor",
                factory=lambda: RiskPredictorService(cfg=cfg, memory=memory, container=container, logger=logger),
                dependencies=["?memcube"],
                init_args={
                    "config": {
                        "bundle_path": "./models/risk/bundle.joblib",
                        "default_domains": ("science", "history", "geography", "tech", "general"),
                        "calib_ttl_s": 3600,
                        "fallback_low": 0.20,
                        "fallback_high": 0.60,
                    }
                },
            )
        except ValueError:
            pass  # already registered

        self.storage = self.container.get("storage")
        self.manifest_manager = ManifestManager(self.storage)

        # Significance
        self.significance_processor = SignificanceProcessor(
            SignificanceConfig(
                n_nulls=getattr(self.cfg, "n_nulls", 100),
                n_bootstrap=getattr(self.cfg, "n_bootstrap", 50),
                random_seed=getattr(self.cfg, "random_seed", 42),
                max_betti_dim=1,
            ),
            logger=self.logger,
        )

        # Data retriever (legacy mode)
        safe_limit = self.cfg.per_dim_cap if self.cfg.per_dim_cap is not None else 10**9
        self.retriever = DataRetriever(
            container,
            logger,
            retriever_cfg=RetrieverConfig(source="memory", limit=safe_limit),
        )
        self.ab_frontier = ABFrontierProcessor(self.cfg, container, logger)
        
        self._init_progress(container, logger)

    # ---------- Nexus A/B resolution ----------
    def _resolve_input_dirs(self, context: Dict[str, Any]) -> Tuple[Path, Path]:
        """
        Priority:
        1) explicit Nexus outputs via context paths:
           - ab_targeted_run_dir, ab_baseline_run_dir
        2) run IDs under cfg.out_dir (default: runs/nexus_vpm)
           - ab_targeted_run_id, ab_baseline_run_id Hello
        3) derive from pipeline_run_id: <out>/<id>-{targeted|baseline}
        4) scan <out> for latest *-targeted/*-baseline that have run_metrics.json
        """
        # 1) paths
        tgt_dir = context.get("ab_targeted_run_dir")
        base_dir = context.get("ab_baseline_run_dir")
        if tgt_dir and base_dir:
            td = Path(tgt_dir).resolve()
            bd = Path(base_dir).resolve()
            if (td / "manifest.json").exists() and (bd / "manifest.json").exists():
                self.logger.log("GapInputResolved", {"mode": "context_paths", "targeted": td.as_posix(), "baseline": bd.as_posix()})
                return td, bd

        nexus_root = Path(getattr(self.cfg, "out_dir", "runs/nexus_vpm")).resolve()

        # 2) IDs
        tgt_id = context.get("ab_targeted_run_id")
        base_id = context.get("ab_baseline_run_id")
        if tgt_id and base_id:
            td = (nexus_root / str(tgt_id)).resolve()
            bd = (nexus_root / str(base_id)).resolve()
            if (td / "manifest.json").exists() and (bd / "manifest.json").exists():
                self.logger.log("GapInputResolved", {"mode": "context_ids", "targeted": td.as_posix(), "baseline": bd.as_posix()})
                return td, bd

        # 3) pipeline id
        pr = context.get("pipeline_run_id") or context.get("run_id") or ""
        if pr:
            td = (nexus_root / f"{pr}-targeted").resolve()
            bd = (nexus_root / f"{pr}-baseline").resolve()
            if (td / "manifest.json").exists() and (bd / "manifest.json").exists():
                self.logger.log("GapInputResolved", {"mode": "pipeline_id", "targeted": td.as_posix(), "baseline": bd.as_posix()})
                return td, bd

        # 4) scan latest
        def _pick(kind: str) -> Optional[Path]:
            cands: List[Path] = []
            if nexus_root.exists():
                for p in nexus_root.iterdir():
                    if p.is_dir() and p.name.endswith(f"-{kind}") and (p / "manifest.json").exists():
                        cands.append(p)
            if not cands:
                return None
            cands.sort(key=lambda p: (p / "manifest.json").stat().st_mtime, reverse=True)
            return cands[0]

        td = _pick("targeted")
        bd = _pick("baseline")
        if td and bd:
            self.logger.log("GapInputResolved", {"mode": "scan", "targeted": td.as_posix(), "baseline": bd.as_posix()})
            return td.resolve(), bd.resolve()

        raise FileNotFoundError(
            "Could not resolve Nexus run directories for GAP. "
            "Provide ab_targeted_run_dir and ab_baseline_run_dir in context, "
            "or ensure runs exist under runs/nexus_vpm/*-{targeted|baseline}."
        )

    def _load_manifest(self, run_dir: Path) -> Dict[str, Any]:
        man = _safe_json_load(run_dir / "manifest.json") or {}
        return man

    # ---------- table extraction helpers (from manifests) ----------
    def _extract_metrics_matrix(
        self, manifest: Dict[str, Any]
    ) -> Tuple[List[str], List[List[float]], List[str]]:
        """
        Returns (columns, matrix, item_ids)
        - columns: list[str] from the first item with metrics_columns
        - matrix: list[list[float]] row per item (aligned to columns where possible, zeros if missing)
        - item_ids: the item_id sequence used
        """
        items = manifest.get("items", []) or []
        # establish canonical columns order
        cols: List[str] = []
        for it in items:
            c = it.get("metrics_columns") or []
            if c:
                cols = list(c)
                break

        # index mapping for each item (tolerate per-item ordering)
        def row_for(it: Dict[str, Any], cols_ref: List[str]) -> List[float]:
            c = it.get("metrics_columns") or []
            v = it.get("metrics_values") or []
            if not cols_ref:
                # no columns anywhere → empty row
                return []
            if c and list(c) == cols_ref:
                try:
                    return [float(x) for x in v]
                except Exception:
                    return [0.0 for _ in cols_ref]
            # remap using position map
            pos = {k: i for i, k in enumerate(c)}
            out = []
            for k in cols_ref:
                if k in pos and pos[k] < len(v):
                    try:
                        out.append(float(v[pos[k]]))
                    except Exception:
                        out.append(0.0)
                else:
                    out.append(0.0)
            return out

        mat: List[List[float]] = []
        ids: List[str] = []
        for it in items:
            ids.append(str(it.get("item_id") or it.get("scorable_id") or ""))
            mat.append(row_for(it, cols))
        return cols, mat, ids

    # ---------- main entry ----------
    async def execute_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the GAP analysis:

        - If Nexus A/B runs are available (paths or IDs or inferable), perform an A/B
          comparison using the precomputed Nexus metrics (no rescoring). We build a
          'score_out' payload that treats Baseline vs Targeted as two models.
        - Otherwise, fall back to the legacy full pipeline:
          Data → Scoring → Analysis → Significance → Calibration → Report.
        """
        run_id = context.get("pipeline_run_id") or context.get("run_id") or "gap_run"
        dataset_name = context.get("dataset", "unknown")

        # allow forcing legacy mode
        if context.get("gap_skip_ab_mode"):
            return await self._run_full_pipeline(context, run_id, dataset_name)

        # Try A/B-from-Nexus first
        try:
            targeted_dir, baseline_dir = self._resolve_input_dirs(context)
        except Exception as e:
            self.logger.log("GapABInputUnresolvable", {
                "error": str(e),
                "run_id": run_id,
            })
            return await self._run_full_pipeline(context, run_id, dataset_name)

        # --- A/B mode: load manifests & build score_out ---
        self.pstart(task=f"ab:load:{run_id}", total=2, meta={"targeted": targeted_dir.as_posix(), "baseline": baseline_dir.as_posix()})
        tg_manifest = self._load_manifest(targeted_dir)
        bl_manifest = self._load_manifest(baseline_dir)
        self.ptick(task=f"ab:load:{run_id}", done=2, total=2)
        self.pdone(task=f"ab:load:{run_id}")

        tg_cols, tg_mat, tg_ids = self._extract_metrics_matrix(tg_manifest)
        bl_cols, bl_mat, bl_ids = self._extract_metrics_matrix(bl_manifest)

        # Reconcile columns (simple union preserving baseline-first then targeted)
        cols = self._reconcile_columns(bl_cols, tg_cols)
        bl_mat = self._reindex_columns(bl_cols, bl_mat, cols)
        tg_mat = self._reindex_columns(tg_cols, tg_mat, cols)

        # Persist input snapshot for audit
        ab_out_dir = self.cfg.base_dir / f"{run_id}-ab_gap"
        ab_out_dir.mkdir(parents=True, exist_ok=True)
        with (ab_out_dir / "ab_input.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "baseline_dir": baseline_dir.as_posix(),
                    "targeted_dir": targeted_dir.as_posix(),
                    "columns": cols,
                    "n_baseline": len(bl_mat),
                    "n_targeted": len(tg_mat),
                },
                f,
                indent=2,
            )

        # Build score_out in the shape analysis expects.
        # We treat BASELINE as alias_a and TARGETED as alias_b (order matters for deltas).
        score_out = {
            "mode": "ab_from_nexus",
            "alias_a": "BASELINE",
            "alias_b": "TARGETED",
            "columns": cols,
            "items_a": bl_ids,
            "items_b": tg_ids,
            "table_a": bl_mat,  # list[list[float]] baseline rows
            "table_b": tg_mat,  # list[list[float]] targeted rows
            # Optional metadata:
            "source": {
                "baseline_dir": baseline_dir.as_posix(),
                "targeted_dir": targeted_dir.as_posix(),
            },
        }

        # ---- Analysis pipeline (no rescoring) ----
        # Prefer an AB-aware method if your AnalysisProcessor provides it; otherwise
        # the default execute_analysis reads the above shape.
        if hasattr(self.analysis_processor, "execute_analysis_ab"):
            analysis_out = await self.analysis_processor.execute_analysis_ab(
                score_out, run_id, manifest=None
            )
            frontier_out_dir = (self.cfg.base_dir / f"{run_id}-ab_gap" / "frontier")
            frontiers = self.ab_frontier.build_frontiers(
                run_id=run_id,
                score_out=score_out,
                bl_manifest=bl_manifest,
                tg_manifest=tg_manifest,
                out_dir=frontier_out_dir,
                # optional: pin the exact pairs you want on the PNGs:
                # metric_pairs=[("clarity", "faithfulness"), ("reasoning", "knowledge")],
                min_pair_sim=0.35,
            )
        else:
            analysis_out = await self.analysis_processor.execute_analysis(
                score_out, run_id, manifest=None
            )

        # Significance
        try:
            significance_out = await self.significance_processor.run(
                run_id, base_dir=self.cfg.base_dir
            )
        except Exception as e:
            self.logger.log("SignificanceStageError", {"run_id": run_id, "error": str(e)})
            significance_out = {"status": "error", "error": str(e)}

        # Calibration (name the arms clearly)
        calib_out = await self.calibration_processor.execute_calibration(
            analysis_out,
            run_id,
            alias_a="BASELINE",
            alias_b="TARGETED",
        )

        # Report
        reporter = ReportBuilder(self.cfg, self.container, self.logger)
        analysis_for_report = {**analysis_out, "significance": significance_out}
        report_out = await reporter.build(run_id, analysis_for_report, score_out)

        # After resolving directories:
        t_run = self._load_nexus_run(targeted_dir)
        b_run = self._load_nexus_run(baseline_dir)

        # Existing summary (kept as-is)
        ab_summary = self._run_ab_gap(context, baseline_dir=baseline_dir, targeted_dir=targeted_dir)

        # New: frontier images directly from manifest metrics
        frontiers_dir = Path(ab_summary["summary_path"]).parent / "frontiers"
        frontiers = self._render_ab_frontiers(
            run_id=context.get("pipeline_run_id", "ab_gap"),
            baseline_run=b_run,
            targeted_run=t_run,
            out_dir=frontiers_dir,
        )

        # Merge and return
        ab_summary["frontiers"] = frontiers
        self.logger.log("ABFrontiersBuilt", {
            "dir": frontiers_dir.as_posix(),
            "n_baseline": frontiers.get("n_baseline", 0),
            "n_targeted": frontiers.get("n_targeted", 0),
            "columns": frontiers.get("columns_used", []),
        })


        result = {
            "mode": "ab_from_nexus",
            "run_id": run_id,
            "analysis": analysis_out,
            "significance": significance_out,
            "calibration": calib_out,
            "report": report_out,
            "ab_inputs": {
                "baseline_dir": baseline_dir.as_posix(),
                "targeted_dir": targeted_dir.as_posix(),
                "columns": cols,
                "n_baseline": len(bl_mat),
                "n_targeted": len(tg_mat),
            },
            "frontier": frontiers, 
        }
        self.logger.log("GapCompleted", {"mode": "ab_from_nexus"})
        context["ab_runs"] = result
        context[self.output_key] = result
        context["frontiers"] = frontiers
        return context

    # ---------- legacy full pipeline ----------
    async def _run_full_pipeline(self, context: Dict[str, Any], run_id: str, dataset_name: str) -> Dict[str, Any]:
        m = self.manifest_manager.start_run(
            run_id=run_id,
            dataset=dataset_name,
            models={
                "hrm": self.cfg.hrm_scorers[0] if self.cfg.hrm_scorers else "hrm",
                "tiny": self.cfg.tiny_scorers[0] if self.cfg.tiny_scorers else "tiny",
            },
        )
        self.manifest_manager.attach_dimensions(run_id, self.cfg.dimensions)

        self.pstart(task=f"data:{run_id}", total=1, meta={"dataset": dataset_name})
        triples_by_dim = await self.retriever.get_triples_by_dimension(
            self.cfg.dimensions, memory=self.memory, limit=self.retriever.cfg.limit
        )
        self.pdone(task=f"data:{run_id}", extra={"dims": len(triples_by_dim)})

        score_out = await self.scoring_processor.execute_scoring(
            triples_by_dim, run_id, manifest=m
        )
        analysis_out = await self.analysis_processor.execute_analysis(
            score_out, run_id, manifest=m
        )
        try:
            significance_out = await self.significance_processor.run(
                run_id, base_dir=self.cfg.base_dir
            )
        except Exception as e:
            self.logger.log("SignificanceStageError", {"run_id": run_id, "error": str(e)})
            significance_out = {"status": "error", "error": str(e)}

        alias_a = score_out.get("alias_a", "HRM")
        alias_b = score_out.get("alias_b", "Tiny")
        calib_out = await self.calibration_processor.execute_calibration(
            analysis_out, run_id, alias_a=alias_a, alias_b=alias_b
        )

        reporter = ReportBuilder(self.cfg, self.container, self.logger)
        analysis_for_report = {**analysis_out, "significance": significance_out}
        report_out = await reporter.build(run_id, analysis_for_report, score_out)

        result = {
            "mode": "full_pipeline",
            "run_id": run_id,
            "score": score_out,
            "analysis": analysis_out,
            "significance": significance_out,
            "calibration": calib_out,
            "report": report_out,
            "manifest": m.to_dict(),
        }
        self.manifest_manager.finish_run(run_id, result)
        self.logger.log("GapCompleted", {"mode": "full_pipeline"})
        return result

    # ---------- column reconciliation helpers ----------
    def _reconcile_columns(self, a: List[str], b: List[str]) -> List[str]:
        seen, out = set(), []
        for k in a + [x for x in b if x not in a]:
            if k not in seen:
                seen.add(k)
                out.append(k)
        return out

    def _reindex_columns(
        self, cols_old: List[str], mat: List[List[float]], cols_new: List[str]
    ) -> List[List[float]]:
        if not cols_old:
            return [[0.0 for _ in cols_new] for _ in mat]
        pos = {k: i for i, k in enumerate(cols_old)}
        out = []
        for row in mat:
            out.append([
                float(row[pos[c]]) if c in pos and pos[c] < len(row) else 0.0
                for c in cols_new
            ])
        return out

    # ---------- optional storage helper ----------
    def _ensure_storage(self, base_dir: str) -> StorageService | None:
        st = getattr(self.container, "storage", None)
        if st:
            return st
        st = StorageService()
        st.initialize(base_dir=base_dir)
        try:
            setattr(self.container, "storage", st)
        except Exception:
            pass
        return st


    def _extract_metrics_matrix(self, manifest_dict: dict) -> Tuple[List[str], np.ndarray, List[dict]]:
        """
        Returns (columns, matrix[n_items, n_cols], items_meta) from manifest.items[*].metrics_vector.
        Falls back to metrics_columns/metrics_values if needed.
        """
        items = manifest_dict.get("items") or []
        rows = []
        cols_union = set()
        # gather keys union for robustness
        for it in items:
            mv = (it.get("metrics_vector") or {})
            if mv:
                cols_union.update(mv.keys())
        cols = sorted(cols_union)
        # if no dict vectors, try columns+values
        if not cols and items:
            # look for first with metrics_columns
            for it in items:
                cols = it.get("metrics_columns") or []
                if cols: break

        for it in items:
            if cols and it.get("metrics_vector"):
                mv = it["metrics_vector"]
                row = [ _safe_float(mv.get(c, 0.0)) for c in cols ]
                rows.append(row)
            elif it.get("metrics_columns") and it.get("metrics_values"):
                c2 = it["metrics_columns"]; v2 = it["metrics_values"]
                # map onto current cols ordering
                m = {c:_safe_float(v2[i]) for i,c in enumerate(c2)}
                row = [ _safe_float(m.get(c, 0.0)) for c in cols ]
                rows.append(row)
            else:
                rows.append([0.0]*len(cols))

        M = np.array(rows, dtype=float) if rows else np.zeros((0,0), dtype=float)
        return cols, M, items

    def _build_tile_atlas(self, items: List[dict], out_path: Path, *, max_each: int = 20, tile_size: int = 128) -> Optional[str]:
        """
        Build a simple grid image from available VPM tiles (items[*].vpm_png), up to max_each.
        Returns path or None.
        """
        paths = []
        for it in items:
            p = it.get("vpm_png")
            if p and Path(p).exists():
                paths.append(p)
            if len(paths) >= max_each:
                break
        if not paths:
            return None

        imgs = []
        for p in paths:
            try:
                im = Image.open(p).convert("RGB").resize((tile_size, tile_size))
                imgs.append(im)
            except Exception:
                pass
        if not imgs:
            return None

        # make a roughly square grid
        n = len(imgs)
        cols = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))
        W = cols*tile_size; H = rows*tile_size
        canvas = Image.new("RGB", (W, H), (12,12,12))
        i = 0
        for r in range(rows):
            for c in range(cols):
                if i >= n: break
                canvas.paste(imgs[i], (c*tile_size, r*tile_size))
                i += 1
        canvas.save(out_path)
        return out_path.as_posix()

    def _render_ab_frontiers(self, *, run_id: str, baseline_run: dict, targeted_run: dict, out_dir: Path) -> Dict[str, Any]:
        """
        Render frontier visuals from manifest metrics (no re-scoring).
        Outputs PNGs into out_dir and returns a dict of paths.
        """
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1) matrices
        bl_cols, bl_M, bl_items = self._extract_metrics_matrix(baseline_run.get("manifest", {}))
        tg_cols, tg_M, tg_items = self._extract_metrics_matrix(targeted_run.get("manifest", {}))
        cols = _common_columns(bl_cols, tg_cols)
        if not cols:
            # fall back to whichever has columns
            cols = bl_cols or tg_cols

        bl = _reindex_columns(bl_cols, bl_M, cols) if bl_M.size else np.zeros((0, len(cols)))
        tg = _reindex_columns(tg_cols, tg_M, cols) if tg_M.size else np.zeros((0, len(cols)))

        # 2) z-score by cohort then stack – simple, avoids cohort size skew
        Z_bl = _zscore(bl) if bl.size else bl
        Z_tg = _zscore(tg) if tg.size else tg
        Z_all = np.vstack([Z_bl, Z_tg]) if (Z_bl.size or Z_tg.size) else np.zeros((0, len(cols)))
        labels = np.array([0]*len(Z_bl) + [1]*len(Z_tg), dtype=int)  # 0=baseline, 1=targeted

        paths = {}

        # 3) PCA scatter
        if Z_all.shape[0] >= 2 and Z_all.shape[1] >= 2:
            X2 = _pca2(Z_all)
            plt.figure(figsize=(8,6))
            if (labels==0).any():
                plt.scatter(X2[labels==0,0], X2[labels==0,1], s=14, alpha=0.6, label="Baseline", edgecolors="none")
            if (labels==1).any():
                plt.scatter(X2[labels==1,0], X2[labels==1,1], s=14, alpha=0.6, label="Targeted", edgecolors="none")
            plt.title(f"Frontier (PCA) — A/B from manifest metrics\n{run_id}")
            plt.xlabel("PC1"); plt.ylabel("PC2"); plt.legend()
            p_scatter = out_dir / "pca_scatter.png"
            plt.tight_layout(); plt.savefig(p_scatter, dpi=140); plt.close()
            paths["pca_scatter"] = p_scatter.as_posix()

        # 4) Top-|Δ| bar chart
        if bl.shape[0] >= 1 and tg.shape[0] >= 1 and len(cols) > 0:
            mu_b = bl.mean(axis=0) if bl.size else np.zeros(len(cols))
            mu_t = tg.mean(axis=0) if tg.size else np.zeros(len(cols))
            delta = mu_t - mu_b
            order = np.argsort(np.abs(delta))[::-1][:20]
            plt.figure(figsize=(10,6))
            plt.bar(range(len(order)), delta[order])
            plt.xticks(range(len(order)), [cols[i] for i in order], rotation=60, ha="right")
            plt.ylabel("Δ mean (Targeted - Baseline)")
            plt.title("Top metric shifts (from manifest metrics)")
            plt.tight_layout()
            p_bar = out_dir / "metric_delta_bar.png"
            plt.savefig(p_bar, dpi=140); plt.close()
            paths["metric_delta_bar"] = p_bar.as_posix()

        # 5) Tile atlases for visual feel (re-uses vpm_png from each cohort)
        base_atlas = self._build_tile_atlas(bl_items, out_dir / "baseline_tile_atlas.png", max_each=20, tile_size=128)
        tgt_atlas  = self._build_tile_atlas(tg_items, out_dir / "targeted_tile_atlas.png", max_each=20, tile_size=128)
        if base_atlas: paths["baseline_tile_atlas"] = base_atlas
        if tgt_atlas:  paths["targeted_tile_atlas"] = tgt_atlas

        # Include the original graphs for convenience if present
        bg = (Path(baseline_run["dir"]) / "graph.html")
        tg = (Path(targeted_run["dir"]) / "graph.html")
        if bg.exists(): paths["baseline_graph"] = bg.as_posix()
        if tg.exists(): paths["targeted_graph"] = tg.as_posix()

        # meta
        paths["columns_used"] = cols
        paths["n_baseline"] = int(bl.shape[0])
        paths["n_targeted"] = int(tg.shape[0])
        return paths


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return 0.0

def _write_csv(path: Path, header: List[str], rows: List[List[Any]]):
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")

def _zscore(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, ddof=0, keepdims=True)
    sd[sd == 0.0] = 1.0
    return (X - mu) / sd

def _pca2(X: np.ndarray) -> np.ndarray:
    # 2D PCA via SVD on mean-centered data
    Xc = X - X.mean(axis=0, keepdims=True)
    # U S Vt where rows of Vt are principal axes
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    W = Vt[:2].T  # (d,2)
    return Xc @ W  # (n,2)

def _common_columns(cols_a: List[str], cols_b: List[str]) -> List[str]:
    ca = set(cols_a); cb = set(cols_b)
    common = [c for c in cols_a if c in cb]
    return common

def _reindex_columns(cols_src: List[str], M_src: np.ndarray, cols_dst: List[str]) -> np.ndarray:
    idx = {c:i for i,c in enumerate(cols_src)}
    out = np.zeros((M_src.shape[0], len(cols_dst)), dtype=float)
    for j, c in enumerate(cols_dst):
        i = idx.get(c, None)
        if i is not None:
            out[:, j] = M_src[:, i]
    return out

