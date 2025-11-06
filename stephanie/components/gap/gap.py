# stephanie/components/gap/gap.py
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import math

from stephanie.agents.base_agent import BaseAgent
from stephanie.components.gap.models import (
    EgBadgeConfig, EgBaselineConfig,
    EgConfig, EgMemConfig, EgModelConfig,
    EgRenderConfig, EgStreams, EgThresholds,
    GapConfig,
)
from stephanie.components.gap.orchestrator import GapAnalysisOrchestrator

log = logging.getLogger(__name__)


class GapAgent(BaseAgent):
    """
    GAP Agent — now defaults to **A/B runs comparison** (baseline vs targeted)
    when Nexus provides:
        - context['ab_baseline_run_dir']
        - context['ab_targeted_run_dir']

    If those aren't present, it falls back to the legacy HRM↔Tiny orchestrator flow.
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self._config = self._load_config(cfg)
        self._orchestrator = GapAnalysisOrchestrator(self._config, container, logger, memory=memory)

    # -------------------------
    # Config
    # -------------------------
    def _load_config(self, raw_config: Dict[str, Any]) -> GapConfig:
        """
        Keep your existing typed GapConfig; no schema changes required.
        """
        return GapConfig(
            dimensions=list(raw_config.get(
                "dimensions",
                ["reasoning", "knowledge", "clarity", "faithfulness", "coverage"]
            )),
            hrm_scorers=list(raw_config.get("hrm_scorers", ["hrm"])),
            tiny_scorers=list(raw_config.get("tiny_scorers", ["tiny"])),
            out_dir=Path(raw_config.get("out_dir", "data/gap_runs/vpm")),
            base_dir=Path(raw_config.get("gap_base_dir", "data/gap_runs")),
            interleave=bool(raw_config.get("interleave", False)),
            progress_log_every=int(raw_config.get("progress_log_every", 25)),
            dedupe_policy=raw_config.get("dedupe_policy", "first_wins"),
            per_dim_cap=raw_config.get("per_dim_cap", 100),
            eg=_merge_eg(raw_config),
        )

    # -------------------------
    # Entry
    # -------------------------
    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        If Nexus A/B run dirs are present, compute the **Gap between runs** using
        their `manifest.json` metrics. Otherwise, defer to legacy orchestrator.
        """
        run_id = context.get("pipeline_run_id", "unknown")

        baseline_dir = context.get("ab_baseline_run_dir")
        targeted_dir = context.get("ab_targeted_run_dir")

        if baseline_dir and targeted_dir:
            # --- A/B runs mode (default) ---
            try:
                self.logger.log("GapABRunsStarted", {
                    "run_id": run_id,
                    "baseline_dir": str(baseline_dir),
                    "targeted_dir": str(targeted_dir),
                })
                result = self._run_ab_gap(context, Path(baseline_dir), Path(targeted_dir))
                context[self.output_key] = result

                self.logger.log("GapABRunsCompleted", {
                    "run_id": run_id,
                    "summary": result.get("summary_path"),
                    "delta_csv": result.get("delta_csv_path"),
                })
                return context

            except Exception as e:
                self.logger.log("GapABRunsError", {
                    "run_id": run_id,
                    "error": str(e),
                })
                # fall back to legacy orchestrator if desired
                # raise  # or continue to legacy below
                pass

        # --- Legacy HRM↔Tiny mode ---
        try:
            self.logger.log("GapAnalysisStartedLegacy", {
                "run_id": run_id,
                "dimensions": self._config.dimensions,
                "hrm_scorers": self._config.hrm_scorers,
                "tiny_scorers": self._config.tiny_scorers
            })
            result = await self._orchestrator.execute_analysis(context)
            context[self.output_key] = result

            self.logger.log("GapAnalysisCompletedLegacy", {
                "run_id": run_id,
                "result_keys": list(result.keys()) if result else [],
                "artifacts_generated": len(result.get("artifacts", [])) if result else 0
            })
            return context

        except Exception as e:
            self.logger.log("GapAnalysisError", {
                "error": str(e),
                "run_id": run_id,
                "config": asdict(self._config) if hasattr(self, '_config') else None
            })
            raise

    # -------------------------
    # A/B Runs Implementation
    # -------------------------
    def _run_ab_gap(self, context: Dict[str, Any], baseline_dir: Path, targeted_dir: Path) -> Dict[str, Any]:
        """
        Compare **baseline vs targeted** Nexus runs by reading their manifest.json files
        and computing per-metric deltas & simple effect sizes.
        """
        out_root = self._config.out_dir
        out_root.mkdir(parents=True, exist_ok=True)

        run_id = context.get("pipeline_run_id", "ab_gap")
        out_dir = out_root / f"{run_id}-ab_gap"
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1) Load manifests
        bl_manifest = self._load_manifest_json(baseline_dir)
        tg_manifest = self._load_manifest_json(targeted_dir)

        # 2) Extract metric tables (columns must be consistent across items)
        bl_cols, bl_mat = self._extract_metrics_matrix(bl_manifest)
        tg_cols, tg_mat = self._extract_metrics_matrix(tg_manifest)

        # Robustness: ensure same ordering/columns
        cols = self._reconcile_columns(bl_cols, tg_cols)
        bl_mat = self._reindex_columns(bl_cols, bl_mat, cols)
        tg_mat = self._reindex_columns(tg_cols, tg_mat, cols)

        # 3) Aggregate per-metric stats and deltas
        agg_baseline = self._aggregate(bl_mat)  # {metric: {"mean":..., "std":..., "n":...}}
        agg_targeted = self._aggregate(tg_mat)

        delta_rows: List[Dict[str, Any]] = []
        for m in cols:
            b = agg_baseline.get(m, {"mean": None, "std": None, "n": 0})
            t = agg_targeted.get(m, {"mean": None, "std": None, "n": 0})
            d_mean = _safe_float(t["mean"]) - _safe_float(b["mean"])
            eff = self._cohens_d(
                mean1=_safe_float(t["mean"]), std1=_safe_float(t["std"]), n1=t["n"],
                mean2=_safe_float(b["mean"]), std2=_safe_float(b["std"]), n2=b["n"]
            )
            delta_rows.append({
                "metric": m,
                "baseline_mean": _safe_float(b["mean"]),
                "targeted_mean": _safe_float(t["mean"]),
                "delta": d_mean,
                "effect_size_d": eff,
                "baseline_std": _safe_float(b["std"]),
                "targeted_std": _safe_float(t["std"]),
                "n_baseline": b["n"], "n_targeted": t["n"],
            })

        # 4) Persist artifacts
        summary = {
            "mode": "ab_runs",
            "run_id": run_id,
            "baseline_dir": str(baseline_dir),
            "targeted_dir": str(targeted_dir),
            "columns": cols,
            "aggregate": {
                "baseline": agg_baseline,
                "targeted": agg_targeted,
            },
            "top_deltas": sorted(delta_rows, key=lambda r: abs(r["delta"]), reverse=True)[:20],
        }

        summary_path = out_dir / "ab_gap_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        delta_csv = out_dir / "ab_gap_delta.csv"
        _write_csv(delta_csv, ["metric","baseline_mean","targeted_mean","delta","effect_size_d","baseline_std","targeted_std","n_baseline","n_targeted"],
                   [[r["metric"], r["baseline_mean"], r["targeted_mean"], r["delta"], r["effect_size_d"], r["baseline_std"], r["targeted_std"], r["n_baseline"], r["n_targeted"]]
                    for r in delta_rows])

        # 5) Return result handle
        return {
            "run_id": run_id,
            "mode": "ab_runs",
            "summary_path": str(summary_path),
            "delta_csv_path": str(delta_csv),
            # Hand through useful viewer artifacts from Nexus if present:
            "baseline_graph": str((baseline_dir / "graph.html").as_posix()),
            "targeted_graph": str((targeted_dir / "graph.html").as_posix()),
            "baseline_frames": str((baseline_dir / "frames.json").as_posix()) if (baseline_dir / "frames.json").exists() else None,
            "targeted_frames": str((targeted_dir / "frames.json").as_posix()) if (targeted_dir / "frames.json").exists() else None,
        }

    # -------------------------
    # Helpers (A/B)
    # -------------------------
    def _load_manifest_json(self, run_dir: Path) -> Dict[str, Any]:
        """
        Expect a manifest at <run_dir>/manifest.json as produced by NexusInlineAgent single-run.
        If the A/B subset export didn't write a manifest, try to lift item metrics from item folders
        if you adopt that layout later. For now, require manifest.json (matches your current runs).
        """
        mpath = run_dir / "manifest.json"
        if not mpath.exists():
            raise FileNotFoundError(f"manifest.json not found in {run_dir}")
        return json.loads(mpath.read_text(encoding="utf-8"))

    def _extract_metrics_matrix(self, manifest: Dict[str, Any]) -> Tuple[List[str], List[List[float]]]:
        """
        Pull a consistent matrix from manifest items:
          - Prefer item['metrics_columns'] + item['metrics_values'] (already aligned)
        """
        items = manifest.get("items") or []
        if not items:
            return [], []

        # find the first non-empty columns
        cols: List[str] = []
        for it in items:
            cols = list(it.get("metrics_columns") or [])
            if cols:
                break

        if not cols:
            return [], []

        mat: List[List[float]] = []
        for it in items:
            vals = it.get("metrics_values") or []
            # ensure length; pad/truncate safely
            row = [float(vals[i]) if i < len(vals) else float("nan") for i in range(len(cols))]
            mat.append(row)
        return cols, mat

    def _reconcile_columns(self, a: List[str], b: List[str]) -> List[str]:
        """
        Build a deterministic union with a-first ordering, then append b-only.
        """
        seen = set()
        out: List[str] = []
        for c in a:
            if c not in seen:
                seen.add(c); out.append(c)
        for c in b:
            if c not in seen:
                seen.add(c); out.append(c)
        return out

    def _reindex_columns(self, cols_old: List[str], mat: List[List[float]], cols_new: List[str]) -> List[List[float]]:
        """
        Map an old matrix to a new column order (pad with NaN if missing).
        """
        idx = {c: i for i, c in enumerate(cols_old)}
        out: List[List[float]] = []
        for row in mat:
            out.append([float(row[idx[c]]) if c in idx and idx[c] < len(row) else float("nan") for c in cols_new])
        return out

    def _aggregate(self, mat: List[List[float]]) -> Dict[str, Dict[str, float]]:
        """
        Column-wise mean/std (ignoring NaNs). Returns {metric: {"mean":..., "std":..., "n":...}}
        """
        if not mat:
            return {}
        import math
        import statistics

        # transpose with padding already handled
        n_rows = len(mat)
        n_cols = len(mat[0])
        agg: Dict[str, Dict[str, float]] = {}
        # we don't know column names here; caller will join with names
        for j in range(n_cols):
            col = [r[j] for r in mat if j < len(r)]
            col = [x for x in col if not (isinstance(x, float) and math.isnan(x))]
            n = len(col)
            if n == 0:
                mean, std = float("nan"), float("nan")
            else:
                mean = sum(col) / n
                std = statistics.pstdev(col) if n > 1 else 0.0
            agg[j] = {"mean": mean, "std": std, "n": n}

        # Convert index keys to metric names in caller after we compute union
        # For convenience, transform here when caller passes names
        # We'll remap outside (kept here for clarity)
        return {str(j): agg[j] for j in range(n_cols)}

    def _cohens_d(self, *, mean1, std1, n1, mean2, std2, n2) -> float:
        """
        Simple pooled SD Cohen's d (targeted vs baseline). Handles small-n safely.
        """
        try:
            n1 = int(n1 or 0); n2 = int(n2 or 0)
            if n1 < 1 or n2 < 1:
                return float("nan")
            # pooled variance
            v1 = (std1 or 0.0) ** 2
            v2 = (std2 or 0.0) ** 2
            if n1 + n2 < 3:
                return float("nan")
            s_p = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)) if (n1 + n2 - 2) > 0 else 0.0
            if s_p == 0.0:
                return 0.0
            return (mean1 - mean2) / s_p
        except Exception:
            return float("nan")


# -------------------------
# Utilities
# -------------------------
def _merge_eg(raw: dict) -> EgConfig:
    eg = raw.get("eg", {}) or {}
    return EgConfig(
        enabled=eg.get("enabled", True),
        badge=EgBadgeConfig(**eg.get("badge", {})),
        render=EgRenderConfig(**{**asdict(EgRenderConfig()), **eg.get("render", {})}),
        thresholds=EgThresholds(**eg.get("thresholds", {})),
        streams=EgStreams(**eg.get("streams", {})),
        mem=EgMemConfig(**eg.get("mem", {})),
        models=EgModelConfig(**{**asdict(EgModelConfig()), **eg.get("models", {})}),
        baseline=EgBaselineConfig(**eg.get("baseline", {})),
    )


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _write_csv(path: Path, headers: List[str], rows: List[List[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for r in rows:
            f.write(",".join(_csv_cell(v) for v in r) + "\n")


def _csv_cell(v: Any) -> str:
    s = "" if v is None else str(v)
    # rudimentary CSV escaping
    if any(ch in s for ch in [",", "\"", "\n", "\r"]):
        s = "\"" + s.replace("\"", "\"\"") + "\""
    return s
