# stephanie/components/gap/gap.py
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from stephanie.agents.base_agent import BaseAgent
from stephanie.components.gap.models import (EgBadgeConfig, EgBaselineConfig,
                                             EgConfig, EgMemConfig,
                                             EgModelConfig, EgRenderConfig,
                                             EgStreams, EgThresholds,
                                             GapConfig)
from stephanie.components.gap.orchestrator import GapAnalysisOrchestrator

log = logging.getLogger(__name__)


class GapAgent(BaseAgent):
    """
    GAP Agent — now does two things in one run:
      1) Fast A/B delta over two Nexus runs (baseline vs targeted)
      2) Full GAP pipeline via the orchestrator (scoring → analysis → significance → calibration → report)
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self._config = self._load_config(cfg)
        self._orchestrator = GapAnalysisOrchestrator(
            self._config, container, logger, memory=memory
        )

    def _load_config(self, raw_config: Dict[str, Any]) -> GapConfig:
        return GapConfig(
            dimensions=list(raw_config.get(
                "dimensions",
                ["reasoning", "knowledge", "clarity", "faithfulness", "coverage"]
            )),
            hrm_scorers=list(raw_config.get("hrm_scorers", ["hrm"])),
            tiny_scorers=list(raw_config.get("tiny_scorers", ["tiny"])),
            out_dir=Path(raw_config.get("out_dir", "runs/nexus_vpm")),
            base_dir=Path(raw_config.get("gap_base_dir", "data/gap_runs")),
            interleave=bool(raw_config.get("interleave", False)),
            progress_log_every=int(raw_config.get("progress_log_every", 25)),
            dedupe_policy=raw_config.get("dedupe_policy", "first_wins"),
            per_dim_cap=raw_config.get("per_dim_cap", 100),
            eg=_merge_eg(raw_config),
        )

    # -------------------------
    # A/B Runs Implementation
    # -------------------------
    def _run_ab_gap(self, context: Dict[str, Any], baseline_dir: Path, targeted_dir: Path) -> Dict[str, Any]:
        """
        Compare **baseline vs targeted** Nexus runs by reading their manifest.json files
        and computing per-metric deltas & simple effect sizes.
        """
        out_root = self._config.base_dir
        out_root.mkdir(parents=True, exist_ok=True)

        run_id = context.get("pipeline_run_id", "ab_gap")
        out_dir = baseline_dir.parent / f"{run_id}-ab_gap"
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1) Load manifests
        bl_manifest = self._load_manifest_json(baseline_dir)
        tg_manifest = self._load_manifest_json(targeted_dir)

        # 2) Extract metric tables (columns must be consistent across items)
        bl_cols, bl_mat = self._extract_metrics_matrix(bl_manifest)
        tg_cols, tg_mat = self._extract_metrics_matrix(tg_manifest)

        # Ensure same column order
        cols = self._reconcile_columns(bl_cols, tg_cols)
        bl_mat = self._reindex_columns(bl_cols, bl_mat, cols)
        tg_mat = self._reindex_columns(tg_cols, tg_mat, cols)

        # 3) Aggregate per-metric stats and deltas
        agg_baseline = self._aggregate(bl_mat)
        agg_targeted = self._aggregate(tg_mat)

        delta_rows: List[Dict[str, Any]] = []
        for m in cols:
            b = agg_baseline.get(m, {"mean": None, "std": None, "n": 0})
            t = agg_targeted.get(m, {"mean": None, "std": None, "n": 0})
            d_mean = self._safe_float(t["mean"]) - self._safe_float(b["mean"])
            eff = self._cohens_d(
                mean1=self._safe_float(t["mean"]), std1=self._safe_float(t["std"]), n1=t["n"],
                mean2=self._safe_float(b["mean"]), std2=self._safe_float(b["std"]), n2=b["n"]
            )
            delta_rows.append({
                "metric": m,
                "baseline_mean": self._safe_float(b["mean"]),
                "targeted_mean": self._safe_float(t["mean"]),
                "delta": d_mean,
                "effect_size_d": eff,
                "baseline_std": self._safe_float(b["std"]),
                "targeted_std": self._safe_float(t["std"]),
                "n_baseline": b["n"], "n_targeted": t["n"],
            })

        # 4) Persist artifacts
        summary = {
            "mode": "ab_runs",
            "run_id": run_id,
            "baseline_dir": str(baseline_dir),
            "targeted_dir": str(targeted_dir),
            "columns": cols,
            "aggregate": {"baseline": agg_baseline, "targeted": agg_targeted},
            "top_deltas": sorted(delta_rows, key=lambda r: abs(r["delta"]), reverse=True)[:20],
        }

        summary_path = out_dir / "ab_gap_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        delta_csv = out_dir / "ab_gap_delta.csv"
        self._write_csv(
            delta_csv,
            ["metric","baseline_mean","targeted_mean","delta","effect_size_d","baseline_std","targeted_std","n_baseline","n_targeted"],
            [[r["metric"], r["baseline_mean"], r["targeted_mean"], r["delta"], r["effect_size_d"], r["baseline_std"], r["targeted_std"], r["n_baseline"], r["n_targeted"]] for r in delta_rows],
        )

        return {
            "run_id": run_id,
            "mode": "ab_runs",
            "summary_path": str(summary_path),
            "delta_csv_path": str(delta_csv),
            "baseline_graph": str((baseline_dir / "graph.html").as_posix()),
            "targeted_graph": str((targeted_dir / "graph.html").as_posix()),
            "baseline_frames": str((baseline_dir / "frames.json").as_posix()) if (baseline_dir / "frames.json").exists() else None,
            "targeted_frames": str((targeted_dir / "frames.json").as_posix()) if (targeted_dir / "frames.json").exists() else None,
        }

    # -------------------------
    # Public entrypoint
    # -------------------------
    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        1) Try a fast A/B delta over two Nexus runs (if resolvable)
        2) Then run the full GAP pipeline (force legacy path)
        """
        self.logger.log("GapAnalysisStarted", {
            "run_id": context.get("pipeline_run_id", "unknown"),
            "dimensions": self._config.dimensions,
            "hrm_scorers": self._config.hrm_scorers,
            "tiny_scorers": self._config.tiny_scorers,
        })

        ab_out: Optional[Dict[str, Any]] = None

        # Attempt A/B summary first (non-fatal if not resolvable)
        try:
            td, bd = self._orchestrator._resolve_input_dirs(context)  # reuse orchestrator logic
            ab_out = self._run_ab_gap(context, baseline_dir=bd, targeted_dir=td)
            self.logger.log("GapABSummaryCompleted", {
                "baseline": bd.as_posix(),
                "targeted": td.as_posix(),
                "summary": ab_out.get("summary_path"),
            })
        except Exception as e:
            self.logger.log("GapABSummarySkipped", {"reason": str(e)})

        # Now run full GAP (force full pipeline, skipping orchestrator's AB fast-path)
        # try:
        #     full_out = await self._orchestrator.execute_analysis({**context, "gap_skip_ab_mode": False})
        # except Exception as e:
        #     self.logger.log("GapAnalysisError", {
        #         "error": str(e),
        #         "run_id": context.get("pipeline_run_id", "unknown"),
        #         "config": asdict(self._config) if hasattr(self, "_config") else None,
        #     })
        #     raise

        # Aggregate results
        result = {
            "ab_runs": ab_out,            # may be None if not resolvable
            # "full": full_out,             # always present if no exception
        }
        context[self.output_key] = result

        self.logger.log("GapAnalysisCompleted", {
            "run_id": context.get("pipeline_run_id", "unknown"),
            "has_ab": bool(ab_out),
            # "full_mode": full_out.get("mode"),
        })
        return context

    # -------------------------
    # Helpers for A/B summary
    # -------------------------
    def _load_manifest_json(self, run_dir: Path) -> Dict[str, Any]:
        p = run_dir / "manifest.json"
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            raise FileNotFoundError(f"Missing or unreadable manifest at {p}: {e}")

    def _extract_metrics_matrix(self, manifest: Dict[str, Any]) -> Tuple[List[str], List[List[float]]]:
        items = manifest.get("items", [])
        cols: List[str] = []
        mat: List[List[float]] = []
        for it in items:
            c = it.get("metrics_columns") or []
            v = it.get("metrics_values") or []
            if not cols and c:
                cols = list(c)
            if c and cols and list(c) != cols:
                # tolerate per-item column order variance: use item-specific order later
                pass
            try:
                mat.append([float(x) for x in v])
            except Exception:
                mat.append([])
        return cols, mat

    def _reconcile_columns(self, a: List[str], b: List[str]) -> List[str]:
        # union with a-first ordering, then any b-only columns
        seen, out = set(), []
        for k in a + [x for x in b if x not in a]:
            if k not in seen:
                seen.add(k)
                out.append(k)
        return out

    def _reindex_columns(self, cols_old: List[str], mat: List[List[float]], cols_new: List[str]) -> List[List[float]]:
        if not cols_old:
            # empty old → pad zeros
            return [[0.0 for _ in cols_new] for _ in mat]
        pos = {k: i for i, k in enumerate(cols_old)}
        out = []
        for row in mat:
            out.append([float(row[pos[c]]) if c in pos and pos[c] < len(row) else 0.0 for c in cols_new])
        return out

    def _aggregate(self, mat: List[List[float]]) -> Dict[str, Dict[str, float]]:
        if not mat:
            return {}
        n = len(mat)
        d = len(mat[0]) if mat[0] else 0
        sums = [0.0] * d
        sums2 = [0.0] * d
        for r in mat:
            for i, x in enumerate(r):
                sums[i] += x
                sums2[i] += x * x
        means = [s / n for s in sums]
        stds = []
        for i in range(d):
            # population std (you can switch to sample if you prefer)
            mu = means[i]
            var = max(0.0, (sums2[i] / n) - mu * mu)
            stds.append(var ** 0.5)
        return {f"{i}": {"mean": means[i], "std": stds[i], "n": n} for i in range(d)}

    def _cohens_d(self, *, mean1: float, std1: float, n1: int, mean2: float, std2: float, n2: int) -> float:
        # Pooled SD; guard small n
        if n1 <= 1 or n2 <= 1:
            return 0.0
        # avoid div/0
        s1, s2 = float(std1 or 0.0), float(std2 or 0.0)
        pooled = (( (n1 - 1) * s1 * s1 + (n2 - 1) * s2 * s2 ) / (n1 + n2 - 2)) ** 0.5 if (n1 + n2 - 2) > 0 else 1.0
        if pooled == 0.0:
            return 0.0
        return (mean1 - mean2) / pooled

    def _write_csv(self, path: Path, header: List[str], rows: List[List[Any]]) -> None:
        lines = []
        lines.append(",".join(header))
        for r in rows:
            lines.append(",".join(str(x) for x in r))
        path.write_text("\n".join(lines), encoding="utf-8")

    @staticmethod
    def _safe_float(x: Any) -> float:
        try:
            return float(x)
        except Exception:
            return 0.0


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
