# stephanie/components/critic/reports/cohort_report.py
from __future__ import annotations

import csv
from datetime import datetime
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import differential_entropy

from stephanie.scoring.metrics.frontier_lens import graph_quality_from_report
from stephanie.utils.json_sanitize import dumps_safe

log = logging.getLogger(__name__)

@staticmethod
def normalize01(arr: np.ndarray) -> np.ndarray:
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
def top_left_order_pair(
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

def save_vpm_heatmap(
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
    m = normalize01(matrix)

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

def render_ab_topleft_heatmaps(
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

def save_vpm_matrix_csv(matrix: np.ndarray, metric_names: list[str], item_ids: list[str], out_path: Path) -> None:
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
    log.info(f"Saved VisiCalc matrix CSV: path: {out_path}, rows: {int(matrix.shape[0])}, cols: {int(matrix.shape[1])}")

def save_ab_npz_dataset(vpm_base: np.ndarray, vpm_tgt: np.ndarray, metric_names: list[str], out_path: Path) -> None:
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
    log.info(f"Saved ab dataset: path: {out_path}, X_shape: {list(X.shape)}, y_len: {int(y.shape[0])}")

def save_ab_episode_features(vc_base, vc_tgt, out_path: Path) -> None:
    """
    Build a tiny dataset for the run-level critic:

        X: [x_base, x_target] where each x is FrontierLens episode features
        y: [0, 1]  (baseline=0, targeted=1)
    """
    try:
        feats_base = np.asarray(vc_base.features, dtype=np.float32).reshape(1, -1)
        feats_tgt  = np.asarray(vc_tgt.features, dtype=np.float32).reshape(1, -1)
        X = np.vstack([feats_base, feats_tgt])
        y = np.array([0, 1], dtype=np.int64)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_path.as_posix(),
            X=X,
            y=y,
            feature_names=np.array(vc_tgt.feature_names, dtype=object),
        )
        log.info(f"EpisodeDatasetSaved path: {out_path}, features: {int(X.shape[1])}")
    except Exception:
        log.exception("CriticCohortAgent: failed to save episode feature dataset")


def compute_metric_separability(
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

# --------------------------------------------------------------------------- #
# Critic Cohort Reporter
# --------------------------------------------------------------------------- #

class CriticCohortReporter:
    """
    Lightweight, agent-local reporter that inspects the CriticCohortAgent
    context, runs a few structural/sanity checks, and emits a Markdown + JSON
    summary.

    Intended usage:
        reporter = CriticCohortReporter(
            run_id=self.run_id,
            out_dir=self.out_dir,
            input_key=self.input_key,
            output_key=self.output_key,
            logger=self.logger,
        )
        reporter(context)
    """

    def __init__(
        self,
        *,
        run_id: str,
        out_dir: Path,
        input_key: str,
        output_key: str,
        logger: Optional[logging.Logger] = None,
        top_k_metrics: int = 10,
    ) -> None:
        self.run_id = run_id
        self.out_dir = Path(out_dir)
        self.input_key = input_key
        self.output_key = output_key
        self.logger = logger
        self.top_k_metrics = int(top_k_metrics)

    # Make the reporter callable so we can do `reporter(context)`
    def __call__(self, context: Dict[str, Any]) -> Dict[str, Any]:
        summary = self._build_summary(context)
        markdown = self._render_markdown(summary)

        # Ensure output dir exists
        self.out_dir.mkdir(parents=True, exist_ok=True)

        md_path = self.out_dir / "critic_cohort_report.md"
        json_path = self.out_dir / "critic_cohort_summary.json"

        md_path.write_text(markdown, encoding="utf-8")

        # Small JSON version of the same info (no giant arrays)
        with json_path.open("w", encoding="utf-8") as f:
            f.write(dumps_safe(summary, indent=2))

        if self.logger:
            self.logger.info(
                "[CriticCohortReporter] saved report → %s", md_path.as_posix()
            )

        # Expose on context for downstream agents / external inspection
        context["critic_cohort_summary"] = summary
        context["critic_cohort_report_markdown"] = markdown
        context["critic_cohort_report_path"] = md_path.as_posix()
        context["critic_cohort_summary_path"] = json_path.as_posix()

        return context

    # ------------------------------------------------------------------ #
    # Summary extraction
    # ------------------------------------------------------------------ #
    def _build_summary(self, context: Dict[str, Any]) -> Dict[str, Any]:
        now = datetime.utcnow().isoformat()

        scorables_all = context.get(self.input_key) or []
        rows_all = context.get(self.output_key) or []

        scorables_tgt = context.get("scorables_targeted") or []
        scorables_base = context.get("scorables_baseline") or []

        has_ab = bool(
            context.get("visicalc_targeted_report")
            and context.get("visicalc_baseline_report")
        )

        # VisiCalc / FrontierLens reports (A/B or single)
        rep_single = context.get("visicalc_report") or {}
        rep_tgt = context.get("visicalc_targeted_report") or {}
        rep_base = context.get("visicalc_baseline_report") or {}

        global_single = rep_single.get("global") or {}
        global_tgt = rep_tgt.get("global") or {}
        global_base = rep_base.get("global") or {}

        frontier_single = rep_single.get("frontier") or {}
        frontier_tgt = rep_tgt.get("frontier") or {}
        frontier_base = rep_base.get("frontier") or {}

        # Metric separability (Cohen's d style) from _compute_metric_separability
        sep = context.get("visicalc_metric_importance") or {}
        metrics_by_d = sep.get("metrics_by_cohens_d") or []
        metrics_all = sep.get("metrics_all") or {}
        n_metrics = len(metrics_all) if metrics_all else (len(metrics_by_d) or 0)

        # AB diff + TopLeft stats if present
        ab_diff = context.get("visicalc_ab_diff") or {}
        ab_topleft = context.get("visicalc_ab_topleft") or {}

        target_quality = context.get("visicalc_target_quality")
        baseline_quality = context.get("visicalc_baseline_quality")
        single_quality = context.get("visicalc_quality")

        # Choose a frontier block for basic config
        frontier_block = (
            frontier_tgt or frontier_single or frontier_base or {}
        )

        summary: Dict[str, Any] = {
            "run_id": self.run_id,
            "generated_utc": now,
            "has_ab": bool(has_ab),
            # Cohort sizes
            "n_scorables_all": len(scorables_all),
            "n_rows_total": len(rows_all),
            "n_scorables_targeted": len(scorables_tgt),
            "n_scorables_baseline": len(scorables_base),
            # Frontier configuration
            "frontier_metric": frontier_block.get("metric"),
            "frontier_band_low": frontier_block.get("low"),
            "frontier_band_high": frontier_block.get("high"),
            "row_region_splits": frontier_block.get("row_region_splits"),
            # Metric separability meta
            "metric_separability": {
                "n_metrics": int(n_metrics),
                "n_rows_target": sep.get("n_tgt"),
                "n_rows_baseline": sep.get("n_base"),
            },
            # Global frontier fractions
            "global": {},
            # Qualities
            "quality": {},
            # AB-specific blocks (filled if has_ab=True)
            "ab_diff": {},
            "topleft": {},
            # Warnings + status
            "warnings": [],
            "status": "ok",
        }

        # -------------------------
        # Global stats (single / AB)
        # -------------------------
        if has_ab:
            summary["global"]["frontier_frac_target"] = global_tgt.get(
                "frontier_frac"
            )
            summary["global"]["frontier_frac_baseline"] = global_base.get(
                "frontier_frac"
            )
            ft = summary["global"]["frontier_frac_target"]
            fb = summary["global"]["frontier_frac_baseline"]
            if ft is not None and fb is not None:
                summary["global"]["frontier_frac_delta"] = float(ft) - float(fb)
        else:
            summary["global"]["frontier_frac_single"] = global_single.get(
                "frontier_frac"
            )

        # -------------------------
        # Quality summary
        # -------------------------
        if has_ab:
            summary["quality"]["target"] = target_quality
            summary["quality"]["baseline"] = baseline_quality
            if target_quality is not None and baseline_quality is not None:
                summary["quality"]["delta"] = float(target_quality) - float(
                    baseline_quality
                )
        else:
            if single_quality is not None:
                summary["quality"]["single"] = single_quality

        # -------------------------
        # AB diff block
        # -------------------------
        if has_ab and ab_diff:
            gdelta = ab_diff.get("global_delta") or {}
            summary["ab_diff"] = {
                "global_frontier_frac_delta": gdelta.get("frontier_frac"),
                "global_mean_score_delta": gdelta.get("mean_score"),
            }

        # -------------------------
        # TopLeft block (AB only)
        # -------------------------
        if has_ab and ab_topleft:
            summary["topleft"] = {
                "gain": ab_topleft.get("gain"),
                "loss": ab_topleft.get("loss"),
                "improvement_ratio": ab_topleft.get("improvement_ratio"),
                "status": ab_topleft.get("status", "ok"),
            }

        # -------------------------
        # Top-k metrics by Cohen's d
        # -------------------------
        top_metrics = []
        for i, m in enumerate(metrics_by_d[: self.top_k_metrics], start=1):
            stats = m.get("stats") or {}
            top_metrics.append(
                {
                    "rank": i,
                    "metric": m.get("metric"),
                    "cohens_d": stats.get("cohens_d"),
                    "delta_mean": stats.get("delta_mean"),
                    "mean_target": stats.get("mean_tgt"),
                    "mean_baseline": stats.get("mean_base"),
                }
            )
        summary["metric_separability"]["top_metrics_by_cohens_d"] = top_metrics

        # -------------------------
        # Warnings / sanity checks
        # -------------------------
        warnings: List[str] = []

        # Basic cohort checks
        if len(scorables_all) == 0:
            warnings.append("No scorables found on input_key; nothing to analyze.")
        if len(rows_all) == 0:
            warnings.append("No feature rows produced; ScorableProcessor returned 0.")

        # Metric separability checks
        ms_meta = summary["metric_separability"]
        if ms_meta["n_metrics"] == 0:
            warnings.append(
                "Metric separability report has 0 metrics; importance computation may have failed."
            )

        # Row count consistency (only if we have AB + separability counts)
        if has_ab:
            rt = ms_meta.get("n_rows_target")
            rb = ms_meta.get("n_rows_baseline")
            if (
                isinstance(rt, int)
                and isinstance(rb, int)
                and isinstance(summary["n_rows_total"], int)
            ):
                if summary["n_rows_total"] != rt + rb:
                    warnings.append(
                        f"Row count mismatch: total={summary['n_rows_total']} "
                        f"but target+baseline={rt + rb} (target={rt}, baseline={rb})."
                    )

        # Quality warnings
        if has_ab and target_quality is not None and baseline_quality is not None:
            dq = summary["quality"]["delta"]
            if abs(dq) < 1e-3:
                warnings.append(
                    f"Target vs baseline quality are nearly identical (Δ={dq:.4f}); "
                    "AB effect may be weak."
                )
            if dq < 0:
                warnings.append(
                    f"Target quality is lower than baseline (Δ={dq:.4f}); "
                    "this may indicate a regression."
                )

        # TopLeft warnings
        if has_ab and summary["topleft"]:
            ir = summary["topleft"].get("improvement_ratio")
            if isinstance(ir, (int, float)):
                if ir < 0:
                    warnings.append(
                        f"TopLeft improvement_ratio={ir:.3f} (< 0 → net loss in high-intensity region)."
                    )
                elif ir < 0.5:
                    warnings.append(
                        f"TopLeft improvement_ratio={ir:.3f} (≤ 0.5 → weak gain in high-intensity region)."
                    )

        if not summary.get("frontier_metric"):
            warnings.append(
                "Frontier metric not set on report; check VisiCalc/frontier_lens configuration."
            )

        summary["warnings"] = warnings
        summary["status"] = "ok" if not warnings else "warnings"

        return summary

    # ------------------------------------------------------------------ #
    # Markdown rendering
    # ------------------------------------------------------------------ #
    def _render_markdown(self, s: Dict[str, Any]) -> str:
        lines: List[str] = []

        lines.append(f"# Critic Cohort Report")
        lines.append("")
        lines.append(f"- **run_id**: `{s['run_id']}`")
        lines.append(f"- **generated_utc**: `{s['generated_utc']}`")
        lines.append(f"- **mode**: `{'A/B (target vs baseline)' if s['has_ab'] else 'single cohort'}`")
        lines.append(f"- **status**: `{s['status']}`")
        lines.append("")

        # Cohort sizes
        lines.append("## Cohort sizes")
        lines.append("")
        lines.append(f"- Total scorables on input: `{s['n_scorables_all']}`")
        lines.append(f"- Total feature rows: `{s['n_rows_total']}`")
        lines.append(f"- Targeted scorables: `{s['n_scorables_targeted']}`")
        lines.append(f"- Baseline scorables: `{s['n_scorables_baseline']}`")
        ms_meta = s["metric_separability"]
        lines.append(
            f"- Rows used for separability: target=`{ms_meta.get('n_rows_target')}`, "
            f"baseline=`{ms_meta.get('n_rows_baseline')}`"
        )
        lines.append(f"- Metrics in separability: `{ms_meta.get('n_metrics')}`")
        lines.append("")

        # Frontier configuration
        lines.append("## Frontier configuration")
        lines.append("")
        lines.append(f"- Frontier metric: `{s.get('frontier_metric')}`")
        lines.append(
            f"- Frontier band: "
            f"`[{s.get('frontier_band_low')}, {s.get('frontier_band_high')}]`"
        )
        lines.append(f"- Row region splits: `{s.get('row_region_splits')}`")
        lines.append("")

        # Quality + global frontier fractions
        lines.append("## Quality & global stats")
        lines.append("")
        if s["has_ab"]:
            q = s["quality"]
            g = s["global"]
            lines.append(
                f"- Quality (target): `{q.get('target')}` "
                f"| Quality (baseline): `{q.get('baseline')}` "
                f"| Δ(target − baseline): `{q.get('delta')}`"
            )
            lines.append(
                f"- Frontier fraction (target): `{g.get('frontier_frac_target')}` "
                f"| baseline: `{g.get('frontier_frac_baseline')}` "
                f"| Δ: `{g.get('frontier_frac_delta')}`"
            )
            if s["ab_diff"]:
                lines.append(
                    f"- Global Δ(frontier_frac): `{s['ab_diff'].get('global_frontier_frac_delta')}`"
                )
                lines.append(
                    f"- Global Δ(mean_score): `{s['ab_diff'].get('global_mean_score_delta')}`"
                )
        else:
            q = s["quality"]
            g = s["global"]
            lines.append(f"- Quality (single cohort): `{q.get('single')}`")
            lines.append(
                f"- Global frontier fraction: `{g.get('frontier_frac_single')}`"
            )
        lines.append("")

        # TopLeft block
        if s["has_ab"] and s["topleft"]:
            tl = s["topleft"]
            lines.append("## TopLeft high-intensity region (AB)")
            lines.append("")
            lines.append(f"- Gain: `{tl.get('gain')}`")
            lines.append(f"- Loss: `{tl.get('loss')}`")
            lines.append(
                f"- Improvement ratio (gain / (gain + loss)): `{tl.get('improvement_ratio')}`"
            )
            lines.append(f"- Status: `{tl.get('status')}`")
            lines.append("")

        # Top-k metrics
        top = ms_meta.get("top_metrics_by_cohens_d") or []
        if top:
            lines.append(f"## Top {len(top)} metrics by |Cohen's d|")
            lines.append("")
            lines.append(
                "| # | Metric | Cohen's d | Δ mean (target−baseline) | mean_target | mean_baseline |"
            )
            lines.append(
                "|---|--------|-----------|--------------------------|-------------|---------------|"
            )
            for m in top:
                lines.append(
                    f"| {m['rank']} "
                    f"| `{m['metric']}` "
                    f"| {m.get('cohens_d')} "
                    f"| {m.get('delta_mean')} "
                    f"| {m.get('mean_target')} "
                    f"| {m.get('mean_baseline')} |"
                )
            lines.append("")

        # Warnings / sanity
        lines.append("## Sanity checks")
        lines.append("")
        if not s["warnings"]:
            lines.append("- ✅ No warnings raised; basic structural checks passed.")
        else:
            lines.append(f"- ⚠️ {len(s['warnings'])} warning(s):")
            for w in s["warnings"]:
                lines.append(f"  - {w}")
        lines.append("")

        return "\n".join(lines)
