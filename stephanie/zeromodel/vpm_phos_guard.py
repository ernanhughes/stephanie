# stephanie/zeromodel/vpm_phos_guard.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------
# Low-level utils
# ---------------------------

def robust01(x: np.ndarray, p_lo: float = 10.0, p_hi: float = 90.0) -> np.ndarray:
    """Robust [0,1] scaling using percentiles to damp outliers."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return x
    lo = float(np.percentile(x, p_lo))
    hi = float(np.percentile(x, p_hi))
    if hi - lo < 1e-12:
        hi = lo + 1.0
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0)


def to_square(vec: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Pad a 1D vector to the next square length and reshape to (s, s).
    Returns (img, s).
    """
    v = np.asarray(vec, dtype=np.float64).reshape(-1)
    n = v.size
    if n == 0:
        return np.zeros((1, 1), dtype=np.float64), 1
    s = int(np.ceil(np.sqrt(n)))
    pad = s * s - n
    if pad > 0:
        v = np.pad(v, (0, pad), mode="constant")
    return v.reshape(s, s), s


def phos_sort_pack(vec: np.ndarray) -> np.ndarray:
    """
    PHOS: sort descending by intensity and pack row-major into a square.
    Brightest flows to the top-left.
    """
    v = np.asarray(vec, dtype=np.float64).reshape(-1)
    if v.size == 0:
        return np.zeros((1, 1), dtype=np.float64)
    order = np.argsort(-v)                 # descending
    sorted_v = v[order]
    img, s = to_square(sorted_v)
    return img


def image_entropy(img: np.ndarray) -> float:
    """Shannon entropy over normalized pixel mass (not a 256-bin histogram)."""
    x = np.asarray(img, dtype=np.float64)
    mass = x.sum()
    if mass <= 0:
        return 0.0
    p = (x / mass).reshape(-1)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p + 1e-12)))


def brightness_concentration(img: np.ndarray, tl_frac: float = 0.25) -> float:
    """
    Fraction of total mass that sits in the top-left ‘tl_frac’ area.
    tl_frac is an area fraction (0..1). The top-left block side is ~ sqrt(tl_frac).
    """
    x = np.asarray(img, dtype=np.float64)
    s = x.shape[0]
    if s == 0:
        return 0.0
    area = max(1, int(round(np.sqrt(max(tl_frac, 1e-9)) * s)))
    area = min(area, s)
    tl_sum = float(x[:area, :area].sum())
    total = float(x.sum()) + 1e-12
    return tl_sum / total


def save_img(img: np.ndarray, path: str, title: str = "") -> None:
    """Save a grayscale image (0..1)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.imshow(np.clip(img, 0.0, 1.0), cmap="gray", vmin=0.0, vmax=1.0)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


# ---------------------------
# VPM vectorization
# ---------------------------

def vpm_vector_from_df(
    df: pd.DataFrame,
    model: str,
    dimensions: List[str],
    *,
    interleave: bool = False,
    weights: Dict[str, float] | None = None,
    p_lo: float = 10.0,
    p_hi: float = 90.0,
) -> np.ndarray:
    """
    Build a single 1D VPM vector for a model by stitching per-dimension,
    robustly scaled columns. Supports both MultiIndex and flat "model.dimension" columns.
    - interleave=False → [all dim0 | all dim1 | ...]
    - interleave=True  → row-wise interleave across dimensions
    """
    cols = []
    for dim in dimensions:
        col = None
        if isinstance(df.columns, pd.MultiIndex):
            key = (model, dim)
            if key in df.columns:
                col = df[key].to_numpy()
        else:
            flat_key = f"{model}.{dim}"
            if flat_key in df.columns:
                col = df[flat_key].to_numpy()

        if col is None:
            col = np.zeros(len(df), dtype=np.float64)
        col = robust01(col, p_lo=p_lo, p_hi=p_hi)
        if weights and dim in weights:
            col = col * float(weights[dim])
        cols.append(col)

    if not cols:
        return np.zeros(0, dtype=np.float64)

    if interleave:
        return np.column_stack(cols).reshape(-1)
    else:
        return np.concatenate(cols, axis=0)


# ---------------------------
# Artifact builders
# ---------------------------

def build_vpm_phos_artifacts(
    df: pd.DataFrame,
    *,
    model: str,
    dimensions: List[str],
    out_prefix: str,
    tl_frac: float = 0.25,
    interleave: bool = False,
    weights: Dict[str, float] | None = None,
) -> Dict:
    """
    Produces:
      - raw VPM (row-major square of the un-sorted vector)
      - PHOS VPM (sorted/packed)
      - metrics for both
      - saves PNGs to disk
    Returns a dict with metrics, paths, and tl_frac used.
    """
    vec = vpm_vector_from_df(df, model, dimensions, interleave=interleave, weights=weights)
    # raw image: row-major reshape (no sorting)
    raw_img, _ = to_square(vec)
    # phos image: sorted packing
    phos_img = phos_sort_pack(vec)

    # metrics
    raw_metrics = {
        "brightness_top_left": brightness_concentration(raw_img, tl_frac=tl_frac),
        "mean": float(raw_img.mean()),
        "std": float(raw_img.std()),
        "entropy": image_entropy(raw_img),
    }
    phos_metrics = {
        "brightness_top_left": brightness_concentration(phos_img, tl_frac=tl_frac),
        "mean": float(phos_img.mean()),
        "std": float(phos_img.std()),
        "entropy": image_entropy(phos_img),
    }

    # save
    raw_path = f"{out_prefix}_vpm_raw.png"
    phos_path = f"{out_prefix}_vpm_phos.png"
    save_img(raw_img, raw_path, title=f"{model.upper()} VPM (raw)")
    save_img(phos_img, phos_path, title=f"{model.upper()} VPM (PHOS)")

    return {
        "model": model,
        "tl_frac": float(tl_frac),
        "paths": {"raw": raw_path, "phos": phos_path},
        "metrics": {"raw": raw_metrics, "phos": phos_metrics},
    }


def _chosen_from_sweep(sweep: List[Dict], delta: float) -> Dict:
    """
    Choose the best PHOS candidate:
      - Prefer first that improves top-left brightness > (1+delta) over raw
      - Fallback to the highest phos brightness
    """
    # sort by phos_conc descending
    cand = sorted(sweep, key=lambda r: r["phos_conc"], reverse=True)
    for r in cand:
        if r.get("improved"):
            return r
    return cand[0] if cand else {}


def build_hrm_vs_tiny_guarded(
    df: pd.DataFrame,
    *,
    dimensions: List[str],
    out_prefix: str,
    tl_fracs: Iterable[float] = (0.25, 0.16, 0.36, 0.09),
    delta: float = 0.02,
    interleave: bool = False,
    weights: Dict[str, float] | None = None,
) -> Dict:
    """
    For each model (hrm, tiny):
      - sweep tl_fracs
      - build raw & PHOS artifacts
      - compute improvement guard
      - select a 'chosen' PHOS image
    Also writes JSON metrics and a diff image on chosen pair (if shapes match).
    """
    results: Dict[str, Dict] = {"sweep": {}}
    models = ["hrm", "tiny"]

    for model in models:
        model_sweep = []
        for tl in tl_fracs:
            prefix = f"{out_prefix}_{model}_tl{tl:.2f}"
            res = build_vpm_phos_artifacts(
                df, model=model, dimensions=dimensions,
                out_prefix=prefix, tl_frac=tl,
                interleave=interleave, weights=weights
            )
            raw_c  = res["metrics"]["raw"]["brightness_top_left"]
            phos_c = res["metrics"]["phos"]["brightness_top_left"]
            improved = phos_c > raw_c * (1.0 + float(delta))
            model_sweep.append({
                "tl_frac": float(tl),
                "raw_conc": float(raw_c),
                "phos_conc": float(phos_c),
                "improved": bool(improved),
                "raw_path": res["paths"]["raw"],
                "phos_path": res["paths"]["phos"],
            })

        chosen = _chosen_from_sweep(model_sweep, delta=delta)
        results["sweep"][model] = model_sweep
        results[f"{model}_chosen"] = chosen

        # Persist guard metrics per model
        with open(f"{out_prefix}_{model}_vpm_guard_metrics.json", "w", encoding="utf-8") as f:
            json.dump({"model": model, "delta": float(delta), "sweep": model_sweep, "chosen": chosen}, f, indent=2)

        # Convenience copy for consistent name
        if chosen:
            # Make a simple copy (symlink portability varies by OS)
            import shutil
            dst = f"{out_prefix}_{model}_vpm_chosen.png"
            try:
                shutil.copyfile(chosen["phos_path"], dst)
            except Exception:
                pass

    # Diff on chosen pair (only if they have same side)
    try:
        hrm_vec = vpm_vector_from_df(df, "hrm", dimensions, interleave=interleave, weights=weights)
        tiny_vec = vpm_vector_from_df(df, "tiny", dimensions, interleave=interleave, weights=weights)
        hrm_img = phos_sort_pack(hrm_vec)
        tiny_img = phos_sort_pack(tiny_vec)
        if hrm_img.shape == tiny_img.shape:
            diff = hrm_img - tiny_img
            dmin, dmax = float(diff.min()), float(diff.max())
            # Normalize to [0,1] for visualization
            diff_vis = (diff - dmin) / (dmax - dmin + 1e-12)
            save_img(diff_vis, f"{out_prefix}_vpm_chosen_diff.png", title="PHOS(HRM) − PHOS(Tiny)")
            results["diff_range"] = [dmin, dmax]
    except Exception:
        # non-fatal
        pass

    # Comparison JSON
    with open(f"{out_prefix}_guard_compare.json", "w", encoding="utf-8") as f:
        json.dump({
            "delta": float(delta),
            "hrm_chosen": results.get("hrm_chosen"),
            "tiny_chosen": results.get("tiny_chosen"),
        }, f, indent=2)

    return results


# ---------------------------
# Minimal CLI (optional)
# ---------------------------

def _load_scores_csv(path: str) -> pd.DataFrame:
    """
    Load a CSV produced by your score-matrix step.
    Accepts either MultiIndex columns (model,dimension) or flat 'model.dimension'.
    """
    df = pd.read_csv(path)
    # Try to detect flat columns like 'hrm.reasoning'
    flat = [c for c in df.columns if isinstance(c, str) and c.count(".") == 1]
    if flat:
        # leave as-is; vpm_vector_from_df handles it
        return df
    # If someone saved MultiIndex via CSV with unnamed levels, try to fix
    return df


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Build guarded PHOS VPMs for HRM vs Tiny.")
    p.add_argument("--scores", required=True, help="Path to CSV with columns like 'hrm.reasoning', 'tiny.reasoning', ...")
    p.add_argument("--out_prefix", default="artifacts/hrm_vs_tiny", help="Output path prefix for images/metrics.")
    p.add_argument("--dimensions", nargs="+", required=True, help="Dimensions in order, e.g. reasoning knowledge clarity faithfulness coverage")
    p.add_argument("--tl_fracs", nargs="+", type=float, default=[0.25, 0.16, 0.36, 0.09], help="Area fractions to sweep.")
    p.add_argument("--delta", type=float, default=0.02, help="Guard threshold: require > (1+delta) improvement in TL brightness.")
    p.add_argument("--interleave", action="store_true", help="Interleave dims across rows instead of concatenating by blocks.")
    args = p.parse_args()

    df = _load_scores_csv(args.scores)
    out = build_hrm_vs_tiny_guarded(
        df,
        dimensions=args.dimensions,
        out_prefix=args.out_prefix,
        tl_fracs=args.tl_fracs,
        delta=args.delta,
        interleave=bool(args.interleave),
    )
    print(json.dumps({"ok": True, "out_prefix": args.out_prefix, "summary_keys": list(out.keys())}, indent=2))
