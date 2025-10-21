# stephanie/zeromodel/vpm_phos.py
"""
VPM (Vectorized Performance Map) and PHOS (Packed High-Order Structure) Visualization Module

This module provides functionality for creating and analyzing visual performance representations
of AI model outputs across multiple evaluation dimensions. It implements the VPM/PHOS methodology
for model comparison and quality assessment.

Key Features:
- Robust vector normalization and scaling
- PHOS packing algorithms for visual pattern recognition
- Multi-dimensional performance visualization
- Automated artifact selection with improvement guards
- HRM vs Tiny model comparison framework

The PHOS algorithm sorts performance vectors and packs them into 2D representations
that highlight performance concentration patterns, making model strengths/weaknesses
visually apparent.

Author: Stephanie AI Team
Version: 1.0
Date: 2024
"""

from __future__ import annotations
from pathlib import Path

import json
import pandas as pd
import numpy as np
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Optional

import matplotlib.pyplot as plt

from stephanie.utils.json_sanitize import dumps_safe

CAND_SUFFIXES = ["", ".score", ".aggregate", ".raw", ".value"]

# --- fix: dataclass decorator + types ---
@dataclass
class RunManifest:
    run_id: str
    dataset: str
    models: dict
    preproc_version: str = "v1"
    created_at: float = time.time()


def build_vpm(scores: dict, metric_whitelist=None):
    # scores: dict[str] -> np.array[T] or [T,K]
    cols, mats = [], []
    for k, v in scores.items():
        if metric_whitelist and k not in metric_whitelist: continue
        v = np.asarray(v)
        if v.ndim == 1:
            cols.append(k); mats.append(v[:,None])
        elif v.ndim == 2:
            for j in range(v.shape[1]):
                cols.append(f"{k}[{j}]"); mats.append(v[:,j:j+1])
    X = np.concatenate(mats, axis=1) if mats else np.zeros((0,0))
    return X, cols

def robust_normalize(X, eps=1e-9):
    med = np.nanmedian(X, axis=0, keepdims=True)
    mad = np.nanmedian(np.abs(X - med), axis=0, keepdims=True) + eps
    Z = (X - med) / mad
    return np.clip(Z, -5, 5)  # squash extremes

# ---------------------------
# Low-level utils
# ---------------------------

def robust01(x: np.ndarray, p_lo: float = 10.0, p_hi: float = 90.0) -> np.ndarray:
    """
    Robust [0,1] scaling using percentiles to damp outliers.
    
    Args:
        x: Input array to normalize
        p_lo: Lower percentile for scaling (default: 10th percentile)
        p_hi: Upper percentile for scaling (default: 90th percentile)
    
    Returns:
        Array scaled to [0,1] range based on percentile bounds
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return x
    lo = float(np.percentile(x, p_lo))
    hi = float(np.percentile(x, p_hi))
    if hi - lo < 1e-12:
        hi = lo + 1.0
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0)


def learn_layout(*vpm_lists):
    # stack absolute activations to emphasize structure
    U = np.concatenate([np.abs(X) for (X, _) in vpm_lists], axis=1)
    # simple heuristic: order by decreasing L2 norm then by correlation structure
    col_energy = np.linalg.norm(U, axis=0)
    order = np.argsort(-col_energy)
    return order.tolist()

def save_layout(order, names, path):
    with open(path, "w") as f: 
        f.write(dumps_safe({"columns":[names[i] for i in order], "index":order}, indent=2))


def project(X, order):
    keep = [i for i in order if i < X.shape[1]]
    return X[:, keep]

def guess_diag_cols(names):
    DIAG_SUFFIX = ("uncertainty","ood","ood_hat","temp01","entropy","jacobian","consistency","halt_prob")
    return [i for i,n in enumerate(names) if any(s in n for s in DIAG_SUFFIX)]

def correlate_abs_delta_with_diags(Delta, X_diag):
    y = np.abs(Delta).mean(axis=1)  # per-turn intensity
    R = np.corrcoef(X_diag.T, y)[-1,:-1]  # quick/dirty
    return R

def delta_metrics(XA, XB):
    Delta = XA - XB
    absA, absB = np.abs(XA).ravel(), np.abs(XB).ravel()
    overlap = (absA @ absB) / (np.linalg.norm(absA)+1e-9) / (np.linalg.norm(absB)+1e-9)
    dmass = np.mean(np.abs(Delta))  # whole-field mass; optional TL window
    col_scores = np.mean(np.abs(Delta), axis=0)
    row_scores = np.mean(np.abs(Delta), axis=1)
    top_cols = np.argsort(-col_scores)[:25].tolist()
    top_rows = np.argsort(-row_scores)[:25].tolist()
    return Delta, {"delta_mass": float(dmass), "overlap": float(overlap),
                   "top_cols": top_cols, "top_rows": top_rows}

def save_delta(meta, names, path_json):
    meta["column_names"] = names
    with open(path_json, "w") as f:
        f.write(dumps_safe(meta, indent=2))

def to_square(vec: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Pad a 1D vector to the next square length and reshape to (s, s).
    
    Args:
        vec: Input 1D vector
        
    Returns:
        Tuple of (square_image, side_length)
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

def route(use_A_cond, A_agg, B_agg):
    # use A (expensive) when diagnostics say so, else B
    return np.where(use_A_cond, A_agg, B_agg)

def phos_sort_pack(v: np.ndarray, *, tl_frac: float = 0.25) -> np.ndarray:
    """
    PHOS (Packed High-Order Structure) algorithm.
    
    Sorts values in descending order and packs them into a square image with
    top values concentrated in the top-left region.
    
    Args:
        v: Input performance vector
        tl_frac: Fraction of area to allocate for top-left concentration
    
    Returns:
        Square image with sorted and packed values
    """
    v = np.asarray(v, dtype=np.float64).ravel()
    if v.size == 0: 
        return np.zeros((1, 1), dtype=np.float64)
    
    # Normalize and prepare for packing
    v01 = robust01(v)
    n = v01.size
    s = int(np.ceil(np.sqrt(n)))
    pad = s * s - n
    if pad > 0: 
        v01 = np.concatenate([v01, np.zeros(pad)])
    
    # Sort values in descending order
    order = np.argsort(v01)[::-1]
    sorted_vals = v01[order]
    img = sorted_vals.reshape(s, s)
    
    # Calculate top-left block size
    k = max(1, int(round(s * s * tl_frac)))
    packed = np.zeros_like(img)
    packed[:][:] = 0.0
    
    r = int(np.floor(np.sqrt(k)))
    if r <= 0: 
        r = 1
    rr = r
    if rr * r * r > k:
        rr = int(np.floor(np.sqrt(k)))
    
    # Fill top-left with highest values
    top = sorted_vals[:k]
    tl = np.zeros_like(img)
    tl[:rr, :rr] = top[:rr * rr].reshape(rr, rr)
    rest = sorted_vals[rr * rr:]
    
    # Pack remaining values
    packed[:rr, :rr] = tl[:rr, :rr]
    flat = packed.ravel()
    flat[rr * rr:rr * rr + rest.size] = rest
    
    return flat.reshape(s, s)


def image_entropy(img: np.ndarray) -> float:
    """
    Calculate Shannon entropy over normalized pixel mass.
    
    Measures the information content/distribution uniformity in the image.
    Higher entropy = more uniform distribution, lower entropy = more concentrated.
    
    Args:
        img: Input image array
        
    Returns:
        Shannon entropy value
    """
    x = np.asarray(img, dtype=np.float64)
    mass = x.sum()
    if mass <= 0:
        return 0.0
    p = (x / mass).reshape(-1)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p + 1e-12)))


def brightness_concentration(img: np.ndarray, tl_frac: float = 0.25) -> float:
    """
    Calculate fraction of total mass in the top-left region.
    
    Measures how concentrated the high values are in the designated area.
    Used as a key metric for PHOS effectiveness.
    
    Args:
        img: Input image array
        tl_frac: Area fraction for top-left region
        
    Returns:
        Concentration ratio (0-1)
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
    """
    Save grayscale image to disk.
    
    Args:
        img: Image array (values 0-1)
        path: Output file path
        title: Image title
    """
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
    Build a single 1D VPM vector from DataFrame columns.
    
    Supports both MultiIndex and flat column naming conventions.
    Can concatenate or interleave dimensions.
    
    Args:
        df: DataFrame containing model performance scores
        model: Model identifier (e.g., 'hrm', 'tiny')
        dimensions: List of dimension names to include
        interleave: If True, interleave dimensions; if False, concatenate
        weights: Optional dimension weights for weighted combination
        p_lo: Lower percentile for robust scaling
        p_hi: Upper percentile for robust scaling
        
    Returns:
        1D VPM vector combining all specified dimensions
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
    Build both raw VPM and PHOS-packed artifacts for a single model.
    
    Produces:
      - Raw VPM image (simple reshaping)
      - PHOS VPM image (sorted packing)
      - Comprehensive metrics for both
      - PNG files saved to disk
    
    Args:
        df: Input DataFrame with performance scores
        model: Target model name
        dimensions: Evaluation dimensions to include
        out_prefix: Output path prefix
        tl_frac: Top-left area fraction for PHOS packing
        interleave: Whether to interleave dimensions
        weights: Optional dimension weights
        
    Returns:
        Dictionary containing paths, metrics, and configuration
    """
    vec = vpm_vector_from_df(df, model, dimensions, interleave=interleave, weights=weights)
    
    # Generate raw and PHOS images
    raw_img, _ = to_square(vec)
    phos_img = phos_sort_pack(vec)

    # Calculate comparison metrics
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

    # Save visualization files
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
    Select best PHOS candidate from parameter sweep.
    
    Selection strategy:
    1. Prefer first candidate that shows significant improvement over raw
    2. Fallback to candidate with highest PHOS concentration
    
    Args:
        sweep: List of sweep results
        delta: Minimum improvement threshold
        
    Returns:
        Selected candidate configuration
    """
    cand = sorted(sweep, key=lambda r: r["phos_conc"], reverse=True)
    for r in cand:
        if r.get("improved"):
            return r
    return cand[0] if cand else {}



# --- helper: pick best sweep result (prefer improved; else best phos conc) ---
def _chosen_from_sweep(sweep: List[Dict], *, delta: float = 0.02) -> Optional[Dict]:
    if not sweep:
        return None
    improved = [r for r in sweep if r.get("improved")]
    return max(improved, key=lambda r: r.get("phos_conc", 0.0)) if improved else \
           max(sweep,    key=lambda r: r.get("phos_conc", 0.0))

def build_compare_guarded(
    df: pd.DataFrame,
    *,
    dimensions: List[str],
    out_prefix: str,
    model_A: str,                 # e.g. "hf_HRM" or "Llama3-8B"
    model_B: str,                 # e.g. "hf_TinyLama" or "Phi-3-mini"
    tl_fracs: Iterable[float] = (0.25, 0.16, 0.36, 0.09),
    delta: float = 0.02,
    interleave: bool = False,
    weights: Dict[str, float] | None = None,
) -> Dict:
    """
    Compare two models (by alias) with a PHOS guard sweep.

    - Sweeps multiple tl_frac values per model
    - Selects the best (guarded) PHOS config per model
    - Builds a PHOS difference visualization (A − B) if shapes match
    - Emits a summary JSON keyed by the real model aliases
    """
    # Ensure parent folder exists for all outputs derived from out_prefix
    Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict] = {"sweep": {}, "models": [model_A, model_B]}

    # ---- 1) Per-model sweeps -------------------------------------------------
    for model in (model_A, model_B):
        model_sweep: List[Dict] = []
        for tl in tl_fracs:
            prefix = f"{out_prefix}_{model}_tl{float(tl):.2f}"

            # Build both raw and PHOS artifacts for this (model, tl)
            res = build_vpm_phos_artifacts(
                df,
                model=model,
                dimensions=dimensions,
                out_prefix=prefix,
                tl_frac=float(tl),
                interleave=interleave,
                weights=weights,
            )

            raw_c  = float(res["metrics"]["raw"]["brightness_top_left"])
            phos_c = float(res["metrics"]["phos"]["brightness_top_left"])
            improved = phos_c > raw_c * (1.0 + float(delta))

            model_sweep.append({
                "tl_frac": float(tl),
                "raw_conc": raw_c,
                "phos_conc": phos_c,
                "improved": bool(improved),
                "raw_path": res["paths"]["raw"],
                "phos_path": res["paths"]["phos"],
            })

        chosen = _chosen_from_sweep(model_sweep, delta=delta)
        results["sweep"][model] = model_sweep
        results[f"{model}_chosen"] = chosen

        # Persist the sweep details for this model
        with open(f"{out_prefix}_{model}_vpm_guard_metrics.json", "w", encoding="utf-8") as f:
            json.dump({
                "model": model,
                "delta": float(delta),
                "sweep": model_sweep,
                "chosen": chosen,
            }, f, indent=2)

        # Convenience: copy chosen PHOS to a stable name (ignore if not available)
        if chosen and chosen.get("phos_path"):
            import shutil
            dst = f"{out_prefix}_{model}_vpm_chosen.png"
            try:
                shutil.copyfile(chosen["phos_path"], dst)
            except Exception:
                pass

    # ---- 2) PHOS(A) − PHOS(B) visualization (using current DataFrame) -------
    # Use your existing vectorizer + packer. If either fails, we just skip diff.
    try:
        vec_A  = vpm_vector_from_df(df, model_A, dimensions, interleave=interleave, weights=weights)
        vec_B  = vpm_vector_from_df(df, model_B, dimensions, interleave=interleave, weights=weights)
        img_A  = phos_sort_pack(vec_A)
        img_B  = phos_sort_pack(vec_B)

        if img_A.shape == img_B.shape and img_A.size and img_B.size:
            diff   = img_A - img_B
            dmin   = float(diff.min()); dmax = float(diff.max())
            # Normalize to [0,1] for viewing
            diff01 = (diff - dmin) / (dmax - dmin + 1e-12)
            save_img(diff01, f"{out_prefix}_vpm_chosen_diff.png",
                     title=f"PHOS({model_A}) − PHOS({model_B})")
            results["diff_range"] = [dmin, dmax]
        else:
            results["diff_range"] = None
    except Exception:
        # Non-fatal: the sweeps and chosen outputs are still useful
        results["diff_range"] = None

    # ---- 3) Summary (keyed by real aliases) ---------------------------------
    summary = {
        "delta": float(delta),
        "chosen": {
            model_A: results.get(f"{model_A}_chosen"),
            model_B: results.get(f"{model_B}_chosen"),
        }
    }
    with open(f"{out_prefix}_guard_compare.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    results["summary"] = summary
    return results

def pick_metric_column(df: pd.DataFrame, base: str) -> str | None:
    for suf in CAND_SUFFIXES:
        cand = f"{base}{suf}"
        if cand in df.columns:
            return cand
    pref = f"{base}."
    for c in df.columns:
        if isinstance(c, str) and c.startswith(pref):
            return c
    return None

def project_dimensions(df_in: pd.DataFrame, dims: list[str]) -> pd.DataFrame:
    out = {"node_id": df_in["node_id"].values}
    missing = {"hrm": [], "tiny": []}
    for d in dims:
        h = pick_metric_column(df_in, f"hrm.{d}")
        t = pick_metric_column(df_in, f"tiny.{d}")

        if h is None:
            missing["hrm"].append(d); out[f"hrm.{d}"] = 0.0
        else:
            out[f"hrm.{d}"] = df_in[h].astype(float).fillna(0.0)
        if t is None:
            missing["tiny"].append(d); out[f"tiny.{d}"] = 0.0
        else:
            out[f"tiny.{d}"] = df_in[t].astype(float).fillna(0.0)
