# stephanie/components/critic/reporting/frontier_lens_viz.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np

from stephanie.scoring.metrics.frontier_lens import FrontierLens

log = logging.getLogger(__name__)
PathLike = Union[str, Path]


def render_frontier_lens_figure(
    lens: FrontierLens,
    out_path: PathLike,
    *,
    per_metric_normalize: bool = True,
    dpi: int = 220,
) -> Path:
    """
    Render a 3-panel figure that visually explains the Frontier Lens:

    - Panel 1: all metrics (normalized), frontier metric column highlighted
    - Panel 2: frontier metric vs row index, with band [low, high] and in-band rows highlighted
    - Panel 3: only rows that fall inside the frontier band (the 'focused' region)

    Returns the path to the saved PNG.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    X = np.asarray(lens.scores, dtype=np.float32)
    n_rows, n_metrics = X.shape

    # --- normalization ---
    if per_metric_normalize:
        col_min = X.min(axis=0, keepdims=True)
        col_max = X.max(axis=0, keepdims=True)
        denom = np.where(col_max > col_min, col_max - col_min, 1.0)
        X_norm = (X - col_min) / denom
    else:
        gmin, gmax = float(X.min()), float(X.max())
        denom = (gmax - gmin) or 1.0
        X_norm = (X - gmin) / denom

    # --- frontier metric & band ---
    try:
        fm_idx = lens.metric_names.index(lens.frontier_metric)
    except ValueError:
        log.warning(
            "Frontier metric %r not in metric_names; using index 0",
            lens.frontier_metric,
        )
        fm_idx = 0

    frontier_vals = X_norm[:, fm_idx]
    low, high = float(lens.frontier_low), float(lens.frontier_high)
    in_band = (frontier_vals >= low) & (frontier_vals <= high)
    idx = np.arange(n_rows)

    # --- build figure ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

    # 1) All metrics
    ax0 = axes[0]
    im0 = ax0.imshow(X_norm, aspect="auto", interpolation="nearest", origin="upper")
    ax0.set_title("All metrics (normalized)")
    ax0.set_xlabel("metric")
    ax0.set_ylabel("item")
    ax0.axvline(fm_idx, linestyle="--", linewidth=1.5)  # frontier metric
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

    # 2) Frontier band
    ax1 = axes[1]
    # out-of-band grey, in-band default
    ax1.scatter(frontier_vals[~in_band], idx[~in_band], s=10, alpha=0.3)
    ax1.scatter(frontier_vals[in_band], idx[in_band], s=12)
    ax1.axvline(low, linestyle="--")
    ax1.axvline(high, linestyle="--")
    ax1.invert_yaxis()
    ax1.set_title(f"Frontier metric: {lens.frontier_metric}")
    ax1.set_xlabel("normalized score")
    ax1.set_ylabel("item")

    # 3) Focused region
    ax2 = axes[2]
    if in_band.any():
        X_band = X_norm[in_band]
        im2 = ax2.imshow(X_band, aspect="auto", interpolation="nearest", origin="upper")
        ax2.axvline(fm_idx, linestyle="--", linewidth=1.5)
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        ax2.set_title(f"Focused region (in-band rows: {in_band.sum()})")
    else:
        ax2.text(
            0.5,
            0.5,
            "No rows in frontier band",
            ha="center",
            va="center",
            transform=ax2.transAxes,
        )
        ax2.set_title("Focused region (empty)")
    ax2.set_xlabel("metric")
    ax2.set_ylabel("item (in-band)")

    cohort_label = lens.meta.get("cohort_label") if lens.meta else None
    title = f"Frontier Lens — {lens.episode_id}"
    if cohort_label:
        title += f" ({cohort_label})"
    fig.suptitle(title, fontsize=11)

    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

    log.info("Wrote frontier lens figure to %s", out_path)
    return out_path
