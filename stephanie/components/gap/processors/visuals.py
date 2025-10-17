# stephanie/components/gap/processors/visuals.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt

# Which SCM columns count as Tier-1 core dims (for radar plot)
CORE5 = [
    "scm.reasoning.score01",
    "scm.knowledge.score01",
    "scm.clarity.score01",
    "scm.faithfulness.score01",
    "scm.coverage.score01",
]

def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def _save(fig, path: Path) -> str:
    _ensure_dir(path)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return str(path)

def _means_stds(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return np.nanmean(mat, axis=0), np.nanstd(mat, axis=0)

def _idx(cols: List[str], name: str) -> int:
    try:
        return cols.index(name)
    except ValueError:
        return -1

def render_scm_images(
    hrm_scm: np.ndarray,
    tiny_scm: np.ndarray,
    columns: List[str],
    out_dir: str | Path,
) -> Dict[str, str]:
    """
    Produce a small suite of visuals from aligned SCM matrices.
    Returns dict of image paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, str] = {}

    # ---- 1) Core-5 radar (mean) --------------------------------------------
    core_idxs = [i for i, c in enumerate(columns) if c in CORE5]
    if core_idxs and len(core_idxs) == 5:
        labels = [columns[i].split(".")[1] for i in core_idxs]  # reasoning, knowledge, ...
        theta = np.linspace(0, 2*np.pi, len(core_idxs), endpoint=False)
        theta = np.concatenate([theta, theta[:1]])

        h_mean = np.mean(hrm_scm[:, core_idxs], axis=0)
        t_mean = np.mean(tiny_scm[:, core_idxs], axis=0)

        h_plot = np.concatenate([h_mean, h_mean[:1]])
        t_plot = np.concatenate([t_mean, t_mean[:1]])

        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111, projection="polar")
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        ax.set_thetagrids(theta[:-1] * 180/np.pi, labels)
        ax.plot(theta, h_plot, linewidth=2, label="HRM")
        ax.fill(theta, h_plot, alpha=0.15)
        ax.plot(theta, t_plot, linewidth=2, linestyle="--", label="Tiny")
        ax.fill(theta, t_plot, alpha=0.15)
        ax.set_title("SCM Core-5 (means)")
        ax.set_rlabel_position(0)
        ax.set_ylim(0, 1)
        ax.legend(loc="lower right", bbox_to_anchor=(1.15, 0.0))

        paths["scm_core5_radar"] = _save(fig, out_dir / "scm_core5_radar.png")

    # ---- 2) Delta bar (HRM - Tiny) across all SCM columns -------------------
    h_mean, _ = _means_stds(hrm_scm)
    t_mean, _ = _means_stds(tiny_scm)
    delta = h_mean - t_mean

    fig = plt.figure(figsize=(max(8, len(columns) * 0.48), 4.8))
    ax = fig.add_subplot(111)
    ax.bar(range(len(columns)), delta)
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels([c.replace("scm.", "") for c in columns], rotation=60, ha="right")
    ax.axhline(0, linewidth=1)
    ax.set_ylabel("Mean Δ (HRM − Tiny)")
    ax.set_title("SCM: mean difference by metric")
    paths["scm_delta_bar"] = _save(fig, out_dir / "scm_delta_bar.png")

    # ---- 3) Overlaid histograms for each Core-5 (HRM vs Tiny) --------------
    for c in CORE5:
        i = _idx(columns, c)
        if i < 0:
            continue
        h = np.clip(hrm_scm[:, i], 0, 1)
        t = np.clip(tiny_scm[:, i], 0, 1)

        fig = plt.figure(figsize=(5.5, 3.8))
        ax = fig.add_subplot(111)
        ax.hist(h, bins=24, alpha=0.55, label="HRM", density=True)
        ax.hist(t, bins=24, alpha=0.55, label="Tiny", density=True)
        ax.set_xlim(0, 1)
        ax.set_title(f"SCM distribution: {c.replace('scm.', '')}")
        ax.set_xlabel("score")
        ax.set_ylabel("density")
        ax.legend()
        paths[f"hist_{c.split('.')[1]}"] = _save(fig, out_dir / f"scm_hist_{c.split('.')[1]}.png")

    # ---- 4) Scatter HRM vs Tiny (aggregate01) -------------------------------
    agg_i = _idx(columns, "scm.aggregate01")
    if agg_i >= 0:
        fig = plt.figure(figsize=(4.8, 4.8))
        ax = fig.add_subplot(111)
        ax.scatter(tiny_scm[:, agg_i], hrm_scm[:, agg_i], s=8, alpha=0.5)
        ax.set_xlabel("Tiny aggregate01")
        ax.set_ylabel("HRM aggregate01")
        ax.set_title("SCM aggregate: HRM vs Tiny")
        ax.plot([0,1],[0,1], linewidth=1)
        paths["scm_aggregate_scatter"] = _save(fig, out_dir / "scm_aggregate_scatter.png")

    return paths
