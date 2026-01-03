# stephanie/scoring/analysis/temporal_abstraction_metrics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch


@dataclass
class TemporalAbstractionMetrics:
    mean_step_cos: float
    boundary_rate: float
    mean_dwell: float
    mode_switch_rate: float
    n_modes: int


def _cos(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    return float(
        np.dot(a, b) / ((np.linalg.norm(a) + eps) * (np.linalg.norm(b) + eps))
    )


def stability_metrics(
    traj_TD: np.ndarray, *, boundary_thresh: float
) -> Tuple[float, float, float]:
    T = traj_TD.shape[0]
    if T < 2:
        return 1.0, 0.0, float(T)

    step_cos = np.array(
        [_cos(traj_TD[t - 1], traj_TD[t]) for t in range(1, T)],
        dtype=np.float64,
    )
    boundaries = (step_cos < boundary_thresh).astype(np.int32)

    mean_step_cos = float(step_cos.mean())
    boundary_rate = float(boundaries.mean())

    # dwell lengths
    dwells = []
    cur = 1
    for b in boundaries:
        if b == 0:
            cur += 1
        else:
            dwells.append(cur)
            cur = 1
    dwells.append(cur)
    mean_dwell = float(np.mean(dwells)) if dwells else 0.0
    return mean_step_cos, boundary_rate, mean_dwell


def tiny_kmeans_labels(
    z: np.ndarray, k: int = 4, iters: int = 10
) -> np.ndarray:
    """
    Minimal k-means to avoid sklearn dependency.
    z: [T, d]
    """
    T = z.shape[0]
    K = min(k, T)
    # init centers by spaced sampling
    cent = z[np.linspace(0, T - 1, K).astype(int)]

    for _ in range(iters):
        dist = ((z[:, None, :] - cent[None, :, :]) ** 2).sum(axis=-1)
        lab = dist.argmin(axis=1)
        for j in range(K):
            idx = np.where(lab == j)[0]
            if len(idx) > 0:
                cent[j] = z[idx].mean(axis=0)
    return lab


def mode_metrics(traj_TD: np.ndarray, *, n_modes: int) -> Tuple[int, float]:
    T, D = traj_TD.shape
    if T < 2:
        return 1, 0.0

    # quick SVD-based projection to <= 8 dims
    x = traj_TD - traj_TD.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(x, full_matrices=False)
    k = min(8, Vt.shape[0])
    z = x @ Vt[:k].T  # [T, k]

    labels = tiny_kmeans_labels(z, k=n_modes, iters=10)
    switches = float(np.mean(labels[1:] != labels[:-1]))
    return int(len(np.unique(labels))), switches


def compute_temporal_metrics(
    traj_TBD: torch.Tensor, *, boundary_thresh: float = 0.90, n_modes: int = 4
) -> TemporalAbstractionMetrics:
    """
    traj_TBD: [T, B, D]
    Returns batch-averaged metrics.
    """
    if traj_TBD.numel() == 0:
        return TemporalAbstractionMetrics(1.0, 0.0, 0.0, 0.0, 1)

    T, B, D = traj_TBD.shape
    ms, br, dw, nm, sw = [], [], [], [], []

    x = traj_TBD.detach().float().cpu().numpy()
    for b in range(B):
        traj = x[:, b, :]  # [T, D]
        m1, m2, m3 = stability_metrics(traj, boundary_thresh=boundary_thresh)
        k, s = mode_metrics(traj, n_modes=n_modes)
        ms.append(m1)
        br.append(m2)
        dw.append(m3)
        nm.append(k)
        sw.append(s)

    return TemporalAbstractionMetrics(
        mean_step_cos=float(np.mean(ms)),
        boundary_rate=float(np.mean(br)),
        mean_dwell=float(np.mean(dw)),
        mode_switch_rate=float(np.mean(sw)),
        n_modes=int(round(np.mean(nm))),
    )
