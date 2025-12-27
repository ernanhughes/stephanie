from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict

import numpy as np
import torch

from stephanie.scoring.analysis.trace_tap import TraceTap


def _cos(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) + eps) * (np.linalg.norm(b) + eps)))


def _boundary_rate(traj_TD: np.ndarray, thresh: float) -> float:
    if traj_TD.shape[0] < 2:
        return 0.0
    cos = np.array([_cos(traj_TD[t - 1], traj_TD[t]) for t in range(1, traj_TD.shape[0])])
    return float(np.mean(cos < thresh))


def _mean_step_cos(traj_TD: np.ndarray) -> float:
    if traj_TD.shape[0] < 2:
        return 1.0
    cos = np.array([_cos(traj_TD[t - 1], traj_TD[t]) for t in range(1, traj_TD.shape[0])])
    return float(cos.mean())


def _mean_dwell(traj_TD: np.ndarray, thresh: float) -> float:
    if traj_TD.shape[0] < 2:
        return float(traj_TD.shape[0])
    cos = np.array([_cos(traj_TD[t - 1], traj_TD[t]) for t in range(1, traj_TD.shape[0])])
    boundaries = (cos < thresh).astype(np.int32)
    dwells = []
    cur = 1
    for b in boundaries:
        if b == 0:
            cur += 1
        else:
            dwells.append(cur)
            cur = 1
    dwells.append(cur)
    return float(np.mean(dwells)) if dwells else 0.0


class PaperTemporalAbstractionProcessor:
    """
    Paper-only metrics on traces:
      - stability (mean step cosine)
      - boundary rate
      - mean dwell length

    Reads context['trace_taps'] produced by scorers.
    """

    def __init__(self, *, boundary_thresh: float = 0.90) -> None:
        self.boundary_thresh = boundary_thresh

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        taps = context.get("trace_taps", {})
        out: Dict[str, Any] = {}

        def _analyze(prefix: str, tap: TraceTap, prefer_keys: list[str]) -> None:
            dump = tap.dump()
            traj = None
            for k in prefer_keys:
                if k in dump and dump[k].numel() > 0:
                    traj = dump[k]
                    break
            if traj is None:
                return

            # expect [T, B, D] — reduce B by mean for global metrics
            x = traj.detach().float().cpu().numpy()
            if x.ndim == 3:
                x = x.mean(axis=1)  # [T, D]

            out[f"{prefix}.mean_step_cos"] = _mean_step_cos(x)
            out[f"{prefix}.boundary_rate"] = _boundary_rate(x, self.boundary_thresh)
            out[f"{prefix}.mean_dwell"] = _mean_dwell(x, self.boundary_thresh)

        # Analyze each dimension independently
        for dim, tap in taps.get("hrm", {}).items():
            _analyze(f"paper.hrm.{dim}", tap, ["hrm/zH", "hrm/zL", "hrm/zH_final"])
        for dim, tap in taps.get("tiny", {}).items():
            _analyze(f"paper.tiny.{dim}", tap, ["tiny/z_cur", "tiny/z_next", "tiny/z_final"])

        # Simple mismatch signals per dimension (paper-only)
        for dim in taps.get("hrm", {}).keys():
            hr = out.get(f"paper.hrm.{dim}.boundary_rate")
            tr = out.get(f"paper.tiny.{dim}.boundary_rate")
            if hr is not None and tr is not None:
                out[f"paper.mismatch.{dim}.boundary_rate"] = abs(float(hr) - float(tr))

        # Put back into context so later stages can persist or plot
        context.setdefault("analysis", {})["paper_temporal"] = out
        return context
