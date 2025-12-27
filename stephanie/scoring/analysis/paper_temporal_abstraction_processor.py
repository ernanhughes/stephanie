# stephanie/scoring/analysis/paper_temporal_abstraction_processor.py
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict

from stephanie.scoring.analysis.temporal_abstraction_metrics import compute_temporal_metrics
from stephanie.scoring.analysis.trace_tap import TraceTap


class PaperTemporalAbstractionProcessor:
    """
    Paper-only analysis: measure temporal abstraction signatures from traces.

    Inputs (per item):
      - hrm_tap.dump(): tensors like hrm/zH, hrm/zL
      - tiny_tap.dump(): tensors like tiny/z_cur, tiny/z_next

    Outputs:
      - metrics dict you can save into EvaluationAttributes or into artifacts.
    """

    def __init__(self, *, boundary_thresh: float = 0.90, n_modes: int = 4) -> None:
        self.boundary_thresh = boundary_thresh
        self.n_modes = n_modes

    def analyze(self, *, hrm_tap: TraceTap, tiny_tap: TraceTap) -> Dict[str, Any]:
        hrm = hrm_tap.dump()
        tiny = tiny_tap.dump()

        # Choose the most relevant trajectories (you can adjust later)
        hrm_traj = hrm.get("hrm/zH") or hrm.get("hrm/zL")
        tiny_traj = tiny.get("tiny/z_cur") or tiny.get("tiny/z_next")

        out: Dict[str, Any] = {}

        if hrm_traj is not None and hrm_traj.numel() > 0:
            m = compute_temporal_metrics(hrm_traj, boundary_thresh=self.boundary_thresh, n_modes=self.n_modes)
            for k, v in asdict(m).items():
                out[f"paper.hrm.{k}"] = v

        if tiny_traj is not None and tiny_traj.numel() > 0:
            m = compute_temporal_metrics(tiny_traj, boundary_thresh=self.boundary_thresh, n_modes=self.n_modes)
            for k, v in asdict(m).items():
                out[f"paper.tiny.{k}"] = v

        # simple mismatches (still paper-only; no GAP)
        if "paper.hrm.boundary_rate" in out and "paper.tiny.boundary_rate" in out:
            out["paper.mismatch.boundary_rate"] = abs(out["paper.hrm.boundary_rate"] - out["paper.tiny.boundary_rate"])
        if "paper.hrm.mode_switch_rate" in out and "paper.tiny.mode_switch_rate" in out:
            out["paper.mismatch.mode_switch_rate"] = abs(out["paper.hrm.mode_switch_rate"] - out["paper.tiny.mode_switch_rate"])

        return out
