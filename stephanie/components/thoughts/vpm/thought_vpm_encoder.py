from __future__ import annotations
import numpy as np
from pathlib import Path
from PIL import Image
from ..thought_trace import ThoughtTrace

class ThoughtVPMEncoder:
    """Encodes a ThoughtTrace into a simple H×W tile with channels for score/uncertainty.
    Export: PNG under runs/thoughts/<run_id>.png
    """
    def __init__(self, out_root: str = "runs/thoughts"):
        self.out_root = Path(out_root)
        self.out_root.mkdir(parents=True, exist_ok=True)

    def _norm(self, x: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(x)
        x = (x - x.min()) / (x.ptp() + 1e-9)
        return (x * 255).astype(np.uint8)

    def encode(self, trace: ThoughtTrace) -> str:
        n = max(1, len(trace.thoughts))
        scores = np.array([t.score for t in trace.thoughts] + [0.0]*(max(0, 64-n)))[:64]
        uncs   = np.array([t.uncertainty for t in trace.thoughts] + [1.0]*(max(0, 64-n)))[:64]
        # Square tile (8×8)
        S = 8
        scores = scores.reshape(S, S)
        uncs   = uncs.reshape(S, S)

        r = self._norm(scores)
        g = self._norm(1.0 - uncs)
        b = np.zeros_like(r)
        rgb = np.stack([r, g, b], axis=-1)

        img = Image.fromarray(rgb, mode="RGB")
        out = self.out_root / f"{trace.run_id or 'trace'}.png"
        img.save(out)
        return str(out)
