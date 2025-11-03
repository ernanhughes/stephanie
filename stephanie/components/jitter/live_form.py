# stephanie/components/jitter/live_form.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

Array = np.ndarray

# ---------- small helpers ----------
def robust01(x: Array, eps: float = 1e-9) -> Array:
    x = x.astype(np.float32, copy=False)
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    return np.clip((x - lo) / max(hi - lo, eps), 0, 1)

def roi_stats(tile: Array, roi: Tuple[slice, slice, slice]) -> Dict[str, float]:
    v = tile[roi]
    return dict(mean=float(v.mean()), max=float(v.max()), std=float(v.std()))

# ---------- config ----------
@dataclass
class LiveFormConfig:
    tick_hz: float = 4.0
    tile_size: int = 256
    channels: List[str] = field(default_factory=lambda: [
        "reasoning","knowledge","clarity","faithfulness","coverage",
        "uncertainty","novelty","contradiction","energy"
    ])
    critical_roi: Tuple[slice, slice, slice] = (slice(0,16), slice(0,16), slice(0,1))  # top-left, ch0
    act_threshold: float = 0.78
    learn_rate: float = 0.2
    crisis_hi: float = 0.85
    crisis_lo: float = 0.65
    min_energy: float = 0.10
    max_bad_ticks: int = 50

# ---------- the live form ----------
class LiveForm:
    def __init__(
        self,
        cfg: LiveFormConfig,
        *,
        vpm_encoder: Callable[[Dict[str, Any]], Array],
        actuator: Callable[[Dict[str, Any]], None],
        reward_fn: Callable[[Array, Dict[str, Any]], float],
        logger: Optional[Callable[[str], None]] = None,
    ):
        self.cfg = cfg
        self.vpm_encoder = vpm_encoder
        self.actuator = actuator
        self.reward_fn = reward_fn
        self.log = logger or (lambda s: None)

        # tiny head: y = w·x + b on [mean,max,std] of critical ROI + global energy
        self.w = np.zeros(4, dtype=np.float32)
        self.b = 0.0

        self.energy_cognitive = 0.5
        self.energy_metabolic = 0.5
        self.pathway_rate = 1.0  # compute throttle
        self.crisis_ticks = 0
        self.tick = 0

    # ----- main loop step -----
    def step(self, sensory: Dict[str, Any]) -> Dict[str, Any]:
        self.tick += 1

        # 1) sense → VPM tile
        tile = self.vpm_encoder(sensory)  # shape (H, W, C) float32 in [0,1]
        assert tile.ndim == 3, "tile must be HxWxC"
        H, W, C = tile.shape

        # 2) think (extract features from the image)
        stats = roi_stats(tile, self.cfg.critical_roi)           # [mean,max,std] in critical zone
        energy = float(tile[..., self._energy_ch()].mean())      # energy channel heuristic
        x = np.array([stats["mean"], stats["max"], stats["std"], energy], dtype=np.float32)

        # 3) decide
        y_hat = float(self.w @ x + self.b)
        should_act = y_hat > self.cfg.act_threshold

        # 4) act (minimal, non-destructive)
        if should_act:
            self.actuator({
                "tick": self.tick,
                "reason": "critical_roi_high",
                "score": y_hat,
                "roi_stats": stats,
            })

        # 5) learn (online)
        reward = self.reward_fn(tile, {"acted": should_act, "score": y_hat})
        td_error = reward - y_hat
        self.w += self.cfg.learn_rate * td_error * x
        self.b += self.cfg.learn_rate * td_error

        # 6) homeostasis (compute budget + crisis hysteresis)
        ch = self._contradiction_ch()
        if getattr(tile, "ndim", 2) >= 3:
            crisis_level = float(tile[..., ch].mean())
        else:
            crisis_level = float(tile.mean())

        if crisis_level > self.cfg.crisis_hi:
            self.crisis_ticks += 1
            self.pathway_rate = min(5.0, self.pathway_rate * 1.1)  # spend more to stabilize
            self.energy_metabolic = max(0.0, self.energy_metabolic - 0.02)
        elif crisis_level < self.cfg.crisis_lo:
            self.crisis_ticks = max(0, self.crisis_ticks - 1)
            self.pathway_rate = max(0.2, self.pathway_rate * 0.9)

        # synthetic energy decay
        self.energy_cognitive = max(0.0, self.energy_cognitive - 0.005 * self.pathway_rate)

        # 7) lifecycle (apoptosis)
        die = (
            (self.energy_cognitive < self.cfg.min_energy and self.energy_metabolic < self.cfg.min_energy)
            or (self.crisis_ticks > self.cfg.max_bad_ticks)
        )

        info = {
            "tick": self.tick,
            "tile_shape": (H, W, C),
            "x": x.tolist(),
            "y_hat": y_hat,
            "should_act": should_act,
            "reward": reward,
            "energy": {"cognitive": self.energy_cognitive, "metabolic": self.energy_metabolic},
            "pathway_rate": self.pathway_rate,
            "crisis_ticks": self.crisis_ticks,
            "die": die,
        }
        if die:
            self.log(f"[apoptosis] reason={'energy' if self.energy_cognitive < self.cfg.min_energy else 'prolonged_crisis'} tick={self.tick}")
        return info

    # ----- channel helpers (by name) -----
    def _ch(self, name: str) -> int:
        try:
            return self.cfg.channels.index(name)
        except ValueError:
            return len(self.cfg.channels) - 1  # last as fallback

    def _energy_ch(self) -> int: return self._ch("energy")
    def _contradiction_ch(self) -> int: return self._ch("contradiction")
