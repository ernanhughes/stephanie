from __future__ import annotations
from typing import Dict, Any
import numpy as np

class _PID:
    def __init__(self, kp=0.8, ki=0.0, kd=0.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self._i = 0.0
        self._prev = None

    def step(self, error: float) -> float:
        self._i += error
        d = 0.0 if self._prev is None else (error - self._prev)
        self._prev = error
        return self.kp * error + self.ki * self._i + self.kd * d


class EnhancedHomeostasis:
    """
    Keeps four dimensions near setpoints:
      - energy_balance (metabolic target)
      - boundary_integrity
      - cognitive_flow (simple proxy via cognitive energy)
      - vpm_diversity (stubbed via manager or 0.6)
    Produces telemetry including health/crisis.
    """
    def __init__(self, cfg: Dict[str, Any]):
        sp = cfg or {}
        self.setpoints = {
            "energy_balance": float(sp.get("energy_setpoint", 1.0)),   # multiplier target
            "boundary_integrity": float(sp.get("boundary_setpoint", 0.8)),
            "cognitive_flow": float(sp.get("cognitive_setpoint", 0.6)),
            "vpm_diversity": float(sp.get("diversity_setpoint", 0.7)),
        }
        self.controllers = {
            "energy_balance": _PID(0.6, 0.0, 0.0),
            "boundary_integrity": _PID(0.8, 0.0, 0.0),
            "cognitive_flow": _PID(0.4, 0.0, 0.0),
            "vpm_diversity": _PID(0.3, 0.0, 0.0),
        }

    def regulate(self, core) -> Dict[str, float]:
        # energy balance: try to keep metabolic near 50 * energy_balance
        target_metabolic = 50.0 * self.setpoints["energy_balance"]
        e_err = (target_metabolic - core.energy.level("metabolic")) / 50.0
        e_out = self.controllers["energy_balance"].step(e_err)

        # move a bit between pools based on controller output
        if e_out > 0.0 and core.energy.level("reserve") > 0.2:
            core.energy.transfer("reserve", "metabolic", min(1.0, e_out * 2.0))
        elif e_out < 0.0 and core.energy.level("metabolic") > 1.0:
            core.energy.transfer("metabolic", "reserve", min(1.0, -e_out * 1.0))

        # boundary integrity: adjust thickness slowly
        b_err = self.setpoints["boundary_integrity"] - core.membrane.integrity
        b_out = self.controllers["boundary_integrity"].step(b_err)
        core.membrane.fortify(max(-0.002, min(0.002, b_out * 0.01)))

        # cognitive_flow: proxy by small transfer into cognitive
        c_err = self.setpoints["cognitive_flow"] - (core.energy.level("cognitive") / 100.0)
        c_out = self.controllers["cognitive_flow"].step(c_err)
        if c_out > 0.0 and core.energy.level("metabolic") > 0.5:
            core.energy.transfer("metabolic", "cognitive", min(0.7, c_out * 1.2))

        # vpm_diversity is advisory only here
        reg_actions = {
            "energy_balance": e_out,
            "boundary_integrity": b_out,
            "cognitive_flow": c_out,
            "vpm_diversity": 0.0,
        }

        # compute simple health & crisis
        health = float(np.clip(
            0.30 * core.membrane.integrity
            + 0.30 * (core.energy.level("metabolic") / 100.0)
            + 0.20 * (core.energy.level("reserve") / 100.0)
            + 0.20 * (core.energy.level("cognitive") / 100.0), 0.0, 1.0))

        crisis = float(np.clip(1.0 - health, 0.0, 1.0))
        self._last_telem = {
            "health": health,
            "crisis_level": crisis,
            "homeostatic_error": abs(e_err) + abs(b_err) + abs(c_err),
            "regulatory_actions": reg_actions,
            "setpoints": self.setpoints.copy(),
        }
        return self._last_telem

    def get_telemetry(self) -> Dict[str, Any]:
        return getattr(self, "_last_telem", {
            "health": 0.5, "crisis_level": 0.0, "homeostatic_error": 0.0,
            "regulatory_actions": {"energy_balance":0,"boundary_integrity":0,"cognitive_flow":0,"vpm_diversity":0},
            "setpoints": self.setpoints.copy(),
        })
