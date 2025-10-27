# stephanie/components/jitter/core.py
"""
JAS Core – Metabolic Engine + Boundary Membrane + Core Orchestration
Drop-in production module (no external deps beyond torch + typing + logging).

Expected external integrations:
- ebt_model: provides compatibility scoring (see _ebt_compat()).
- vpm_manager: provides diversity() and active_count() if available.

Telemetry is produced by the lifecycle agent; this core returns vitals per cycle.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn as nn

log = logging.getLogger("stephanie.jas.core")



class EnergyPools:
    def __init__(self, cognitive: float = 50.0, metabolic: float = 50.0, reserve: float = 20.0):
        self.energy_pools = {
            "cognitive": float(cognitive),
            "metabolic": float(metabolic),
            "reserve": float(reserve),
        }
        self.max_reserve = 100.
        self.pathway_rate_factor: float = 1.0

    def level(self, name: str) -> float:
        return float(self.energy_pools.get(name, 0.0))

    def replenish(self, name: str, amount: float) -> None:
        self.energy_pools[name] = max(0.0, self.energy_pools.get(name, 0.0) + float(amount))

    def consume(self, name: str, amount: float) -> None:
        self.energy_pools[name] = max(0.0, self.energy_pools.get(name, 0.0) - float(amount))

    def transfer(self, src: str, dst: str, amount: float) -> float:
        amount = max(0.0, float(amount))
        avail = min(self.level(src), amount)
        self.consume(src, avail)
        self.replenish(dst, avail)
        return avail

    def alive(self) -> bool:
        # minimal: dies if both cognitive & metabolic exhausted
        return self.level("metabolic") > 0.5 or self.level("cognitive") > 0.5

    def adjust_pathway_rates(self, factor: float) -> None:
        """
        Adjust global pathway “throughput”. For simple pools this is stored as a
        scalar that other components may consult. It’s clamped to a safe range.
        """
        try:
            f = max(0.1, float(factor))
        except Exception:
            f = 1.0
        # multiplicative update, then clamp overall factor
        self.pathway_rate_factor = max(0.2, min(5.0, self.pathway_rate_factor * f))

@dataclass
class Membrane:
    integrity: float = 0.8      # 0..1
    thickness: float = 0.8      # 0..1
    permeability: float = 0.5   # 0..1
    last_stress: float = 0.0    # 0..1

    def apply_stress(self, stress: float) -> None:
        self.last_stress = float(max(0.0, min(1.0, stress)))
        # Higher thickness resists stress
        dmg = self.last_stress * (1.0 - 0.5 * self.thickness) * 0.02
        self.integrity = float(max(0.0, min(1.0, self.integrity - dmg)))

    def fortify(self, delta: float) -> None:
        self.thickness = float(max(0.0, min(1.0, self.thickness + delta)))


class AutopoieticCore:
    """
    Minimal core that:
      - burns metabolic baseline
      - applies membrane stress fed from cognition (reptilian threat)
      - supports telemetry snapshot
    """
    def __init__(self, cfg: Dict[str, Any], container, memory):
        e = cfg.get("energy", {})
        self.energy = EnergyPools(
            cognitive=e.get("initial_cognitive", 50.0),
            metabolic=e.get("initial_metabolic", 50.0),
            reserve=e.get("initial_reserve", 20.0),
        )
        m = cfg.get("membrane", {})
        self.membrane = Membrane(
            integrity=m.get("initial_integrity", 0.8),
            thickness=m.get("initial_thickness", 0.8),
            permeability=m.get("initial_permeability", 0.5),
        )
        self.id = cfg.get("id", f"jas_{int(time.time())}")
        self.parent_id = cfg.get("parent_id", "")
        self.generation = int(cfg.get("generation", 0))
        self.tick = 0

    def cycle(self, sensory_input: torch.Tensor) -> Dict[str, Any]:
        self.tick += 1
        # baseline burn
        baseline = 0.6
        self.energy.consume("metabolic", baseline)

        # cognitive upkeep cost scales with cognitive pool
        upkeep = 0.04 * (1.0 + 0.2 * (self.energy.level("cognitive") / 50.0))
        self.energy.consume("cognitive", upkeep)

        # emergency draw from reserve if metabolic too low
        if self.energy.level("metabolic") < 5.0 and self.energy.level("reserve") > 1.0:
            self.energy.transfer("reserve", "metabolic", 2.0)

        # small natural repair if enough metabolic energy
        if self.energy.level("metabolic") > 20.0:
            self.membrane.integrity = min(1.0, self.membrane.integrity + 0.002)

        return {
            "tick": self.tick,
            "baseline": baseline,
            "upkeep": upkeep,
            "integrity": self.membrane.integrity,
        }

    def snapshot(self) -> Dict[str, Any]:
        return {
            "tick": self.tick,
            "membrane": {
                "integrity": self.membrane.integrity,
                "thickness": self.membrane.thickness,
                "permeability": self.membrane.permeability,
                "last_stress": self.membrane.last_stress,
            },
            "energy": {
                "cognitive": self.energy.level("cognitive"),
                "metabolic": self.energy.level("metabolic"),
                "reserve": self.energy.level("reserve"),
            },
            "id": self.id,
            "parent_id": self.parent_id,
            "generation": self.generation,
        }

# ----------------------------- Energy Metabolism -----------------------------


class EnergyMetabolism(nn.Module):
    """
    Biological-style energy pools with two learnable conversion pathways.
    - Extraction converts compatible stimuli → raw energy
    - Pathways allocate energy between pools
    - Maintenance consumes energy proportional to system load
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.pools = EnergyPools(
            cognitive=float(cfg.get("initial_cognitive", 50.0)),
            metabolic=float(cfg.get("initial_metabolic", 50.0)),
            reserve=float(cfg.get("initial_reserve", 20.0)),
        )
        self.extraction_gain = float(cfg.get("extraction_gain", 1.0))
        self.max_level = float(cfg.get("max_level", 100.0))
        self.vpm_cost_per = float(cfg.get("vpm_cost_per", 0.001))  # per active VPM per tick
        self.boundary_cost_gain = float(cfg.get("boundary_cost_gain", 0.05))

        # Learnable pathways (small, stable MLPs)
        self.pathways = nn.ModuleDict({
            "cognitive_to_metabolic": nn.Sequential(
                nn.Linear(1, 8), nn.Tanh(), nn.Linear(8, 1)
            ),
            "metabolic_to_reserve": nn.Sequential(
                nn.Linear(1, 8), nn.Sigmoid(), nn.Linear(8, 1)
            ),
        })

        # Optional hooks
        self._vpm_manager = None  # set via attach_vpm_manager()

    # ---- wiring hooks -------------------------------------------------------
    def attach_vpm_manager(self, vpm_manager) -> None:
        self._vpm_manager = vpm_manager

    # ---- public API ---------------------------------------------------------
    def level(self, pool: str) -> float:
        return float(getattr(self.pools, pool, 0.0))

    def alive(self) -> bool:
        return self.pools.cognitive > 0.0 and self.pools.metabolic > 0.0

    def replenish(self, pool: str, amount: float) -> None:
        new_val = min(self.max_level, max(0.0, self.level(pool) + float(amount)))
        setattr(self.pools, pool, new_val)

    def consume(self, pool: str, amount: float) -> None:
        new_val = max(0.0, self.level(pool) - float(amount))
        setattr(self.pools, pool, new_val)

    def ratio(self) -> float:
        return self.level("cognitive") / (self.level("metabolic") + 1e-8)

    def snapshot(self) -> Dict[str, float]:
        return {
            "cognitive": self.pools.cognitive,
            "metabolic": self.pools.metabolic,
            "reserve": self.pools.reserve,
        }

    def adjust_c2m_bias(self, delta: float) -> None:
        """
        Adjusts bias of the cognitive→metabolic pathway final layer (homeostasis hook).
        Positive delta pushes more to metabolic; negative reduces.
        """
        last = list(self.pathways["cognitive_to_metabolic"].children())[-1]
        if isinstance(last, nn.Linear):
            with torch.no_grad():
                last.bias.add_(float(delta))

    # ---- internal mechanics -------------------------------------------------
    def _distribute(self, amount: float) -> None:
        amt = torch.tensor([[float(amount)]], dtype=torch.float32)

        # Decide cognitive portion
        c_gain = self.pathways["cognitive_to_metabolic"](amt).clamp(min=0.0).item()
        c_gain = max(0.0, min(float(amount), c_gain))

        self.replenish("cognitive", c_gain)
        self.replenish("metabolic", float(amount) - c_gain)

        # Trickling metabolic→reserve (saves for reproduction/long-term)
        meta_small = torch.tensor([[self.level("metabolic") * 0.01]], dtype=torch.float32)
        to_reserve = self.pathways["metabolic_to_reserve"](meta_small).clamp(min=0.0).item()
        to_reserve = min(self.level("metabolic") * 0.05, to_reserve)
        if to_reserve > 0:
            self.consume("metabolic", to_reserve)
            self.replenish("reserve", to_reserve)

    def _maintenance(self, boundary_integrity: float) -> None:
        # Boundary upkeep costs more when integrity is low
        boundary_cost = self.boundary_cost_gain * (1.0 - float(boundary_integrity))
        self.consume("metabolic", boundary_cost)

        # Cognitive housekeeping scales with active VPMs if available
        active = 0
        if self._vpm_manager and hasattr(self._vpm_manager, "active_count"):
            try:
                active = int(self._vpm_manager.active_count())
            except Exception:
                active = 0
        self.consume("cognitive", self.vpm_cost_per * active)

    # ---- main conversion ----------------------------------------------------
    def process(self, emb: torch.Tensor, ebt_compat_score: float, boundary_integrity: float) -> float:
        """
        Convert input → energy.
        ebt_compat_score ∈ [0,1]; higher = more compatible (we invert for raw extraction)
        Returns net extracted energy before maintenance.
        """
        raw = max(0.0, float(ebt_compat_score)) * self.extraction_gain
        self._distribute(raw)
        self._maintenance(boundary_integrity)
        return raw


# ------------------------------- Membrane -----------------------------------

