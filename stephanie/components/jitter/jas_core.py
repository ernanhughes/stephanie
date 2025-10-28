# stephanie/components/jitter/jas_core.py
"""
JAS Core – Metabolic Engine + Boundary Membrane + Core Orchestration
Drop-in production module (no external deps beyond torch + typing + logging).
Expected external integrations:
ebt_model: provides compatibility scoring (see _ebt_compat()).
vpm_manager: provides diversity() and active_count() if available.
Telemetry is produced by the lifecycle agent; this core returns vitals per cycle.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Any
import logging

log = logging.getLogger("stephanie.jas.core")


class EnergyPools:
    """Simple energy management for JAS."""

    def __init__(self, cognitive: float = 50.0, metabolic: float = 50.0, reserve: float = 20.0):
        self.energy_pools = {
            "cognitive": float(cognitive),
            "metabolic": float(metabolic),
            "reserve": float(reserve),
        }
        self.max_reserve = 100.0

    def level(self, pool: str) -> float:
        """Get current level of an energy pool."""
        return self.energy_pools.get(pool, 0.0)

    def replenish(self, pool: str, amount: float):
        """Add energy to a pool."""
        if pool in self.energy_pools:
            self.energy_pools[pool] = min(
                self.max_reserve if pool == "reserve" else 100.0,
                self.energy_pools[pool] + amount
            )

    def consume(self, pool: str, amount: float) -> bool:
        """Consume energy from a pool. Returns True if sufficient energy."""
        if pool in self.energy_pools and self.energy_pools[pool] >= amount:
            self.energy_pools[pool] -= amount
            return True
        return False

    def alive(self) -> bool:
        """Check if the organism has sufficient energy to survive."""
        return (
            self.energy_pools.get("cognitive", 0.0) > 1.0 and
            self.energy_pools.get("metabolic", 0.0) > 1.0
        )

    def snapshot(self) -> Dict[str, float]:
        """Get a snapshot of current energy levels."""
        return self.energy_pools.copy()


class Membrane:
    """Simple membrane system for JAS."""

    def __init__(self, integrity: float = 0.8, thickness: float = 0.8, permeability: float = 0.5):
        self.integrity = float(integrity)
        self.thickness = float(thickness)
        self.permeability = float(permeability)
        self.last_stress = 0.0

    def apply_stress(self, stress_level: float):
        """Apply environmental stress to the membrane."""
        self.last_stress = stress_level
        # Reduce integrity based on stress
        self.integrity = max(0.0, self.integrity - stress_level * 0.1)

    def repair(self, energy_available: float) -> float:
        """Repair membrane damage using available energy."""
        # Calculate repair amount
        repair_amount = min(energy_available * 0.05, 1.0 - self.integrity)
        self.integrity = min(1.0, self.integrity + repair_amount)
        return repair_amount

    def snapshot(self) -> Dict[str, float]:
        """Get a snapshot of current membrane state."""
        return {
            "integrity": self.integrity,
            "thickness": self.thickness,
            "permeability": self.permeability,
            "last_stress": self.last_stress
        }


@dataclass
class JASCoreState:
    """Snapshot of the core system state."""
    energy: Dict[str, float]
    membrane: Dict[str, float]
    tick: int
    generation: int
    id: str


class AutopoieticCore:
    """
    Minimal core that:
    - burns metabolic baseline
    - applies membrane stress fed from cognition (reptilian threat)
    - supports telemetry snapshot
    """

    def __init__(self, cfg: Dict[str, Any], ebt, vpm_manager, logger=None):
        self.logger = logger or log
        self.ebt = ebt
        self.vpm_manager = vpm_manager

        # Initialize energy pools
        e = cfg.get("energy", {})
        self.energy = EnergyPools(
            cognitive=e.get("initial_cognitive", 50.0),
            metabolic=e.get("initial_metabolic", 50.0),
            reserve=e.get("initial_reserve", 20.0),
        )

        # Initialize membrane
        m = cfg.get("membrane", {})
        self.membrane = Membrane(
            integrity=m.get("initial_integrity", 0.8),
            thickness=m.get("initial_thickness", 0.8),
            permeability=m.get("initial_permeability", 0.5),
        )

        # Core identifiers
        self.id = cfg.get("id", f"jas_{int(time.time())}")
        self.parent_id = cfg.get("parent_id", "")
        self.generation = int(cfg.get("generation", 0))
        self.tick = 0

        self.logger.info(f"JAS Core initialized (ID: {self.id}, Gen: {self.generation})")

    def cycle(self, sensory_input) -> Dict[str, Any]:
        """
        Execute a single autopoietic cycle.
        This is the core loop where the organism maintains itself.
        """
        # 1. Burn metabolic baseline (basic energy consumption)
        self.energy.consume("metabolic", 0.1)  # Baseline metabolic cost

        # 2. Apply stress from sensory input (simplified - in reality, this comes from cognition)
        # For demonstration, let's assume sensory_input has a 'stress' field
        stress_level = sensory_input.get("stress", 0.0)
        self.membrane.apply_stress(stress_level)

        # 3. Repair membrane if energy allows
        if self.energy.level("reserve") > 5.0:
            self.membrane.repair(self.energy.level("reserve"))

        # 4. Update tick count
        self.tick += 1

        # 5. Collect vital signs for telemetry
        vitals = self.snapshot()

        # 6. Return vitals (could be published via telemetry)
        return vitals

    def snapshot(self) -> Dict[str, Any]:
        """Create a snapshot of the core state for telemetry."""
        return {
            "id": self.id,
            "generation": self.generation,
            "tick": self.tick,
            "energy": self.energy.snapshot(),
            "membrane": self.membrane.snapshot(),
            "vpm_diversity": self.vpm_manager.diversity_score() if self.vpm_manager else 0.0,
            "vpm_count": self.vpm_manager.active_count() if self.vpm_manager else 0,
        }

    def attach_homeostasis(self, homeostasis):
        """Attach a homeostasis controller to this core."""
        self.homeostasis = homeostasis

    def get_health_score(self) -> float:
        """Calculate a basic health score."""
        energy_balance = (
            self.energy.level("cognitive") +
            self.energy.level("metabolic") +
            self.energy.level("reserve")
        ) / 3.0
        integrity_score = self.membrane.integrity
        return (energy_balance + integrity_score) / 2.0

    def get_vitals(self) -> Dict[str, Any]:
        """Get current vital signs."""
        return {
            "energy_balance": (
                self.energy.level("cognitive") +
                self.energy.level("metabolic") +
                self.energy.level("reserve")
            ) / 3.0,
            "boundary_integrity": self.membrane.integrity,
            "vpm_diversity": self.vpm_manager.diversity_score() if self.vpm_manager else 0.0,
            "health_score": self.get_health_score(),
            "tick": self.tick,
        }
