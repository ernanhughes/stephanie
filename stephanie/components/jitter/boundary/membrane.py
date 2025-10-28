# stephanie/components/jitter/boundary/membrane.py

"""  
# membrane.py
# ===========
# Implementation of the semipermeable membrane for boundary definition.
#
# This module implements Maturana & Varela's concept of a membrane that:
# - Defines the boundary of the autopoietic system
# - Maintains system identity through selective permeability
# - Adapts to environmental perturbations while preserving integrity
# - Produces itself through organizationally closed production
#
# Key Features:
# - Integrity, thickness, and permeability properties
# - Stress assessment and response mechanisms
# - Identity preservation through EBT compatibility
# - Configuration validation with Pydantic
# - Circuit breaker pattern for resilience (per-instance)
# - Comprehensive telemetry and monitoring
# - SSP integration hooks
# - Performance optimizations
"""
from __future__ import annotations

from typing import Dict, Any, Optional
import numpy as np
import logging
import time
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator
from enum import Enum
from functools import wraps

log = logging.getLogger("stephanie.jitter.boundary.membrane")


class BoundaryState(str, Enum):
    """States of the boundary membrane"""
    HEALTHY = "healthy"
    STRESSED = "stressed"
    DAMAGED = "damaged"
    CRITICAL = "critical"
    FAILED = "failed"


class MembraneConfig(BaseModel):
    """Validated configuration for Membrane.

    NOTE on thresholds:
    We compare integrity against (1.0 - threshold). For example, with
    stress_threshold=0.3, integrity >= 0.7 is considered HEALTHY.
    """
    initial_integrity: float = Field(0.8, ge=0.1, le=1.0, description="Initial boundary integrity")
    initial_thickness: float = Field(0.8, ge=0.1, le=1.0, description="Initial membrane thickness")
    initial_permeability: float = Field(0.5, ge=0.1, le=0.9, description="Initial permeability level")
    min_integrity: float = Field(0.1, ge=0.05, le=0.3, description="Minimum integrity threshold")
    repair_rate: float = Field(0.05, ge=0.01, le=0.1, description="Base repair rate per tick")
    stress_threshold: float = Field(0.3, ge=0.1, le=0.5, description="Threshold for stress detection")
    damage_threshold: float = Field(0.6, ge=0.4, le=0.8, description="Threshold for damage detection")
    critical_threshold: float = Field(0.85, ge=0.7, le=0.95, description="Threshold for critical state")
    permeability_decay: float = Field(0.95, ge=0.8, le=0.99, description="Decay factor for permeability")

    @validator('min_integrity')
    def validate_min_max_integrity(cls, v, values):
        if 'initial_integrity' in values and v >= values['initial_integrity']:
            raise ValueError('min_integrity must be less than initial_integrity')
        return v


class CircuitBreakerState:
    """States for circuit breaker pattern"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for service resilience.

    Recommended usage (per-instance, avoids cross-talk):

        def some_method(self, x):
            @self.circuit_breaker
            def _inner(x):
                # do work
                return result
            return _inner(x)
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_attempts: int = 3
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_attempts = half_open_attempts

        self.state = CircuitBreakerState.CLOSED
        self.failures = 0
        self.last_failure_time = 0.0
        self.half_open_successes = 0
        self.logger = logging.getLogger(f"{__name__}.circuit_breaker")

    def __call__(self, func: callable) -> callable:
        """Decorator/wrapper implementation (stateful per-instance)."""

        @wraps(func)
        def wrapper(*args, **kwargs):
            # OPEN → maybe HALF_OPEN
            if self.state == CircuitBreakerState.OPEN:
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.logger.info("Circuit breaker transitioning to HALF_OPEN state")
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.half_open_successes = 0
                else:
                    self.logger.warning("Circuit breaker is OPEN - skipping call")
                    return None  # no-op while OPEN

            try:
                result = func(*args, **kwargs)

                # HALF_OPEN → CLOSED on sufficient successes
                if self.state == CircuitBreakerState.HALF_OPEN:
                    self.half_open_successes += 1
                    if self.half_open_successes >= self.half_open_attempts:
                        self.logger.info("Circuit breaker transitioning to CLOSED state")
                        self.state = CircuitBreakerState.CLOSED
                        self.failures = 0
                        self.half_open_successes = 0

                return result

            except Exception as e:
                self.failures += 1
                self.last_failure_time = time.time()
                self.logger.error(f"Service failure: {e!s}; failures={self.failures}")

                if self.failures >= self.failure_threshold:
                    self.logger.warning("Circuit breaker transitioning to OPEN state")
                    self.state = CircuitBreakerState.OPEN

                raise

        return wrapper

    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state for monitoring"""
        return {
            "state": self.state,
            "failures": self.failures,
            "last_failure_time": self.last_failure_time,
            "half_open_successes": self.half_open_successes,
        }


@dataclass
class MembraneSnapshot:
    """Snapshot of membrane state for telemetry and reproduction"""
    integrity: float
    thickness: float
    permeability: float
    stress: float
    state: str
    repair_rate: float
    last_stress: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class MembraneMetrics:
    """Metrics for membrane performance"""
    integrity_stability: float = 0.5
    repair_efficiency: float = 0.5
    stress_resilience: float = 0.5
    identity_preservation: float = 0.5
    processing_time_ms: float = 0.0


class Membrane:
    """
    Implementation of the semipermeable membrane for boundary definition.

    Key Features:
    - Integrity, thickness, and permeability properties
    - Stress assessment and response mechanisms
    - Identity preservation through EBT compatibility
    - Configuration validation with Pydantic
    - Circuit breaker pattern for resilience (per-instance)
    - Comprehensive telemetry and monitoring
    - SSP integration hooks
    - Performance optimizations
    """

    def __init__(self, cfg: Dict[str, Any]):
        try:
            self.config = MembraneConfig(**cfg)
            log.info("Membrane configuration validated successfully")
        except Exception as e:
            log.error(f"Configuration validation failed: {e!s}")
            self.config = MembraneConfig()  # safe defaults

        # Core state
        self.integrity = self.config.initial_integrity
        self.thickness = self.config.initial_thickness
        self.permeability = self.config.initial_permeability

        self.last_stress = 0.0
        self.stress_history = []

        # Per-instance circuit breaker
        self.circuit_breaker = CircuitBreaker()

        # Metrics + perf
        self.metrics = MembraneMetrics()
        self.processing_times = []
        self.max_processing_history = 100

        log.info("Membrane initialized with semipermeable boundary features")

    # ------------------------------
    # Stress / Repair / Permeability
    # ------------------------------

    def apply_stress(self, stress_level: float) -> Optional[float]:
        """
        Apply environmental stress to the membrane.

        Returns resulting integrity, or None if circuit is OPEN.
        """
        start_time = time.time()

        @self.circuit_breaker
        def _inner(level: float) -> float:
            initial_integrity = self.integrity

            # Effective stress increases with permeability and decreases with thickness
            effective_stress = level * (1.0 - self.thickness) * self.permeability

            # Update integrity (decreases with stress)
            self.integrity = max(self.config.min_integrity, self.integrity - effective_stress)

            # Record stress event
            self.last_stress = level
            self.stress_history.append({
                "timestamp": time.time(),
                "stress_level": level,
                "effective_stress": effective_stress,
                "integrity_before": initial_integrity,
                "integrity_after": self.integrity,
            })
            if len(self.stress_history) > 100:
                self.stress_history.pop(0)

            self._update_metrics()
            return self.integrity

        try:
            result = _inner(stress_level)
            # Perf tracking
            self._record_processing_time(time.time() - start_time)
            log.debug(f"Applied stress (level={stress_level:.3f}, integrity={self.integrity:.3f})")
            return result
        except Exception:
            log.error("Error applying stress", exc_info=True)
            return self.integrity

    def repair(self, energy_available: float) -> Optional[float]:
        """
        Repair membrane damage using available energy.

        Returns repair amount, or None if circuit is OPEN.
        """
        start_time = time.time()

        @self.circuit_breaker
        def _inner(energy: float) -> float:
            repair_amount = energy * self.config.repair_rate
            integrity_deficit = 1.0 - self.integrity
            repair_amount = min(repair_amount, integrity_deficit)

            self.integrity = min(1.0, self.integrity + repair_amount)
            self._update_metrics()
            return repair_amount

        try:
            amount = _inner(energy_available)
            self._record_processing_time(time.time() - start_time)
            log.debug(f"Performed repair (amount={float(amount):.3f}, integrity={self.integrity:.3f})")
            return amount
        except Exception:
            log.error("Error performing repair", exc_info=True)
            return 0.0

    def is_permeable(self, embedding: np.ndarray) -> Optional[bool]:
        """
        Check if the membrane is permeable to a given embedding.

        Returns True/False, or None if circuit is OPEN.
        """
        start_time = time.time()

        @self.circuit_breaker
        def _inner(vec: np.ndarray) -> bool:
            norm = float(np.linalg.norm(vec))
            if norm == 0.0:
                return False

            # Normalize and compare to identity vector
            v = vec / norm
            identity_vector = np.ones_like(v) / np.sqrt(len(v))
            compatibility = float(np.dot(v, identity_vector))

            # Corrected permeability integration:
            # higher compatibility + higher integrity + higher permeability -> more pass-through
            permeability_score = (
                0.5 * compatibility +
                0.3 * self.integrity +
                0.2 * self.permeability
            )
            return permeability_score > 0.5

        try:
            allowed = _inner(embedding)
            self._record_processing_time(time.time() - start_time)
            log.debug(f"Checked permeability (permeable={allowed})")
            return allowed
        except Exception:
            log.error("Error checking permeability", exc_info=True)
            return False

    # -------------
    # State / Views
    # -------------

    def get_state(self) -> BoundaryState:
        """Get current boundary state based on integrity.

        Compares integrity to (1.0 - threshold) levels (see MembraneConfig doc).
        """
        if self.integrity >= 1.0 - self.config.stress_threshold:
            return BoundaryState.HEALTHY
        elif self.integrity >= 1.0 - self.config.damage_threshold:
            return BoundaryState.STRESSED
        elif self.integrity >= 1.0 - self.config.critical_threshold:
            return BoundaryState.DAMAGED
        elif self.integrity > self.config.min_integrity:
            return BoundaryState.CRITICAL
        else:
            return BoundaryState.FAILED

    def get_snapshot(self) -> MembraneSnapshot:
        """Get current membrane snapshot for telemetry and reproduction"""
        return MembraneSnapshot(
            integrity=self.integrity,
            thickness=self.thickness,
            permeability=self.permeability,
            stress=self.last_stress,
            state=self.get_state().value,
            repair_rate=self.config.repair_rate,
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get current membrane metrics for monitoring and adaptation"""
        return {
            "integrity_stability": self.metrics.integrity_stability,
            "repair_efficiency": self.metrics.repair_efficiency,
            "stress_resilience": self.metrics.stress_resilience,
            "identity_preservation": self.metrics.identity_preservation,
            "processing_time_ms": self.metrics.processing_time_ms,
            "circuit_breaker_state": self.circuit_breaker.get_state(),
            "stress_history_size": len(self.stress_history),
        }

    def get_ssp_integration_metrics(self) -> Dict[str, float]:
        """
        Metrics for SSP reward shaping / context.
        """
        m = self.get_metrics()
        return {
            "boundary_integrity": float(self.integrity),
            "boundary_stability": float(m["integrity_stability"]),
            "repair_efficiency": float(m["repair_efficiency"]),
            "stress_resilience": float(m["stress_resilience"]),
            "identity_preservation": float(m["identity_preservation"]),
            "processing_efficiency": float(1.0 / (1.0 + m["processing_time_ms"])),
        }

    # -----------------------
    # Adaptation / Adjusters
    # -----------------------

    def adjust_thickness(self, delta: float):
        """Adjust membrane thickness based on environmental conditions."""
        self.thickness = max(0.1, min(1.0, self.thickness + delta * 0.1))
        log.debug(f"Adjusted membrane thickness: {self.thickness:.3f}")

    def adjust_permeability(self, delta: float):
        """Adjust membrane permeability based on environmental conditions."""
        self.permeability = max(
            0.1,
            min(0.9, self.permeability * self.config.permeability_decay + delta * 0.05)
        )
        log.debug(f"Adjusted membrane permeability: {self.permeability:.3f}")

    # ---------------
    # Internal helpers
    # ---------------

    def _update_metrics(self):
        """Update membrane metrics based on current state."""
        # Stability over recent integrity values
        if len(self.stress_history) >= 10:
            integrity_values = [s["integrity_after"] for s in self.stress_history[-10:]]
            self.metrics.integrity_stability = float(1.0 / (1.0 + np.var(integrity_values)))

        # Repair efficiency: average (Δintegrity)+ per unit stress
        if self.stress_history:
            repairs = [max(0.0, s["integrity_after"] - s["integrity_before"]) for s in self.stress_history]
            stresses = [float(s["stress_level"]) for s in self.stress_history]
            denom = sum(stresses)
            if denom > 0.0:
                self.metrics.repair_efficiency = float(sum(repairs) / denom)

        # Stress resilience: lower average recent stress → higher resilience
        recent_stress = [float(s["stress_level"]) for s in self.stress_history[-5:]] if self.stress_history else []
        if recent_stress:
            avg_stress = float(np.mean(recent_stress))
            self.metrics.stress_resilience = float(1.0 - min(1.0, avg_stress * 1.5))

        # Identity preservation prefers high integrity + lower permeability
        self.metrics.identity_preservation = float(self.integrity * (1.0 - self.permeability))

        # processing_time_ms is updated via _record_processing_time

    def _record_processing_time(self, seconds: float):
        self.processing_times.append(seconds)
        if len(self.processing_times) > self.max_processing_history:
            self.processing_times.pop(0)
        self.metrics.processing_time_ms = float(np.mean(self.processing_times) * 1000.0 if self.processing_times else 0.0)
