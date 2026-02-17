"""
CalibratedThresholds: Statistically-derived safety boundaries for governed self-improvement.

All thresholds are computed from baseline system behavior (μ ± kσ) to ensure:
- Data-driven (not arbitrary)
- System-specific (adapts to your scoring distribution)
- Defensible (statistical justification)
- Serializable (for experiment reproducibility)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CalibratedThresholds:
    """
    Statistically calibrated safety thresholds for governance layer.
    
    All thresholds derived from baseline system behavior:
    - Critical thresholds: μ ± 2σ (95% confidence interval)
    - Warning thresholds: μ ± 1σ (68% confidence interval)
    - Fixed thresholds: validated through pilot studies
    
    Immutable by design - thresholds must not change during experiment execution.
    """
    
    # ===== ENERGY THRESHOLDS (Hallucination Safety) =====
    energy_max: float  # μ + 2σ: Absolute failure boundary
    energy_warning: float  # μ + 1σ: Trigger conservative updates
    
    # ===== HRM THRESHOLDS (Reasoning Quality) =====
    hrm_min: float  # μ - 2σ: Minimum acceptable alignment
    
    # ===== EMBEDDING THRESHOLDS (Geometry Stability) =====
    margin_min: float  # μ - 2σ: Minimum embedding margin
    variance_min: float  # Fixed: Absolute floor for embedding diversity
    collapse_index_max: float  # Fixed: Max eigenvalue ratio (λ_max/λ_min)
    drift_max: float  # Fixed: Max angular drift per update (radians)
    
    # ===== PROVENANCE METADATA =====
    calibration_timestamp: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )
    baseline_episodes: int = 200
    baseline_system: str = "scalar_rl_baseline"
    statistical_method: str = "mean_plus_2std"
    schema_version: str = "1.0"
    
    # ===== VALIDATION =====
    def __post_init__(self):
        """Validate threshold relationships"""
        # Energy thresholds must be ordered
        if not (0.0 <= self.energy_warning < self.energy_max <= 1.0):
            raise ValueError(
                f"Invalid energy thresholds: warning={self.energy_warning}, "
                f"max={self.energy_max} (must satisfy 0 ≤ warning < max ≤ 1)"
            )
        
        # HRM threshold must be valid
        if not (0.0 <= self.hrm_min <= 1.0):
            raise ValueError(f"Invalid HRM min: {self.hrm_min} (must be in [0,1])")
        
        # Embedding thresholds must be positive
        if self.variance_min <= 0:
            raise ValueError(f"Variance min must be > 0, got {self.variance_min}")
        if self.collapse_index_max < 1.0:
            raise ValueError(
                f"Collapse index max must be ≥ 1.0, got {self.collapse_index_max}"
            )
        if not (0.0 < self.drift_max < np.pi):
            raise ValueError(
                f"Drift max must be in (0, π), got {self.drift_max}"
            )
        
        # Logical relationships
        if self.margin_min < 0 or self.margin_min > 1.0:
            raise ValueError(
                f"Margin min must be in [0,1], got {self.margin_min}"
            )
        
        logger.info(
            f"✓ CalibratedThresholds validated | "
            f"energy: [{self.energy_warning:.3f}, {self.energy_max:.3f}] | "
            f"hrm_min: {self.hrm_min:.3f} | "
            f"margin_min: {self.margin_min:.3f}"
        )
    
    # ===== SERIALIZATION =====
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary"""
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CalibratedThresholds":
        """Reconstruct from dictionary"""
        # Handle schema evolution
        kwargs = {
            "energy_max": data["energy_max"],
            "energy_warning": data["energy_warning"],
            "hrm_min": data["hrm_min"],
            "margin_min": data["margin_min"],
            "variance_min": data.get("variance_min", 0.3),
            "collapse_index_max": data.get("collapse_index_max", 10.0),
            "drift_max": data.get("drift_max", 0.15),
            "calibration_timestamp": data.get(
                "calibration_timestamp", 
                datetime.utcnow().isoformat()
            ),
            "baseline_episodes": data.get("baseline_episodes", 200),
            "baseline_system": data.get("baseline_system", "unknown"),
            "statistical_method": data.get("statistical_method", "mean_plus_2std"),
            "schema_version": data.get("schema_version", "1.0")
        }
        return cls(**kwargs)
    
    @classmethod
    def from_json(cls, json_str: str) -> "CalibratedThresholds":
        """Reconstruct from JSON string"""
        return cls.from_dict(json.loads(json_str))
    
    # ===== THRESHOLD CHECKING =====
    def is_energy_critical(self, energy: float) -> bool:
        """Check if energy exceeds critical threshold"""
        return energy > self.energy_max
    
    def is_energy_warning(self, energy: float) -> bool:
        """Check if energy exceeds warning threshold"""
        return energy > self.energy_warning
    
    def is_hrm_critical(self, hrm: float) -> bool:
        """Check if HRM alignment below critical threshold"""
        return hrm < self.hrm_min
    
    def is_margin_critical(self, margin: float) -> bool:
        """Check if embedding margin below critical threshold"""
        return margin < self.margin_min
    
    def is_variance_critical(self, variance: float) -> bool:
        """Check if embedding variance below critical threshold"""
        return variance < self.variance_min
    
    def is_collapse_critical(self, collapse_index: float) -> bool:
        """Check if collapse index exceeds critical threshold"""
        return collapse_index > self.collapse_index_max
    
    def is_drift_critical(self, drift: float) -> bool:
        """Check if angular drift exceeds critical threshold"""
        return drift > self.drift_max
    
    # ===== POLICY REGIME DETERMINATION =====
    def determine_regime(self, metrics: Dict[str, float]) -> str:
        """
        Determine policy regime based on current metrics.
        
        Returns: "safe", "warning", or "critical"
        """
        energy = metrics.get("energy_raw", 0.0)
        hrm = metrics.get("hrm_alignment", 1.0)
        margin = metrics.get("embedding_margin", 1.0)
        variance = metrics.get("embedding_variance", 1.0)
        collapse_index = metrics.get("collapse_index", 1.0)
        drift = metrics.get("angular_drift", 0.0)
        
        # Critical checks (any trigger critical regime)
        if (self.is_energy_critical(energy) or
            self.is_hrm_critical(hrm) or
            self.is_margin_critical(margin) or
            self.is_variance_critical(variance) or
            self.is_collapse_critical(collapse_index) or
            self.is_drift_critical(drift)):
            return "critical"
        
        # Warning checks
        if self.is_energy_warning(energy):
            return "warning"
        
        return "safe"
    
    # ===== HUMAN-READABLE REPORT =====
    def generate_report(self) -> str:
        """Generate human-readable threshold report"""
        lines = [
            "=" * 60,
            "CALIBRATED THRESHOLDS REPORT",
            "=" * 60,
            f"Calibration Time: {self.calibration_timestamp}",
            f"Baseline System: {self.baseline_system}",
            f"Episodes Used: {self.baseline_episodes}",
            f"Statistical Method: {self.statistical_method}",
            "",
            "┌─────────────────────────────────────────────────────────┐",
            "│ ENERGY THRESHOLDS (Hallucination Safety)                │",
            "├─────────────────────────────────────────────────────────┤",
            f"│ Warning:  {self.energy_warning:6.3f} (μ + 1σ)                     │",
            f"│ Critical: {self.energy_max:6.3f} (μ + 2σ)                     │",
            "├─────────────────────────────────────────────────────────┤",
            "│ HRM THRESHOLDS (Reasoning Quality)                      │",
            "├─────────────────────────────────────────────────────────┤",
            f"│ Minimum:  {self.hrm_min:6.3f} (μ - 2σ)                     │",
            "├─────────────────────────────────────────────────────────┤",
            "│ EMBEDDING THRESHOLDS (Geometry Stability)               │",
            "├─────────────────────────────────────────────────────────┤",
            f"│ Margin Min:      {self.margin_min:6.3f} (μ - 2σ)          │",
            f"│ Variance Min:    {self.variance_min:6.3f} (fixed)         │",
            f"│ Collapse Max:    {self.collapse_index_max:6.3f} (fixed)   │",
            f"│ Drift Max:       {self.drift_max:6.3f} rad (fixed)        │",
            "└─────────────────────────────────────────────────────────┘",
            "",
            "Thresholds derived from baseline system behavior.",
            "Critical violations trigger immediate governance intervention.",
            "=" * 60
        ]
        return "\n".join(lines)
    
    def __str__(self) -> str:
        return (
            f"CalibratedThresholds("
            f"energy:[{self.energy_warning:.3f}, {self.energy_max:.3f}], "
            f"hrm_min:{self.hrm_min:.3f}, "
            f"margin_min:{self.margin_min:.3f}, "
            f"var_min:{self.variance_min:.3f})"
        )
    
    def __repr__(self) -> str:
        return self.__str__()


# ============================================================================
# FACTORY FUNCTIONS FOR COMMON SCENARIOS
# ============================================================================

def create_from_baseline_stats(
    energy_stats: Dict[str, float],
    hrm_stats: Dict[str, float],
    margin_stats: Dict[str, float],
    baseline_episodes: int = 200,
    baseline_system: str = "scalar_rl_baseline"
) -> CalibratedThresholds:
    """
    Create thresholds from pre-computed baseline statistics.
    
    Args:
        energy_stats: {"mean": float, "std": float}
        hrm_stats: {"mean": float, "std": float}
        margin_stats: {"mean": float, "std": float}
        baseline_episodes: Number of episodes used for calibration
        baseline_system: Identifier for baseline system
    
    Returns:
        CalibratedThresholds instance
    """
    return CalibratedThresholds(
        energy_max=energy_stats["mean"] + 2 * energy_stats["std"],
        energy_warning=energy_stats["mean"] + 1 * energy_stats["std"],
        hrm_min=hrm_stats["mean"] - 2 * hrm_stats["std"],
        margin_min=margin_stats["mean"] - 2 * margin_stats["std"],
        variance_min=0.3,  # Fixed based on embedding geometry studies
        collapse_index_max=10.0,  # Fixed based on eigenvalue ratio analysis
        drift_max=0.15,  # Fixed based on angular drift studies (≈8.6 degrees)
        baseline_episodes=baseline_episodes,
        baseline_system=baseline_system,
        statistical_method="mean_plus_2std"
    )


def create_conservative_thresholds() -> CalibratedThresholds:
    """
    Create conservative thresholds for high-stakes applications.
    
    Tighter bounds than statistical calibration.
    """
    return CalibratedThresholds(
        energy_max=0.40,
        energy_warning=0.30,
        hrm_min=0.70,
        margin_min=0.50,
        variance_min=0.4,
        collapse_index_max=8.0,
        drift_max=0.10,
        baseline_system="conservative_preset",
        statistical_method="domain_expert"
    )


def create_permissive_thresholds() -> CalibratedThresholds:
    """
    Create permissive thresholds for exploratory research.
    
    Wider bounds to allow more learning velocity.
    """
    return CalibratedThresholds(
        energy_max=0.60,
        energy_warning=0.50,
        hrm_min=0.50,
        margin_min=0.30,
        variance_min=0.2,
        collapse_index_max=15.0,
        drift_max=0.25,
        baseline_system="permissive_preset",
        statistical_method="domain_expert"
    )


# ============================================================================
# VALIDATION UTILITIES
# ============================================================================

def validate_thresholds_against_distribution(
    thresholds: CalibratedThresholds,
    energy_samples: np.ndarray,
    hrm_samples: np.ndarray,
    margin_samples: np.ndarray
) -> Dict[str, Any]:
    """
    Validate thresholds against actual distribution.
    
    Returns diagnostic report showing:
    - % of samples above/below thresholds
    - Statistical coverage
    - Potential calibration issues
    """
    report = {
        "energy": {
            "warning_violation_pct": np.mean(energy_samples > thresholds.energy_warning) * 100,
            "critical_violation_pct": np.mean(energy_samples > thresholds.energy_max) * 100,
            "mean": np.mean(energy_samples),
            "std": np.std(energy_samples)
        },
        "hrm": {
            "critical_violation_pct": np.mean(hrm_samples < thresholds.hrm_min) * 100,
            "mean": np.mean(hrm_samples),
            "std": np.std(hrm_samples)
        },
        "margin": {
            "critical_violation_pct": np.mean(margin_samples < thresholds.margin_min) * 100,
            "mean": np.mean(margin_samples),
            "std": np.std(margin_samples)
        },
        "calibration_quality": "good"
    }
    
    # Flag potential issues
    issues = []
    if report["energy"]["critical_violation_pct"] > 5.0:
        issues.append("Energy critical threshold too strict (>5% baseline violations)")
    if report["hrm"]["critical_violation_pct"] > 5.0:
        issues.append("HRM critical threshold too strict (>5% baseline violations)")
    if report["margin"]["critical_violation_pct"] > 5.0:
        issues.append("Margin critical threshold too strict (>5% baseline violations)")
    
    if issues:
        report["calibration_quality"] = "needs_adjustment"
        report["issues"] = issues
    
    return report


# ============================================================================
# EXPORT
# ============================================================================

__all__ = [
    "CalibratedThresholds",
    "create_from_baseline_stats",
    "create_conservative_thresholds",
    "create_permissive_thresholds",
    "validate_thresholds_against_distribution"
]