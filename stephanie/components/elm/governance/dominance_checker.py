"""
DominanceChecker: Validates strict multi-objective improvement across critical axes.

Ensures:
- No reward hacking (trading one axis for another)
- Direction-aware improvement (higher/lower is better)
- Configurable critical axis sets
- Detailed failure diagnostics
- Trace-native comparison via ScoreBundle.diff()

Critical for preventing unsafe updates in governed self-improvement.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from stephanie.data.score_bundle import ScoreBundle
from stephanie.components.elm.governance.signal_extractor import (
    AxisDirection,
    DIMENSION_TO_AXIS,
    AXIS_SEMANTICS
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DominanceResult:
    """Structured dominance validation result"""
    is_dominant: bool
    failed_axes: List[str] = field(default_factory=list)
    passed_axes: List[str] = field(default_factory=list)
    delta_summary: Dict[str, float] = field(default_factory=dict)
    failure_reasons: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, any]:
        return {
            "is_dominant": self.is_dominant,
            "failed_axes": self.failed_axes,
            "passed_axes": self.passed_axes,
            "delta_summary": self.delta_summary,
            "failure_reasons": self.failure_reasons
        }
    
    def __bool__(self) -> bool:
        return self.is_dominant


class DominanceChecker:
    """
    Validates strict Pareto dominance between ScoreBundle instances.
    
    Design principles:
    - Direction-aware improvement semantics
    - Configurable critical axis sets
    - Detailed failure diagnostics
    - Zero tolerance for safety axis regression
    - Trace-native comparison via ScoreBundle.diff()
    
    Usage:
        checker = DominanceChecker(
            critical_dimensions=["alignment", "energy", "margin"],
            safety_dimensions=["energy"]  # Zero-tolerance axes
        )
        
        result = checker.check(bundle_before, bundle_after)
        if result.is_dominant:
            system.commit_improvement(bundle_after)
        else:
            logger.warning(f"Dominance failed: {result.failure_reasons}")
    """
    
    def __init__(
        self,
        critical_dimensions: Optional[List[str]] = None,
        safety_dimensions: Optional[List[str]] = None,
        tolerance: float = 1e-6
    ):
        """
        Initialize dominance checker.
        
        Args:
            critical_dimensions: Dimensions requiring strict improvement
            safety_dimensions: Subset requiring zero-tolerance (no regression allowed)
            tolerance: Numerical tolerance for delta comparison
        """
        self.critical_dimensions = critical_dimensions or ["alignment", "energy", "margin"]
        self.safety_dimensions = safety_dimensions or ["energy"]  # Hallucination energy is safety-critical
        self.tolerance = tolerance
        
        # Validate dimension mappings
        self._validate_dimensions()
        
        logger.info(
            f"DominanceChecker initialized | "
            f"critical={self.critical_dimensions} | "
            f"safety={self.safety_dimensions}"
        )
    
    def _validate_dimensions(self):
        """Validate all dimensions map to known governance axes"""
        unknown = []
        for dim in self.critical_dimensions + self.safety_dimensions:
            if dim not in DIMENSION_TO_AXIS:
                unknown.append(dim)
        
        if unknown:
            raise ValueError(
                f"Unknown dimensions in dominance config: {unknown}. "
                f"Available dimensions: {list(DIMENSION_TO_AXIS.keys())}"
            )
    
    def check(
        self,
        bundle_before: ScoreBundle,
        bundle_after: ScoreBundle,
        strict_safety: bool = True
    ) -> DominanceResult:
        """
        Check if bundle_after dominates bundle_before on all critical dimensions.
        
        Args:
            bundle_before: Baseline ScoreBundle
            bundle_after: Candidate improved ScoreBundle
            strict_safety: If True, safety dimensions must show strict improvement (no tolerance)
            
        Returns:
            DominanceResult with detailed diagnostics
        """
        diff = bundle_after.diff(bundle_before)
        failed_axes = []
        passed_axes = []
        delta_summary = {}
        failure_reasons = []
        
        # Check each critical dimension
        for dim in self.critical_dimensions:
            if dim not in diff.get("dimensions", {}):
                failed_axes.append(dim)
                failure_reasons.append(f"Dimension '{dim}' missing in diff comparison")
                continue
            
            dim_diff = diff["dimensions"][dim]
            delta = dim_diff.get("score_delta", 0.0)
            delta_summary[dim] = delta
            
            # Map to governance axis
            axis = DIMENSION_TO_AXIS[dim]
            direction = AXIS_SEMANTICS[axis]
            
            # Determine if improvement occurred
            is_improved = self._is_improvement(delta, direction, dim in self.safety_dimensions and strict_safety)
            
            if is_improved:
                passed_axes.append(dim)
            else:
                failed_axes.append(dim)
                direction_str = "increase" if direction == AxisDirection.HIGHER_IS_BETTER else "decrease"
                failure_reasons.append(
                    f"Dimension '{dim}' failed: expected {direction_str} (delta={delta:+.4f})"
                )
        
        is_dominant = len(failed_axes) == 0
        
        if not is_dominant:
            logger.debug(
                f"Dominance check failed | "
                f"passed={passed_axes} | "
                f"failed={failed_axes} | "
                f"reasons={failure_reasons}"
            )
        
        return DominanceResult(
            is_dominant=is_dominant,
            failed_axes=failed_axes,
            passed_axes=passed_axes,
            delta_summary=delta_summary,
            failure_reasons=failure_reasons
        )
    
    def _is_improvement(
        self,
        delta: float,
        direction: AxisDirection,
        is_safety_axis: bool
    ) -> bool:
        """
        Determine if delta represents improvement given direction semantics.
        
        Args:
            delta: Score delta (after - before)
            direction: Axis direction semantics
            is_safety_axis: If True, apply zero-tolerance check
            
        Returns:
            True if delta represents improvement
        """
        if direction == AxisDirection.HIGHER_IS_BETTER:
            # For higher-is-better: delta must be positive
            if is_safety_axis:
                return delta > 0  # Strict: must improve
            return delta > -self.tolerance  # Allow tiny numerical noise
        else:  # LOWER_IS_BETTER
            # For lower-is-better: delta must be negative (value decreased)
            if is_safety_axis:
                return delta < 0  # Strict: must improve
            return delta < self.tolerance  # Allow tiny numerical noise
    
    def get_critical_axes(self) -> List[str]:
        """Get current critical dimension configuration"""
        return list(self.critical_dimensions)
    
    def add_critical_dimension(self, dimension: str) -> None:
        """Add dimension to critical set (runtime configuration)"""
        if dimension not in DIMENSION_TO_AXIS:
            raise ValueError(f"Unknown dimension: {dimension}")
        
        if dimension not in self.critical_dimensions:
            self.critical_dimensions.append(dimension)
            logger.info(f"Added critical dimension: {dimension}")
    
    def remove_critical_dimension(self, dimension: str) -> None:
        """Remove dimension from critical set"""
        if dimension in self.critical_dimensions:
            self.critical_dimensions.remove(dimension)
            logger.info(f"Removed critical dimension: {dimension}")
    
    def __str__(self) -> str:
        return (
            f"DominanceChecker("
            f"critical={len(self.critical_dimensions)}, "
            f"safety={len(self.safety_dimensions)})"
        )
    
    def __repr__(self) -> str:
        return self.__str__()


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_strict_dominance_checker() -> DominanceChecker:
    """
    Create dominance checker with strict safety constraints.
    
    Suitable for high-stakes applications where hallucination safety is non-negotiable.
    """
    return DominanceChecker(
        critical_dimensions=["alignment", "energy", "margin", "context_fidelity"],
        safety_dimensions=["energy", "alignment"],  # Zero tolerance on safety axes
        tolerance=1e-8  # Extremely strict numerical tolerance
    )


def create_research_dominance_checker() -> DominanceChecker:
    """
    Create dominance checker optimized for research settings.
    
    More permissive on non-safety axes to allow exploration.
    """
    return DominanceChecker(
        critical_dimensions=["alignment", "energy", "margin"],
        safety_dimensions=["energy"],  # Only energy is safety-critical
        tolerance=1e-4  # Allow minor numerical fluctuations
    )


# ============================================================================
# EXPORT
# ============================================================================

__all__ = [
    "DominanceChecker",
    "DominanceResult",
    "create_strict_dominance_checker",
    "create_research_dominance_checker"
]