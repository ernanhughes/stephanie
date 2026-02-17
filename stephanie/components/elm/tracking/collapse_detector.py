"""
CollapseDetector: Real-time detection of representation collapse and instability.

Monitors 6 critical failure modes:
1. Energy spiral (hallucination instability)
2. HRM alignment collapse (reasoning quality degradation)
3. Embedding margin collapse (geometric failure)
4. Variance collapse (manifold degeneracy)
5. Collapse index explosion (eigenvalue distortion)
6. Angular drift violation (update instability)

All thresholds derived from CalibratedThresholds.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from ..core.thresholds import CalibratedThresholds

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FailureEvent:
    """Structured failure event with diagnostic details"""
    episode: int
    failure_type: str
    metric_name: str
    metric_value: float
    threshold_value: float
    severity: str  # "warning", "critical"
    description: str
    timestamp: float = field(default_factory=lambda: __import__('time').time())
    
    def to_dict(self) -> Dict[str, any]:
        return {
            "episode": self.episode,
            "failure_type": self.failure_type,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "threshold_value": self.threshold_value,
            "severity": self.severity,
            "description": self.description,
            "timestamp": self.timestamp
        }
    
    def __str__(self) -> str:
        return (
            f"[{self.severity.upper()}] {self.failure_type} | "
            f"{self.metric_name}={self.metric_value:.4f} "
            f"(threshold={self.threshold_value:.4f}) | "
            f"{self.description}"
        )


class CollapseDetector:
    """
    Real-time collapse detection system.
    
    Monitors governance metrics against calibrated thresholds.
    Detects 6 critical failure modes with severity levels.
    Maintains failure history for diagnostics.
    
    Usage:
        detector = CollapseDetector(thresholds)
        
        # Check metrics every episode
        failure = detector.check_failure(
            episode=42,
            metrics={
                "energy_raw": 0.52,
                "hrm_alignment": 0.68,
                "embedding_margin": 0.41,
                "embedding_variance": 0.28,
                "collapse_index": 12.3,
                "angular_drift": 0.18
            }
        )
        
        if failure:
            logger.warning(f"COLLAPSE DETECTED: {failure}")
            if failure.severity == "critical":
                system.freeze_embedding_updates()
    """
    
    # Failure type constants
    ENERGY_SPIRAL = "energy_spiral"
    HRM_COLLAPSE = "hrm_collapse"
    MARGIN_COLLAPSE = "margin_collapse"
    VARIANCE_COLLAPSE = "variance_collapse"
    COLLAPSE_INDEX_EXPLOSION = "collapse_index_explosion"
    ANGULAR_DRIFT_VIOLATION = "angular_drift_violation"
    
    def __init__(
        self,
        thresholds: "CalibratedThresholds",  # Forward reference
        consecutive_failures_required: int = 1,
        warning_buffer: float = 0.05  # 5% buffer below critical threshold for warnings
    ):
        """
        Initialize collapse detector.
        
        Args:
            thresholds: Calibrated safety thresholds
            consecutive_failures_required: Failures must occur this many times consecutively to trigger
            warning_buffer: Fraction below critical threshold where warnings activate
        """
        self.thresholds = thresholds
        self.consecutive_failures_required = consecutive_failures_required
        self.warning_buffer = warning_buffer
        
        # Failure tracking state
        self._failure_streaks: Dict[str, int] = {
            self.ENERGY_SPIRAL: 0,
            self.HRM_COLLAPSE: 0,
            self.MARGIN_COLLAPSE: 0,
            self.VARIANCE_COLLAPSE: 0,
            self.COLLAPSE_INDEX_EXPLOSION: 0,
            self.ANGULAR_DRIFT_VIOLATION: 0
        }
        
        # Failure history (last 100 events)
        self._failure_history: List[FailureEvent] = []
        self._max_history = 100
        
        logger.info(
            f"CollapseDetector initialized | "
            f"thresholds={thresholds} | "
            f"consecutive_required={consecutive_failures_required}"
        )
    
    def check_failure(
        self,
        episode: int,
        metrics: Dict[str, float]
    ) -> Optional[FailureEvent]:
        """
        Check current metrics against thresholds.
        
        Returns:
            FailureEvent if critical failure detected, None otherwise
        """
        # Check each failure mode
        checks = [
            self._check_energy_spiral(episode, metrics),
            self._check_hrm_collapse(episode, metrics),
            self._check_margin_collapse(episode, metrics),
            self._check_variance_collapse(episode, metrics),
            self._check_collapse_index_explosion(episode, metrics),
            self._check_angular_drift_violation(episode, metrics)
        ]
        
        # Return first critical failure (if any)
        for failure in checks:
            if failure and failure.severity == "critical":
                self._record_failure(failure)
                return failure
        
        # No critical failures
        self._reset_streaks()
        return None
    
    def _check_energy_spiral(
        self,
        episode: int,
        metrics: Dict[str, float]
    ) -> Optional[FailureEvent]:
        """Check for hallucination energy spiral"""
        energy = metrics.get("energy_raw", 0.0)
        
        # Critical: exceeds absolute max
        if energy > self.thresholds.energy_max:
            return FailureEvent(
                episode=episode,
                failure_type=self.ENERGY_SPIRAL,
                metric_name="energy_raw",
                metric_value=energy,
                threshold_value=self.thresholds.energy_max,
                severity="critical",
                description=f"Energy exceeded critical threshold ({energy:.3f} > {self.thresholds.energy_max:.3f})"
            )
        
        # Warning: exceeds warning threshold
        if energy > self.thresholds.energy_warning:
            return FailureEvent(
                episode=episode,
                failure_type=self.ENERGY_SPIRAL,
                metric_name="energy_raw",
                metric_value=energy,
                threshold_value=self.thresholds.energy_warning,
                severity="warning",
                description=f"Energy in warning zone ({energy:.3f} > {self.thresholds.energy_warning:.3f})"
            )
        
        return None
    
    def _check_hrm_collapse(
        self,
        episode: int,
        metrics: Dict[str, float]
    ) -> Optional[FailureEvent]:
        """Check for HRM alignment collapse"""
        hrm = metrics.get("hrm_alignment", 1.0)
        
        if hrm < self.thresholds.hrm_min:
            return FailureEvent(
                episode=episode,
                failure_type=self.HRM_COLLAPSE,
                metric_name="hrm_alignment",
                metric_value=hrm,
                threshold_value=self.thresholds.hrm_min,
                severity="critical",
                description=f"HRM alignment collapsed ({hrm:.3f} < {self.thresholds.hrm_min:.3f})"
            )
        
        return None
    
    def _check_margin_collapse(
        self,
        episode: int,
        metrics: Dict[str, float]
    ) -> Optional[FailureEvent]:
        """Check for embedding margin collapse"""
        margin = metrics.get("embedding_margin", 1.0)
        
        if margin < self.thresholds.margin_min:
            return FailureEvent(
                episode=episode,
                failure_type=self.MARGIN_COLLAPSE,
                metric_name="embedding_margin",
                metric_value=margin,
                threshold_value=self.thresholds.margin_min,
                severity="critical",
                description=f"Embedding margin collapsed ({margin:.3f} < {self.thresholds.margin_min:.3f})"
            )
        
        return None
    
    def _check_variance_collapse(
        self,
        episode: int,
        metrics: Dict[str, float]
    ) -> Optional[FailureEvent]:
        """Check for embedding variance collapse"""
        variance = metrics.get("embedding_variance", 1.0)
        
        if variance < self.thresholds.variance_min:
            return FailureEvent(
                episode=episode,
                failure_type=self.VARIANCE_COLLAPSE,
                metric_name="embedding_variance",
                metric_value=variance,
                threshold_value=self.thresholds.variance_min,
                severity="critical",
                description=f"Embedding variance collapsed ({variance:.3f} < {self.thresholds.variance_min:.3f})"
            )
        
        return None
    
    def _check_collapse_index_explosion(
        self,
        episode: int,
        metrics: Dict[str, float]
    ) -> Optional[FailureEvent]:
        """Check for collapse index explosion (manifold distortion)"""
        collapse_index = metrics.get("collapse_index", 1.0)
        
        if collapse_index > self.thresholds.collapse_index_max:
            return FailureEvent(
                episode=episode,
                failure_type=self.COLLAPSE_INDEX_EXPLOSION,
                metric_name="collapse_index",
                metric_value=collapse_index,
                threshold_value=self.thresholds.collapse_index_max,
                severity="critical",
                description=f"Collapse index exploded ({collapse_index:.2f} > {self.thresholds.collapse_index_max:.2f})"
            )
        
        return None
    
    def _check_angular_drift_violation(
        self,
        episode: int,
        metrics: Dict[str, float]
    ) -> Optional[FailureEvent]:
        """Check for excessive angular drift in embedding updates"""
        drift = metrics.get("angular_drift", 0.0)
        
        if drift > self.thresholds.drift_max:
            return FailureEvent(
                episode=episode,
                failure_type=self.ANGULAR_DRIFT_VIOLATION,
                metric_name="angular_drift",
                metric_value=drift,
                threshold_value=self.thresholds.drift_max,
                severity="critical",
                description=f"Angular drift exceeded limit ({drift:.3f} rad > {self.thresholds.drift_max:.3f} rad)"
            )
        
        return None
    
    def _record_failure(self, failure: FailureEvent) -> None:
        """Record failure event and update streaks"""
        # Update streak for this failure type
        self._failure_streaks[failure.failure_type] += 1
        
        # Add to history
        self._failure_history.append(failure)
        if len(self._failure_history) > self._max_history:
            self._failure_history.pop(0)
        
        # Log failure
        log_fn = logger.warning if failure.severity == "warning" else logger.error
        log_fn(f"COLLAPSE DETECTOR: {failure}")
    
    def _reset_streaks(self) -> None:
        """Reset all failure streaks (called when no failures detected)"""
        for key in self._failure_streaks:
            self._failure_streaks[key] = 0
    
    def get_failure_history(self, last_n: int = 10) -> List[FailureEvent]:
        """Get recent failure events"""
        return self._failure_history[-last_n:] if self._failure_history else []
    
    def get_streak(self, failure_type: str) -> int:
        """Get current streak count for failure type"""
        return self._failure_streaks.get(failure_type, 0)
    
    def generate_diagnostic_report(self) -> str:
        """
        Generate diagnostic report of recent failures.
        
        Returns:
            Formatted string with failure analysis
        """
        lines = [
            "=" * 70,
            "COLLAPSE DETECTOR DIAGNOSTIC REPORT",
            "=" * 70,
            "Current Thresholds:",
            f"  Energy Max: {self.thresholds.energy_max:.3f}",
            f"  HRM Min: {self.thresholds.hrm_min:.3f}",
            f"  Margin Min: {self.thresholds.margin_min:.3f}",
            f"  Variance Min: {self.thresholds.variance_min:.3f}",
            f"  Collapse Index Max: {self.thresholds.collapse_index_max:.2f}",
            f"  Drift Max: {self.thresholds.drift_max:.3f} rad",
            "",
            "Failure Streaks (consecutive episodes):"
        ]
        
        for failure_type, streak in self._failure_streaks.items():
            if streak > 0:
                lines.append(f"  {failure_type}: {streak} episodes")
        
        if not any(self._failure_streaks.values()):
            lines.append("  None (system stable)")
        
        lines.append("")
        lines.append("Recent Failures (last 5):")
        
        if self._failure_history:
            for failure in self._failure_history[-5:]:
                lines.append(f"  {failure}")
        else:
            lines.append("  None")
        
        lines.append("=" * 70)
        return "\n".join(lines)
    
    def __str__(self) -> str:
        active = sum(1 for s in self._failure_streaks.values() if s > 0)
        return f"CollapseDetector(active_failures={active})"
    
    def __repr__(self) -> str:
        return self.__str__()