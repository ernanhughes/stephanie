"""
RetentionTracker: Measures persistence of improvements across time horizons.

Critical for distinguishing:
- Short-term reward spikes (unstable)
- Long-term durable improvements (valuable)

Tracks retention per axis with direction-aware delta computation.
Uses exponential moving average for stable scoring.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RetentionMetrics:
    """Structured retention metrics per axis and horizon"""
    axis: str
    horizon: int
    retention_score: float
    sample_count: int
    recent_deltas: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, any]:
        return {
            "axis": self.axis,
            "horizon": self.horizon,
            "retention_score": self.retention_score,
            "sample_count": self.sample_count,
            "recent_deltas_mean": np.mean(self.recent_deltas) if self.recent_deltas else 0.0
        }


class RetentionTracker:
    """
    Tracks long-term improvement retention across multiple horizons.
    
    Design principles:
    - Direction-aware delta computation (higher/lower is better)
    - Exponential moving average for stable scoring
    - Per-axis and per-horizon tracking
    - Statistical significance tracking (sample count)
    - Memory efficient (bounded history windows)
    
    Usage:
        tracker = RetentionTracker(
            critical_axes=["energy_raw", "hrm_alignment", "embedding_margin"],
            horizons=[5, 10, 20],
            alpha=0.1  # EMA smoothing factor
        )
        
        # Update every episode
        tracker.update(episode=42, metrics={
            "energy_raw": 0.28,
            "hrm_alignment": 0.85,
            "embedding_margin": 0.62
        })
        
        # Get current retention scores
        scores = tracker.get_scores()  # {"energy_raw": -0.02, "hrm_alignment": 0.03, ...}
        report = tracker.generate_report()  # Human-readable analysis
    """
    
    def __init__(
        self,
        critical_axes: List[str] = None,
        horizons: List[int] = None,
        alpha: float = 0.1,
        min_samples: int = 10
    ):
        """
        Initialize retention tracker.
        
        Args:
            critical_axes: List of axis names to track (must match metrics keys)
            horizons: Time horizons to evaluate retention (episodes)
            alpha: EMA smoothing factor (0.0 = no smoothing, 1.0 = only latest)
            min_samples: Minimum samples before reporting retention
        """
        self.critical_axes = critical_axes or [
            "energy_raw",        # Lower is better
            "hrm_alignment",     # Higher is better
            "embedding_margin"   # Higher is better
        ]
        self.horizons = sorted(horizons or [5, 10, 20])
        self.alpha = alpha
        self.min_samples = min_samples
        
        # Axis direction semantics (critical for delta computation)
        self.axis_directions: Dict[str, str] = {
            "energy_raw": "lower",      # Lower energy = better
            "hrm_alignment": "higher",  # Higher alignment = better
            "embedding_margin": "higher" # Higher margin = better
        }
        
        # History buffers: {axis: deque of (episode, value)}
        self.history: Dict[str, deque] = {
            axis: deque(maxlen=max(self.horizons) + 1)
            for axis in self.critical_axes
        }
        
        # Retention scores: {axis: {horizon: score}}
        self.retention_scores: Dict[str, Dict[int, float]] = {
            axis: {h: 0.0 for h in self.horizons}
            for axis in self.critical_axes
        }
        
        # Sample counts for statistical significance
        self.sample_counts: Dict[str, Dict[int, int]] = {
            axis: {h: 0 for h in self.horizons}
            for axis in self.critical_axes
        }
        
        # Recent deltas for diagnostics
        self.recent_deltas: Dict[str, Dict[int, deque]] = {
            axis: {h: deque(maxlen=50) for h in self.horizons}
            for axis in self.critical_axes
        }
        
        logger.info(
            f"RetentionTracker initialized | "
            f"axes={self.critical_axes} | "
            f"horizons={self.horizons} | "
            f"alpha={alpha}"
        )
    
    def update(self, episode: int, metrics: Dict[str, float]) -> None:
        """
        Update tracker with new episode metrics.
        
        Computes retention deltas for all horizons where sufficient history exists.
        Updates EMA retention scores.
        
        Args:
            episode: Current episode number
            metrics: Dict of metric values (must include critical axes)
        """
        # Store current values in history
        for axis in self.critical_axes:
            if axis in metrics:
                self.history[axis].append((episode, metrics[axis]))
        
        # Compute retention for each axis and horizon
        for axis in self.critical_axes:
            if len(self.history[axis]) < max(self.horizons) + 1:
                continue  # Not enough history yet
            
            current_value = metrics.get(axis)
            if current_value is None:
                continue
            
            # Compute retention for each horizon
            for horizon in self.horizons:
                if len(self.history[axis]) <= horizon:
                    continue
                
                # Get value from horizon episodes ago
                past_episode, past_value = self.history[axis][-horizon - 1]
                
                # Compute direction-aware delta (positive = improvement)
                delta = self._compute_delta(axis, current_value, past_value)
                
                # Update EMA retention score
                old_score = self.retention_scores[axis][horizon]
                new_score = self.alpha * delta + (1 - self.alpha) * old_score
                self.retention_scores[axis][horizon] = new_score
                
                # Update sample count
                self.sample_counts[axis][horizon] += 1
                
                # Store recent delta for diagnostics
                self.recent_deltas[axis][horizon].append(delta)
    
    def _compute_delta(self, axis: str, current: float, past: float) -> float:
        """
        Compute direction-aware improvement delta.
        
        Positive delta always means improvement, regardless of axis direction.
        
        Args:
            axis: Axis name
            current: Current value
            past: Value from horizon episodes ago
            
        Returns:
            Delta where positive = improvement
        """
        direction = self.axis_directions.get(axis, "higher")
        
        if direction == "lower":  # Lower is better (e.g., energy)
            # Improvement = decrease in value
            return past - current
        else:  # Higher is better (e.g., HRM alignment)
            # Improvement = increase in value
            return current - past
    
    def get_scores(self, horizon: Optional[int] = None) -> Dict[str, float]:
        """
        Get current retention scores.
        
        Args:
            horizon: Specific horizon to report (default: longest horizon)
            
        Returns:
            Dict of {axis: retention_score} for specified horizon
        """
        target_horizon = horizon if horizon is not None else max(self.horizons)
        
        return {
            axis: self.retention_scores[axis][target_horizon]
            for axis in self.critical_axes
            if self.sample_counts[axis][target_horizon] >= self.min_samples
        }
    
    def get_all_scores(self) -> Dict[str, Dict[int, float]]:
        """Get retention scores for all axes and all horizons"""
        return {
            axis: {
                h: score for h, score in horizons.items()
                if self.sample_counts[axis][h] >= self.min_samples
            }
            for axis, horizons in self.retention_scores.items()
        }
    
    def is_positive_retention(self, axis: str, horizon: Optional[int] = None) -> bool:
        """
        Check if retention is positive for given axis.
        
        Args:
            axis: Axis name
            horizon: Horizon to check (default: longest)
            
        Returns:
            True if retention score > 0 with sufficient samples
        """
        target_horizon = horizon if horizon is not None else max(self.horizons)
        
        if self.sample_counts[axis][target_horizon] < self.min_samples:
            return False
        
        return self.retention_scores[axis][target_horizon] > 0
    
    def get_metrics(self, axis: str, horizon: int) -> Optional[RetentionMetrics]:
        """Get detailed metrics for specific axis and horizon"""
        if axis not in self.critical_axes or horizon not in self.horizons:
            return None
        
        if self.sample_counts[axis][horizon] < self.min_samples:
            return None
        
        return RetentionMetrics(
            axis=axis,
            horizon=horizon,
            retention_score=self.retention_scores[axis][horizon],
            sample_count=self.sample_counts[axis][horizon],
            recent_deltas=list(self.recent_deltas[axis][horizon])
        )
    
    def generate_report(self) -> str:
        """
        Generate human-readable retention report.
        
        Returns:
            Formatted string with retention analysis
        """
        lines = [
            "=" * 70,
            "RETENTION TRACKER REPORT",
            "=" * 70,
            f"Critical Axes: {', '.join(self.critical_axes)}",
            f"Horizons Tracked: {self.horizons}",
            f"EMA Smoothing (alpha): {self.alpha}",
            f"Min Samples for Reporting: {self.min_samples}",
            "",
            "┌──────────────────────────────────────────────────────────────────┐",
            "│ RETENTION SCORES (Positive = Durable Improvement)                │",
            "├──────────────────┬──────────┬──────────┬──────────┬──────────────┤",
            "│ Axis             │ Horizon  │ Score    │ Samples  │ Status       │",
            "├──────────────────┼──────────┼──────────┼──────────┼──────────────┤",
        ]
        
        for axis in self.critical_axes:
            for horizon in self.horizons:
                score = self.retention_scores[axis][horizon]
                samples = self.sample_counts[axis][horizon]
                
                if samples < self.min_samples:
                    status = "INSUFFICIENT_DATA"
                    score_str = "N/A"
                elif score > 0.01:
                    status = "✅ IMPROVING"
                    score_str = f"{score:+.4f}"
                elif score < -0.01:
                    status = "⚠️  DEGRADING"
                    score_str = f"{score:+.4f}"
                else:
                    status = "→ STABLE"
                    score_str = f"{score:+.4f}"
                
                samples_str = f"{samples}/{self.min_samples}" if samples < self.min_samples else str(samples)
                
                lines.append(
                    f"│ {axis:<16} │ {horizon:8} │ {score_str:>8} │ {samples_str:>8} │ {status:<12} │"
                )
            lines.append("├──────────────────┼──────────┼──────────┼──────────┼──────────────┤")
        
        # Summary statistics
        lines.append("└──────────────────┴──────────┴──────────┴──────────┴──────────────┘")
        lines.append("")
        lines.append("SUMMARY:")
        
        all_positive = True
        for axis in self.critical_axes:
            if not self.is_positive_retention(axis):
                all_positive = False
                lines.append(f"  ⚠️  {axis}: Negative retention (system degrading)")
        
        if all_positive:
            lines.append("  ✅ All critical axes show positive retention")
        else:
            lines.append("  ⚠️  Some axes show negative retention - investigate")
        
        lines.append("")
        lines.append("Retention scores represent exponential moving average of")
        lines.append("improvement deltas over specified horizons.")
        lines.append("Positive score = durable improvement persists over time.")
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, any]:
        """Serialize tracker state for persistence"""
        return {
            "critical_axes": self.critical_axes,
            "horizons": self.horizons,
            "alpha": self.alpha,
            "min_samples": self.min_samples,
            "axis_directions": self.axis_directions,
            "retention_scores": self.retention_scores,
            "sample_counts": self.sample_counts,
            # Note: history and recent_deltas not serialized (reconstructed from episodes)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, any]) -> "RetentionTracker":
        """Reconstruct tracker from serialized state"""
        tracker = cls(
            critical_axes=data["critical_axes"],
            horizons=data["horizons"],
            alpha=data["alpha"],
            min_samples=data["min_samples"]
        )
        
        # Restore state
        tracker.axis_directions = data.get("axis_directions", tracker.axis_directions)
        tracker.retention_scores = data["retention_scores"]
        tracker.sample_counts = data["sample_counts"]
        
        return tracker
    
    def __str__(self) -> str:
        scores = self.get_scores()
        summary = ", ".join(f"{axis}:{score:+.3f}" for axis, score in scores.items())
        return f"RetentionTracker({summary})"
    
    def __repr__(self) -> str:
        return self.__str__()