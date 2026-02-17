"""
RegimeController: Dynamically adapts system behavior based on stability metrics.

Implements energy-based regime control:
- SAFE: Normal operation velocity
- WARNING: Conservative updates, increased scrutiny
- CRITICAL: Safety interventions, potential rollback

Translates governance metrics into concrete policy actions.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

from stephanie.components.elm.core.thresholds import CalibratedThresholds

logger = logging.getLogger(__name__)


class PolicyRegime(str, Enum):
    """Policy regime states with semantic meaning"""
    SAFE = "safe"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True)
class RegimeAction:
    """Structured policy action recommendation"""
    action_type: str  # "freeze", "reduce", "increase", "rollback"
    target_component: str  # "embedding_updates", "reflection_depth", "governance"
    magnitude: float  # 0.0 to 1.0 scaling factor
    description: str
    urgency: str  # "low", "medium", "high", "immediate"


@dataclass
class RegimeState:
    """Current regime state with transition history"""
    current_regime: PolicyRegime
    previous_regime: Optional[PolicyRegime] = None
    episode: int = 0
    metrics_snapshot: Dict[str, float] = field(default_factory=dict)
    actions_taken: List[RegimeAction] = field(default_factory=list)
    regime_duration: int = 1  # Episodes in current regime
    
    def to_dict(self) -> Dict[str, any]:
        return {
            "current_regime": self.current_regime.value,
            "previous_regime": self.previous_regime.value if self.previous_regime else None,
            "episode": self.episode,
            "metrics_snapshot": self.metrics_snapshot,
            "actions_taken": [a.__dict__ for a in self.actions_taken],
            "regime_duration": self.regime_duration
        }


class RegimeController:
    """
    Adaptive policy controller that responds to system stability metrics.
    
    Design principles:
    - Energy-driven regime transitions (fastest instability signal)
    - Hysteresis to prevent oscillation
    - Action recommendations with concrete parameters
    - Transition logging for forensic analysis
    - Configurable regime boundaries
    
    Usage:
        controller = RegimeController(thresholds)
        
        # Determine regime from current metrics
        regime_state = controller.update(metrics, episode=42)
        
        # Get recommended actions
        actions = regime_state.actions_taken
        
        # Apply actions to system
        for action in actions:
            if action.action_type == "freeze" and action.target_component == "embedding_updates":
                system.freeze_embedding_updates()
    """
    
    # Hysteresis parameters (prevent rapid regime oscillation)
    HYSTERESIS = {
        PolicyRegime.SAFE: 0.02,      # Must exceed warning threshold by margin to leave SAFE
        PolicyRegime.WARNING: 0.03,   # Must exceed critical threshold by margin to leave WARNING
        PolicyRegime.CRITICAL: 0.05   # Must drop well below critical to leave CRITICAL
    }
    
    def __init__(
        self,
        thresholds: CalibratedThresholds,
        enable_hysteresis: bool = True,
        action_callbacks: Optional[List[Callable[[RegimeAction], None]]] = None
    ):
        """
        Initialize regime controller.
        
        Args:
            thresholds: Calibrated safety thresholds
            enable_hysteresis: Prevent rapid regime oscillation
            action_callbacks: Functions to call when actions are recommended
        """
        self.thresholds = thresholds
        self.enable_hysteresis = enable_hysteresis
        self.action_callbacks = action_callbacks or []
        
        # State tracking
        self._current_regime: PolicyRegime = PolicyRegime.SAFE
        self._regime_start_episode: int = 0
        self._previous_metrics: Dict[str, float] = {}
        
        logger.info(
            f"RegimeController initialized | "
            f"hysteresis={'enabled' if enable_hysteresis else 'disabled'}"
        )
    
    def update(
        self,
        metrics: Dict[str, float],
        episode: int
    ) -> RegimeState:
        """
        Update regime state based on current metrics.
        
        Args:
            metrics: Current governance metrics
            episode: Current episode number
            
        Returns:
            RegimeState with current regime and recommended actions
        """
        # Determine target regime
        target_regime = self._determine_target_regime(metrics)
        
        # Apply hysteresis if enabled
        if self.enable_hysteresis:
            target_regime = self._apply_hysteresis(target_regime, metrics)
        
        # Detect regime transition
        is_transition = target_regime != self._current_regime
        
        # Update state
        if is_transition:
            logger.warning(
                f"REGIME TRANSITION: {self._current_regime.value.upper()} → "
                f"{target_regime.value.upper()} at episode {episode}"
            )
            self._previous_regime = self._current_regime
            self._current_regime = target_regime
            self._regime_start_episode = episode
            self._previous_metrics = metrics.copy()
        else:
            self._previous_metrics = metrics.copy()
        
        # Generate actions
        actions = self._generate_actions(target_regime, metrics, episode, is_transition)
        
        # Execute callbacks
        for action in actions:
            for callback in self.action_callbacks:
                try:
                    callback(action)
                except Exception as e:
                    logger.error(f"Action callback failed: {e}")
        
        return RegimeState(
            current_regime=target_regime,
            previous_regime=self._previous_regime if is_transition else self._current_regime,
            episode=episode,
            metrics_snapshot=metrics.copy(),
            actions_taken=actions,
            regime_duration=episode - self._regime_start_episode + 1
        )
    
    def _determine_target_regime(self, metrics: Dict[str, float]) -> PolicyRegime:
        """Determine target regime based on metrics and thresholds"""
        energy = metrics.get("energy_raw", 0.0)
        hrm = metrics.get("hrm_alignment", 1.0)
        margin = metrics.get("embedding_margin", 1.0)
        variance = metrics.get("embedding_variance", 1.0)
        collapse_index = metrics.get("collapse_index", 1.0)
        drift = metrics.get("angular_drift", 0.0)
        
        # CRITICAL checks (any trigger critical regime)
        if (energy > self.thresholds.energy_max or
            hrm < self.thresholds.hrm_min or
            margin < self.thresholds.margin_min or
            variance < self.thresholds.variance_min or
            collapse_index > self.thresholds.collapse_index_max or
            drift > self.thresholds.drift_max):
            return PolicyRegime.CRITICAL
        
        # WARNING checks
        if energy > self.thresholds.energy_warning:
            return PolicyRegime.WARNING
        
        return PolicyRegime.SAFE
    
    def _apply_hysteresis(
        self,
        target_regime: PolicyRegime,
        metrics: Dict[str, float]
    ) -> PolicyRegime:
        """Apply hysteresis to prevent rapid regime oscillation"""
        if self._current_regime == target_regime:
            return target_regime
        
        energy = metrics.get("energy_raw", 0.0)
        prev_energy = self._previous_metrics.get("energy_raw", energy)
        
        # Hysteresis when leaving SAFE regime
        if self._current_regime == PolicyRegime.SAFE and target_regime == PolicyRegime.WARNING:
            if energy < (self.thresholds.energy_warning + self.HYSTERESIS[PolicyRegime.SAFE]):
                return PolicyRegime.SAFE
        
        # Hysteresis when leaving WARNING regime
        if self._current_regime == PolicyRegime.WARNING and target_regime == PolicyRegime.CRITICAL:
            if energy < (self.thresholds.energy_max + self.HYSTERESIS[PolicyRegime.WARNING]):
                return PolicyRegime.WARNING
        
        # Hysteresis when leaving CRITICAL regime (requires significant improvement)
        if self._current_regime == PolicyRegime.CRITICAL and target_regime != PolicyRegime.CRITICAL:
            if energy > (self.thresholds.energy_max - self.HYSTERESIS[PolicyRegime.CRITICAL]):
                return PolicyRegime.CRITICAL
        
        return target_regime
    
    def _generate_actions(
        self,
        regime: PolicyRegime,
        metrics: Dict[str, float],
        episode: int,
        is_transition: bool
    ) -> List[RegimeAction]:
        """Generate concrete policy actions for current regime"""
        actions = []
        energy = metrics.get("energy_raw", 0.0)
        
        if regime == PolicyRegime.SAFE:
            if is_transition:
                actions.append(RegimeAction(
                    action_type="restore",
                    target_component="embedding_updates",
                    magnitude=1.0,
                    description="Restoring normal embedding update velocity",
                    urgency="low"
                ))
                actions.append(RegimeAction(
                    action_type="restore",
                    target_component="reflection_depth",
                    magnitude=1.0,
                    description="Restoring standard reflection depth",
                    urgency="low"
                ))
        
        elif regime == PolicyRegime.WARNING:
            actions.append(RegimeAction(
                action_type="reduce",
                target_component="embedding_update_magnitude",
                magnitude=0.5,
                description=f"Reducing embedding updates by 50% (energy={energy:.3f})",
                urgency="medium"
            ))
            actions.append(RegimeAction(
                action_type="increase",
                target_component="reflection_depth",
                magnitude=1.5,
                description="Increasing reflection depth for thorough correction",
                urgency="medium"
            ))
            actions.append(RegimeAction(
                action_type="increase",
                target_component="hard_negative_sampling",
                magnitude=2.0,
                description="Doubling hard negative sampling for stability",
                urgency="medium"
            ))
        
        elif regime == PolicyRegime.CRITICAL:
            actions.append(RegimeAction(
                action_type="freeze",
                target_component="embedding_updates",
                magnitude=0.0,
                description=f"FREEZING embedding updates (energy={energy:.3f} > {self.thresholds.energy_max:.3f})",
                urgency="immediate"
            ))
            actions.append(RegimeAction(
                action_type="increase",
                target_component="hrm_weighting",
                magnitude=2.0,
                description="Doubling HRM alignment weighting for grounding",
                urgency="high"
            ))
            actions.append(RegimeAction(
                action_type="enforce",
                target_component="grounding_constraints",
                magnitude=1.0,
                description="Enforcing strict evidence grounding constraints",
                urgency="high"
            ))
            
            # Consider rollback if this is a transition into CRITICAL
            if is_transition and episode > 0:
                actions.append(RegimeAction(
                    action_type="rollback",
                    target_component="recent_updates",
                    magnitude=1.0,
                    description="Recommending rollback of recent updates",
                    urgency="high"
                ))
        
        return actions
    
    def get_current_regime(self) -> PolicyRegime:
        """Get current regime state"""
        return self._current_regime
    
    def force_regime(self, regime: PolicyRegime, episode: int) -> RegimeState:
        """
        Force regime transition (for testing or emergency intervention).
        
        Use with extreme caution - bypasses threshold checks.
        """
        logger.critical(f"FORCING REGIME TRANSITION TO {regime.value.upper()}")
        self._previous_regime = self._current_regime
        self._current_regime = regime
        self._regime_start_episode = episode
        
        # Generate actions for forced regime
        dummy_metrics = {"energy_raw": 0.0}
        actions = self._generate_actions(regime, dummy_metrics, episode, is_transition=True)
        
        return RegimeState(
            current_regime=regime,
            previous_regime=self._previous_regime,
            episode=episode,
            metrics_snapshot=dummy_metrics,
            actions_taken=actions,
            regime_duration=1
        )
    
    def __str__(self) -> str:
        return f"RegimeController(current={self._current_regime.value})"
    
    def __repr__(self) -> str:
        return self.__str__()


# ============================================================================
# EXPORT
# ============================================================================

__all__ = [
    "RegimeController",
    "RegimeState",
    "RegimeAction",
    "PolicyRegime"
]