"""
PerturbationInjector: Controlled stress testing for governed self-improvement systems.

Injects calibrated perturbations to validate:
- Governance regime switching responsiveness
- Recovery velocity from instability
- Retention of safety invariants under stress
- Collapse detector sensitivity

All perturbations are reversible via restore_original_state().
Designed for single-episode injection with explicit restoration.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PerturbationConfig:
    """Immutable configuration for perturbation severity levels"""
    misleading_evidence_prob: float  # Probability of injecting misleading evidence snippets
    complexity_factor: float         # Query complexity multiplier
    governance_disable_timeout: int  # Episodes to disable governance (0 = never disable)
    description: str
    
    def validate(self) -> None:
        """Validate configuration constraints"""
        if not (0.0 <= self.misleading_evidence_prob <= 1.0):
            raise ValueError(f"Evidence probability must be in [0,1], got {self.misleading_evidence_prob}")
        if self.complexity_factor < 1.0:
            raise ValueError(f"Complexity factor must be >= 1.0, got {self.complexity_factor}")
        if self.governance_disable_timeout < 0:
            raise ValueError(f"Timeout must be non-negative, got {self.governance_disable_timeout}")


class PerturbationInjector:
    """
    Controlled perturbation injection system for experimental stress testing.
    
    Design principles:
    - Explicit severity levels with documented effects
    - Full reversibility via system.restore_original_state()
    - No internal state persistence (stateless between injections)
    - Comprehensive logging for forensic analysis
    - Type-safe configuration with validation
    
    Usage:
        injector = PerturbationInjector(system)
        
        # Inject at specified episode
        injector.inject(severity="moderate")
        
        # System processes perturbed query
        
        # Restore after episode completes
        injector.restore()
    
    Critical: restore() MUST be called after perturbation episode to prevent
    contamination of subsequent episodes. Experiment harness should handle this.
    """
    
    # Predefined severity configurations
    SEVERITY_CONFIGS: Dict[str, PerturbationConfig] = {
        "light": PerturbationConfig(
            misleading_evidence_prob=0.2,
            complexity_factor=1.2,
            governance_disable_timeout=0,
            description="Mild stress test: minor evidence corruption, slight complexity increase"
        ),
        "moderate": PerturbationConfig(
            misleading_evidence_prob=0.4,
            complexity_factor=1.5,
            governance_disable_timeout=0,
            description="Standard stress test: significant evidence corruption, moderate complexity increase"
        ),
        "severe": PerturbationConfig(
            misleading_evidence_prob=0.6,
            complexity_factor=2.0,
            governance_disable_timeout=3,
            description="Extreme stress test: heavy evidence corruption, high complexity, temporary governance suspension"
        )
    }
    
    def __init__(self, system: Any):
        """
        Initialize perturbation injector.
        
        Args:
            system: System implementing SystemInterface perturbation methods
                Required methods:
                - inject_misleading_evidence(probability: float)
                - increase_query_complexity(factor: float)
                - temporarily_disable_governance(timeout: int) [optional]
                - restore_original_state()
        """
        self.system = system
        self._active_severity: Optional[str] = None
        self._injection_count: int = 0
        
        logger.info("PerturbationInjector initialized")
    
    def inject(self, severity: str = "moderate") -> Dict[str, Any]:
        """
        Inject controlled perturbation at specified severity level.
        
        Args:
            severity: One of "light", "moderate", "severe"
            
        Returns:
            Dict with injection details for experiment logging:
                {
                    "severity": str,
                    "evidence_prob": float,
                    "complexity_factor": float,
                    "governance_disabled": bool,
                    "timestamp": float
                }
        
        Raises:
            ValueError: If severity level is invalid
            RuntimeError: If system lacks required perturbation methods
            Exception: Any unexpected injection failure (logged and re-raised)
        """
        # Validate severity
        if severity not in self.SEVERITY_CONFIGS:
            valid = ", ".join(self.SEVERITY_CONFIGS.keys())
            raise ValueError(f"Invalid severity '{severity}'. Must be one of: {valid}")
        
        config = self.SEVERITY_CONFIGS[severity]
        config.validate()
        
        try:
            logger.warning(f".Injecting {severity.upper()} perturbation | {config.description}")
            
            # Inject misleading evidence
            if config.misleading_evidence_prob > 0:
                self.system.inject_misleading_evidence(
                    probability=config.misleading_evidence_prob
                )
                logger.info(
                    f"✓ Misleading evidence injected (p={config.misleading_evidence_prob:.2f})"
                )
            
            # Increase query complexity
            if config.complexity_factor > 1.0:
                self.system.increase_query_complexity(
                    factor=config.complexity_factor
                )
                logger.info(
                    f"✓ Query complexity increased (factor={config.complexity_factor:.1f})"
                )
            
            # Temporarily disable governance (if configured)
            governance_disabled = False
            if config.governance_disable_timeout > 0:
                try:
                    self.system.temporarily_disable_governance(
                        timeout=config.governance_disable_timeout
                    )
                    governance_disabled = True
                    logger.warning(
                        f"⚠️  Governance temporarily disabled for {config.governance_disable_timeout} episodes"
                    )
                except AttributeError:
                    logger.warning(
                        "System does not support governance disabling - skipping this perturbation component"
                    )
            
            # Track injection
            self._active_severity = severity
            self._injection_count += 1
            
            # Return injection metadata for experiment logging
            return {
                "severity": severity,
                "evidence_prob": config.misleading_evidence_prob,
                "complexity_factor": config.complexity_factor,
                "governance_disabled": governance_disabled,
                "governance_timeout": config.governance_disable_timeout,
                "injection_count": self._injection_count,
                "description": config.description
            }
            
        except AttributeError as e:
            missing_method = str(e).split("'")[-2] if "'" in str(e) else "unknown"
            error_msg = (
                f"System missing required perturbation method: {missing_method}. "
                f"Ensure system implements SystemInterface perturbation methods."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
            
        except Exception as e:
            logger.exception(f"Perturbation injection failed: {e}")
            raise
    
    def restore(self) -> bool:
        """
        Restore system to pre-perturbation state.
        
        Critical: Must be called after perturbation episode completes to prevent
        contamination of subsequent episodes.
        
        Returns:
            True if restoration successful, False if no active perturbation
        
        Raises:
            RuntimeError: If restoration fails catastrophically
        """
        if self._active_severity is None:
            logger.debug("No active perturbation to restore")
            return False
        
        try:
            logger.info(
                f"Restoring system state after {self._active_severity.upper()} perturbation "
                f"(injection #{self._injection_count})"
            )
            
            self.system.restore_original_state()
            
            # Clear state
            prev_severity = self._active_severity
            self._active_severity = None
            
            logger.info(
                f"✓ System restored to pre-perturbation state after {prev_severity} injection"
            )
            return True
            
        except Exception as e:
            logger.exception(f"Restoration failed: {e}")
            raise RuntimeError(f"Failed to restore system state: {e}") from e
    
    def is_active(self) -> bool:
        """Check if perturbation is currently active"""
        return self._active_severity is not None
    
    def get_active_severity(self) -> Optional[str]:
        """Get current active perturbation severity, or None if inactive"""
        return self._active_severity
    
    def get_injection_count(self) -> int:
        """Get total number of perturbations injected"""
        return self._injection_count
    
    def __enter__(self):
        """
        Context manager support for automatic restoration.
        
        Usage:
            with PerturbationInjector(system) as injector:
                injector.inject("moderate")
                # Perturbation active within context
            # Automatic restoration on exit
        """
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure restoration on context exit"""
        if self.is_active():
            try:
                self.restore()
            except Exception as e:
                logger.error(f"Context manager restoration failed: {e}")
                # Don't suppress original exception if one occurred
                if exc_type is None:
                    raise
        return False  # Propagate exceptions
    
    def __str__(self) -> str:
        status = f"active:{self._active_severity}" if self.is_active() else "inactive"
        return f"PerturbationInjector({status}, injections={self._injection_count})"
    
    def __repr__(self) -> str:
        return self.__str__()


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_perturbation_config(
    misleading_evidence_prob: float,
    complexity_factor: float,
    governance_disable_timeout: int = 0,
    description: str = "custom"
) -> PerturbationConfig:
    """
    Create custom perturbation configuration.
    
    Args:
        misleading_evidence_prob: Probability of injecting misleading evidence [0.0, 1.0]
        complexity_factor: Query complexity multiplier (>=1.0)
        governance_disable_timeout: Episodes to disable governance (0 = never)
        description: Human-readable description
    
    Returns:
        Validated PerturbationConfig instance
    """
    config = PerturbationConfig(
        misleading_evidence_prob=misleading_evidence_prob,
        complexity_factor=complexity_factor,
        governance_disable_timeout=governance_disable_timeout,
        description=description
    )
    config.validate()
    return config


def register_custom_severity(
    injector: PerturbationInjector,
    name: str,
    config: PerturbationConfig
) -> None:
    """
    Register custom severity level to injector's SEVERITY_CONFIGS.
    
    Args:
        injector: PerturbationInjector instance
        name: Severity name (e.g., "extreme", "debug")
        config: Validated PerturbationConfig
    
    Raises:
        ValueError: If name conflicts with existing severity
    """
    if name in injector.SEVERITY_CONFIGS:
        raise ValueError(f"Severity '{name}' already exists. Use unique name.")
    
    config.validate()
    injector.SEVERITY_CONFIGS[name] = config
    logger.info(f"Registered custom severity: {name} | {config.description}")


# ============================================================================
# EXPORT
# ============================================================================

__all__ = [
    "PerturbationInjector",
    "PerturbationConfig",
    "create_perturbation_config",
    "register_custom_severity"
]