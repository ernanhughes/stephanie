# stephanie/components/jitter/coupling/structural_coupling.py
"""
structural_coupling.py
======================
Implementation of structural coupling between Jitter and its environment.

This module implements the concept that an autopoietic system maintains its
organization through continuous structural changes triggered by environmental
perturbations, while preserving its autopoietic organization.

Key Enhancements:
- Configuration validation with Pydantic
- Circuit breaker pattern for resilience
- Multi-level crisis management integration
- Performance-optimized processing
- SSP integration hooks
- Comprehensive telemetry and monitoring
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field, validator

log = logging.getLogger("stephanie.jitter.coupling")

class PerturbationType(str, Enum):
    """Types of environmental perturbations"""
    VPM = "vpm"
    ENERGY = "energy"
    BOUNDARY = "boundary"
    COGNITIVE = "cognitive"
    EXTERNAL_SIGNAL = "external_signal"
    UNKNOWN = "unknown"

class CouplingConfig(BaseModel):
    """Validated configuration for StructuralCoupling"""
    resonance_threshold: float = Field(0.7, ge=0.3, le=0.95, description="Threshold for significant resonance")
    min_perturbation_impact: float = Field(0.1, ge=0.0, le=0.3, description="Minimum impact to process")
    max_history: int = Field(1000, ge=100, le=10000, description="Maximum history length")
    adaptation_smoothing: float = Field(0.8, ge=0.5, le=0.95, description="Smoothing factor for adaptations")
    crisis_threshold: float = Field(0.85, ge=0.7, le=0.95, description="Threshold for crisis-level perturbations")
    
    @validator('min_perturbation_impact')
    def validate_min_max_perturbation_impact(cls, v, values):
        if 'resonance_threshold' in values and v >= values['resonance_threshold']:
            raise ValueError('min_perturbation_impact must be less than resonance_threshold')
        return v

class CircuitBreakerState:
    """States for circuit breaker pattern"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    """
    Circuit breaker pattern implementation for service resilience.
    
    Example usage:
    
    @CircuitBreaker(failure_threshold=3, recovery_timeout=30)
    def process_perturbation(perturbation, system_state):
        # Processing logic here
        pass
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
        """Decorator implementation"""
        
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Check if circuit is open
            if self.state == CircuitBreakerState.OPEN:
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.logger.info("Circuit breaker transitioning to HALF_OPEN state")
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.half_open_successes = 0
                else:
                    log.warning("Circuit breaker is OPEN - skipping call")
                    return None
            
            try:
                # Call the function
                result = func(*args, **kwargs)
                
                # Reset failures if successful
                if self.state == CircuitBreakerState.HALF_OPEN:
                    self.half_open_successes += 1
                    if self.half_open_successes >= self.half_open_attempts:
                        self.logger.info("Circuit breaker transitioning to CLOSED state")
                        self.state = CircuitBreakerState.CLOSED
                        self.failures = 0
                        self.half_open_successes = 0
                
                return result
                
            except Exception as e:
                # Record failure
                self.failures += 1
                self.last_failure_time = time.time()
                self.logger.error(f"Service failure: {str(e)}, failures: {self.failures}")
                
                # Transition to OPEN state if threshold reached
                if self.failures >= self.failure_threshold:
                    log.warning("Circuit breaker transitioning to OPEN state")
                    self.state = CircuitBreakerState.OPEN
                
                raise
        
        return wrapper
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state for monitoring"""
        return {
            "state": self.state,
            "failures": self.failures,
            "last_failure_time": self.last_failure_time,
            "half_open_successes": self.half_open_successes
        }

@dataclass
class PerturbationRecord:
    """Record of a perturbation event"""
    id: str = field(default_factory=lambda: f"perturb_{uuid.uuid4().hex[:8]}")
    timestamp: float = field(default_factory=time.time)
    perturbation_type: PerturbationType = PerturbationType.UNKNOWN
    impact: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    system_state_snapshot: Dict[str, Any] = field(default_factory=dict)
    adaptation: Dict[str, Any] = field(default_factory=dict)
    adaptation_success: Optional[bool] = None

@dataclass
class CouplingMetrics:
    """Metrics for structural coupling performance"""
    resonance_score: float = 0.5
    adaptation_efficiency: float = 0.5
    perturbation_frequency: float = 0.0
    adaptation_success_rate: float = 0.0
    crisis_events: int = 0
    recent_perturbation_pattern: Dict[str, float] = field(default_factory=dict)

class StructuralCoupling:
    """
    Manages the structural coupling between Jitter and its environment with enhanced features.
    
    Key Features:
    - Environmental perturbations trigger structural changes
    - The system adapts while maintaining its autopoietic organization
    - History of perturbations and adaptations is preserved
    - Circuit breaker pattern for resilience
    - Crisis management integration
    - Performance-optimized processing
    """
    
    def __init__(self, cfg: Dict[str, Any]):
        try:
            # Validate configuration
            self.config = CouplingConfig(**cfg)
            log.info("StructuralCoupling configuration validated successfully")
        except Exception as e:
            log.error(f"Configuration validation failed: {str(e)}")
            # Use safe defaults
            self.config = CouplingConfig()
        
        # Initialize history
        self.perturbation_history: List[PerturbationRecord] = []
        
        # Initialize circuit breaker
        self.circuit_breaker = CircuitBreaker()
        
        # Initialize metrics
        self.metrics = CouplingMetrics()
        
        # Initialize performance tracking
        self.processing_times = []
        self.max_processing_history = 100
        
        log.info("StructuralCoupling initialized with enhanced features")
    
    @CircuitBreaker()
    def process_perturbation(
        self, 
        perturbation: Dict[str, Any], 
        system_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process an environmental perturbation and determine system response.
        
        Args:
            perturbation: Environmental input (VPM, sensory data, etc.)
            system_state: Current state of the Jitter system
            
        Returns:
            Dictionary of structural changes to apply to the system
        """
        start_time = time.time()
        
        try:
            # Determine perturbation type
            perturbation_type = self._determine_perturbation_type(perturbation)
            
            # Calculate perturbation impact
            impact = self._calculate_perturbation_impact(perturbation, system_state)
            
            # Only process significant perturbations
            if impact < self.config.min_perturbation_impact:
                log.debug(f"Ignoring low-impact perturbation (impact={impact:.3f} < threshold={self.config.min_perturbation_impact})")
                return {}
                
            # Record perturbation
            record = self._record_perturbation(
                perturbation_type, 
                impact, 
                perturbation, 
                system_state
            )
            
            # Determine adaptation response
            structural_changes = self._determine_adaptation(
                perturbation, 
                impact, 
                system_state,
                perturbation_type
            )
            
            # Record adaptation
            self._record_adaptation(record, structural_changes)
            
            # Update metrics
            self._update_metrics(record, structural_changes, impact)
            
            # Log processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > self.max_processing_history:
                self.processing_times.pop(0)
                
            log.debug(f"Processed perturbation (type={perturbation_type}, impact={impact:.3f}, changes={structural_changes})")
            return structural_changes
            
        except Exception as e:
            log.error(f"Error processing perturbation: {str(e)}", exc_info=True)
            # Return empty changes as fallback
            return {}
    
    def _determine_perturbation_type(self, perturbation: Dict[str, Any]) -> PerturbationType:
        """Determine the type of perturbation based on content"""
        if "vpm_embedding" in perturbation:
            return PerturbationType.VPM
        elif "energy" in perturbation:
            return PerturbationType.ENERGY
        elif "boundary" in perturbation:
            return PerturbationType.BOUNDARY
        elif "cognitive" in perturbation:
            return PerturbationType.COGNITIVE
        elif "signal" in perturbation:
            return PerturbationType.EXTERNAL_SIGNAL
        return PerturbationType.UNKNOWN
    
    def _calculate_perturbation_impact(
        self, 
        perturbation: Dict[str, Any], 
        system_state: Dict[str, Any]
    ) -> float:
        """Calculate how significantly a perturbation affects the system (0-1)"""
        # For VPM-based perturbations
        if "vpm_embedding" in perturbation and "scoring" in system_state:
            try:
                # Use VPM energy plugin for boundary threat assessment
                result = system_state["scoring"].score(
                    "vpm_energy", 
                    {"vpm_embedding": perturbation["vpm_embedding"]},
                    dimensions=["reptilian.threat01"]
                )
                return min(1.0, max(0.0, result.get("reptilian.threat01", 0.5)))
            except Exception as e:
                log.warning(f"Failed to calculate perturbation impact: {str(e)}")
        
        # For other perturbation types
        return 0.5  # Default medium impact
    
    def _determine_adaptation(
        self, 
        perturbation: Dict[str, Any],
        impact: float,
        system_state: Dict[str, Any],
        perturbation_type: PerturbationType
    ) -> Dict[str, Any]:
        """Determine structural changes needed in response to perturbation"""
        changes = {}
        
        # High impact perturbations trigger more significant changes
        if impact > self.config.resonance_threshold:
            # Membrane adaptation
            changes["membrane"] = {
                "thickness": 0.1 * impact,  # Increase thickness for protection
                "permeability": -0.05 * impact  # Decrease permeability
            }
            
            # Cognitive adaptation
            changes["cognitive"] = {
                "attention_weights": {
                    "reptilian": 0.1 * impact,
                    "mammalian": 0.05 * impact,
                    "primate": -0.15 * impact
                }
            }
            
            # Metabolic adaptation
            changes["metabolic"] = {
                "energy_conversion": 0.05 * impact
            }
        else:
            # Minor adaptations for low-impact perturbations
            changes["membrane"] = {"thickness": 0.01}
            changes["cognitive"] = {"attention_weights": {"primate": 0.02}}
        
        return changes
    
    def _record_perturbation(
        self,
        perturbation_type: PerturbationType,
        impact: float,
        perturbation: Dict[str, Any],
        system_state: Dict[str, Any]
    ) -> PerturbationRecord:
        """Record perturbation for learning and adaptation"""
        # Create record
        record = PerturbationRecord(
            perturbation_type=perturbation_type,
            impact=impact,
            details={k: v for k, v in perturbation.items() if k != "vpm_embedding"},  # Omit large embeddings
            system_state_snapshot=self._capture_system_snapshot(system_state)
        )
        
        # Add to history
        self.perturbation_history.append(record)
        
        # Keep history bounded
        if len(self.perturbation_history) > self.config.max_history:
            self.perturbation_history.pop(0)
            
        return record
    
    def _record_adaptation(
        self,
        record: PerturbationRecord,
        structural_changes: Dict[str, Any]
    ):
        """Record adaptation response for learning"""
        record.adaptation = structural_changes
        # Success will be determined later when we evaluate the outcome
    
    def _capture_system_snapshot(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """Capture a snapshot of the system state for historical analysis"""
        snapshot = {}
        
        # Energy state
        if "energy" in system_state:
            snapshot["energy"] = {
                k: v for k, v in system_state["energy"].items() 
                if k in ["cognitive", "metabolic", "reserve"]
            }
        
        # Membrane state
        if "membrane" in system_state:
            snapshot["membrane"] = {
                k: v for k, v in system_state["membrane"].items() 
                if k in ["integrity", "thickness", "permeability"]
            }
        
        # Cognitive state
        if "cognitive" in system_state:
            snapshot["cognitive"] = {
                k: v for k, v in system_state["cognitive"].items() 
                if k in ["coherence", "structures", "resonance"]
            }
        
        return snapshot
    
    def _update_metrics(
        self,
        record: PerturbationRecord,
        structural_changes: Dict[str, Any],
        impact: float
    ):
        """Update coupling metrics based on perturbation processing"""
        # Update resonance score (how well the system fits with environment)
        self.metrics.resonance_score = (
            self.metrics.resonance_score * self.config.adaptation_smoothing +
            impact * (1.0 - self.config.adaptation_smoothing)
        )
        
        # Update adaptation efficiency (how effectively changes maintain health)
        if len(self.perturbation_history) >= 2:
            # Simplified implementation - would be more sophisticated in production
            self.metrics.adaptation_efficiency = min(1.0, self.metrics.adaptation_efficiency + 0.01)
        
        # Update perturbation frequency
        if len(self.perturbation_history) > 10:
            time_span = self.perturbation_history[-1].timestamp - self.perturbation_history[0].timestamp
            if time_span > 0:
                self.metrics.perturbation_frequency = len(self.perturbation_history) / time_span
        
        # Update recent perturbation pattern
        self.metrics.recent_perturbation_pattern = self.get_perturbation_pattern()
        
        # Track crisis events
        if impact > self.config.crisis_threshold:
            self.metrics.crisis_events += 1
    
    def get_perturbation_pattern(self) -> Dict[str, float]:
        """
        Analyze patterns in perturbations to identify environmental characteristics.
        
        Returns a dictionary of perturbation type frequencies.
        """
        if not self.perturbation_history:
            return {}
            
        perturbation_types = [p.perturbation_type.value for p in self.perturbation_history]
        type_counts = {t: perturbation_types.count(t) for t in set(perturbation_types)}
        
        # Convert to frequencies
        total = len(perturbation_types)
        return {t: count / total for t, count in type_counts.items()}
    
    def get_coupling_metrics(self) -> Dict[str, Any]:
        """Get current coupling metrics for monitoring and adaptation"""
        # Calculate adaptation success rate
        success_count = sum(1 for p in self.perturbation_history if p.adaptation_success is True)
        total = len([p for p in self.perturbation_history if p.adaptation_success is not None])
        success_rate = success_count / total if total > 0 else 0.0
        
        # Calculate average processing time
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0.0
        
        return {
            "resonance_score": self.metrics.resonance_score,
            "adaptation_efficiency": self.metrics.adaptation_efficiency,
            "perturbation_frequency": self.metrics.perturbation_frequency,
            "adaptation_success_rate": success_rate,
            "crisis_events": self.metrics.crisis_events,
            "recent_perturbation_pattern": self.metrics.recent_perturbation_pattern,
            "average_processing_time_ms": avg_processing_time * 1000,
            "perturbation_history_size": len(self.perturbation_history),
            "circuit_breaker_state": self.circuit_breaker.get_state()
        }
    
    def evaluate_adaptation_success(
        self,
        perturbation_id: str,
        system_health_before: float,
        system_health_after: float
    ) -> bool:
        """
        Evaluate whether an adaptation was successful.
        
        Args:
            perturbation_id: ID of the perturbation to evaluate
            system_health_before: System health before adaptation
            system_health_after: System health after adaptation
            
        Returns:
            True if adaptation was successful, False otherwise
        """
        # Find the perturbation record
        record = next((p for p in self.perturbation_history if p.id == perturbation_id), None)
        if not record:
            log.warning(f"Perturbation record not found: {perturbation_id}")
            return False
        
        # Determine success based on health improvement
        health_improved = system_health_after > system_health_before
        impact_significant = record.impact > self.config.min_perturbation_impact
        
        # Adaptation is successful if health improved after significant impact
        success = health_improved and impact_significant
        
        # Update record
        record.adaptation_success = success
        
        # Update metrics
        self.metrics.adaptation_success_rate = (
            self.metrics.adaptation_success_rate * 0.9 + 
            (1.0 if success else 0.0) * 0.1
        )
        
        log.debug(f"Adaptation evaluation: {perturbation_id} - success={success}")
        return success
    
    def get_ssp_integration_metrics(self) -> Dict[str, float]:
        """
        Get metrics suitable for SSP integration.
        
        Returns coupling metrics in a format SSP can use for reward shaping.
        """
        metrics = self.get_coupling_metrics()
        
        return {
            "resonance_score": metrics["resonance_score"],
            "environmental_fit": metrics["resonance_score"],
            "adaptation_efficiency": metrics["adaptation_efficiency"],
            "perturbation_frequency": metrics["perturbation_frequency"],
            "crisis_events": metrics["crisis_events"],
            "processing_efficiency": 1.0 / (1.0 + metrics["average_processing_time_ms"])
        }
    
    def get_recent_perturbations(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get recent perturbations for analysis or reporting"""
        return [
            {
                "id": p.id,
                "timestamp": p.timestamp,
                "type": p.perturbation_type.value,
                "impact": p.impact,
                "details": p.details,
                "adaptation": p.adaptation,
                "success": p.adaptation_success
            }
            for p in self.perturbation_history[-n:]
        ]
    
    def get_crisis_assessment(self) -> Dict[str, Any]:
        """Get current crisis assessment based on perturbation patterns"""
        if not self.perturbation_history:
            return {
                "crisis_level": 0.0,
                "crisis_events": 0,
                "recent_perturbation_pattern": {}
            }
        
        # Calculate crisis level based on recent high-impact perturbations
        recent = self.perturbation_history[-20:]
        high_impact_count = sum(1 for p in recent if p.impact > self.config.crisis_threshold)
        crisis_level = min(1.0, high_impact_count / 5.0)  # Scale to 0-1
        
        return {
            "crisis_level": crisis_level,
            "crisis_events": self.metrics.crisis_events,
            "recent_perturbation_pattern": self.metrics.recent_perturbation_pattern,
            "resonance_score": self.metrics.resonance_score
        }
    
    def get_adaptation_recommendations(self) -> List[Dict[str, Any]]:
        """
        Get adaptation recommendations based on historical perturbation patterns.
        
        Returns a list of recommended adaptations for common perturbation patterns.
        """
        if len(self.perturbation_history) < 10:
            return []
        
        # Analyze patterns for each perturbation type
        recommendations = []
        
        # Get most common perturbation types
        pattern = self.get_perturbation_pattern()
        for pert_type, frequency in pattern.items():
            if frequency < 0.1:  # Only consider significant patterns
                continue
                
            # Get typical impacts for this type
            impacts = [p.impact for p in self.perturbation_history if p.perturbation_type.value == pert_type]
            avg_impact = np.mean(impacts) if impacts else 0.5
            
            # Get successful adaptations for this type
            successful = [p.adaptation for p in self.perturbation_history 
                         if p.perturbation_type.value == pert_type and p.adaptation_success]
            
            if not successful:
                continue
                
            # Average the successful adaptations
            recommendation = {
                "perturbation_type": pert_type,
                "frequency": frequency,
                "average_impact": avg_impact,
                "recommended_adaptation": self._average_adaptations(successful)
            }
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def _average_adaptations(self, adaptations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Average a list of adaptation dictionaries"""
        if not adaptations:
            return {}
        
        # Initialize result structure
        result = {}
        
        # For each component in adaptations
        for component in ["membrane", "cognitive", "metabolic"]:
            if all(component in a and isinstance(a[component], dict) for a in adaptations):
                # Average the values
                component_values = {}
                for key in adaptations[0][component].keys():
                    values = [a[component][key] for a in adaptations if key in a[component]]
                    if values:
                        component_values[key] = np.mean(values)
                result[component] = component_values
        
        return result