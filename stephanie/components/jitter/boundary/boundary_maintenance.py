# stephanie/components/jitter/boundary/boundary_maintenance.py
"""
boundary_maintenance.py
=======================
Implementation of boundary maintenance and production.

This module implements the organizationally closed production of boundary components:
- Membrane production and repair
- Boundary integrity maintenance
- Crisis response for boundary failures
- Configuration validation with Pydantic
- Circuit breaker pattern for resilience
- Comprehensive telemetry and monitoring
- SSP integration hooks
- Performance optimizations
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Dict, List

import numpy as np
from pydantic import BaseModel, Field, validator

from .membrane import BoundaryState, Membrane, MembraneSnapshot

log = logging.getLogger("stephanie.jitter.boundary.maintenance")

class MaintenanceProtocol(str, Enum):
    """Types of boundary maintenance protocols"""
    ROUTINE = "routine"
    STRESS_RESPONSE = "stress_response"
    CRISIS_RESPONSE = "crisis_response"
    REPAIR = "repair"
    IDENTITY_MAINTENANCE = "identity_maintenance"

class BoundaryMaintenanceConfig(BaseModel):
    """Validated configuration for BoundaryMaintenance"""
    routine_check_interval: int = Field(10, ge=5, le=50, description="Interval for routine checks")
    stress_response_threshold: float = Field(0.3, ge=0.1, le=0.5, description="Threshold for stress response")
    crisis_response_threshold: float = Field(0.7, ge=0.5, le=0.9, description="Threshold for crisis response")
    max_history: int = Field(1000, ge=100, le=10000, description="Maximum history length")
    production_rate: float = Field(0.1, ge=0.05, le=0.2, description="Base production rate for boundary components")
    repair_priority: float = Field(0.7, ge=0.5, le=0.9, description="Priority for repair operations")
    
    @validator('stress_response_threshold')
    def validate_thresholds(cls, v, values):
        if 'crisis_response_threshold' in values and v >= values['crisis_response_threshold']:
            raise ValueError('stress_response_threshold must be less than crisis_response_threshold')
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
    def maintain_boundary():
        # Maintenance logic here
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
                    self.logger.warning("Circuit breaker is OPEN - skipping call")
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
            "half_open_successes": self.half_open_successes
        }

@dataclass
class MaintenanceRecord:
    """Record of a maintenance operation for learning and adaptation"""
    id: str = field(default_factory=lambda: f"maint_{int(time.time())}_{np.random.randint(1000)}")
    timestamp: float = field(default_factory=time.time)
    protocol: str = MaintenanceProtocol.ROUTINE.value
    actions: Dict[str, Any] = field(default_factory=dict)
    outcome: Dict[str, Any] = field(default_factory=dict)
    success: bool = False
    energy_used: float = 0.0

@dataclass
class BoundaryMaintenanceMetrics:
    """Metrics for boundary maintenance performance"""
    maintenance_efficiency: float = 0.5
    crisis_response_rate: float = 0.0
    repair_success_rate: float = 0.5
    identity_preservation: float = 0.5
    processing_time_ms: float = 0.0

class BoundaryMaintenance:
    """
    Manages boundary maintenance and production for the autopoietic system.
    
    Key Features:
    - Organizationally closed production of boundary components
    - Routine maintenance and stress response protocols
    - Crisis response for boundary failures
    - Identity preservation mechanisms
    - Configuration validation with Pydantic
    - Circuit breaker pattern for resilience
    - Comprehensive telemetry and monitoring
    - SSP integration hooks
    - Performance optimizations
    """
    
    def __init__(
        self,
        cfg: Dict[str, Any],
        membrane: Membrane
    ):
        try:
            # Validate configuration
            self.config = BoundaryMaintenanceConfig(**cfg)
            log.info("BoundaryMaintenance configuration validated successfully")
        except Exception as e:
            log.error(f"Configuration validation failed: {str(e)}")
            # Use safe defaults
            self.config = BoundaryMaintenanceConfig()
        
        self.membrane = membrane
        self.tick_count = 0
        
        # Initialize history
        self.maintenance_history: List[MaintenanceRecord] = []
        
        # Initialize circuit breaker
        self.circuit_breaker = CircuitBreaker()
        
        # Initialize metrics
        self.metrics = BoundaryMaintenanceMetrics()
        
        # Initialize performance tracking
        self.processing_times = []
        self.max_processing_history = 100
        
        log.info("BoundaryMaintenance initialized with organizationally closed production features")
    
    @CircuitBreaker()
    def maintain_boundary(
        self,
        energy_available: float,
        system_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform boundary maintenance based on current conditions.
        
        Args:
            energy_available: Amount of energy available for maintenance
            system_state: Current system state for context
            
        Returns:
            Dictionary of maintenance actions and outcomes
        """
        start_time = time.time()
        
        try:
            self.tick_count += 1
            
            # Determine maintenance protocol based on boundary state
            boundary_state = self.membrane.get_state()
            
            # Always perform routine maintenance
            routine_result = self._perform_routine_maintenance(energy_available * 0.3)
            
            # Perform stress response if needed
            stress_result = {"actions": {}, "energy_used": 0.0}
            if boundary_state in [BoundaryState.STRESSED, BoundaryState.DAMAGED, BoundaryState.CRITICAL]:
                stress_result = self._perform_stress_response(
                    energy_available * 0.5,
                    system_state
                )
            
            # Perform crisis response if needed
            crisis_result = {"actions": {}, "energy_used": 0.0}
            if boundary_state in [BoundaryState.CRITICAL, BoundaryState.FAILED]:
                crisis_result = self._perform_crisis_response(
                    energy_available * 0.7,
                    system_state
                )
            
            # Perform identity maintenance
            identity_result = self._perform_identity_maintenance(
                energy_available * 0.2,
                system_state
            )
            
            # Record maintenance
            self._record_maintenance(
                routine_result,
                stress_result,
                crisis_result,
                identity_result,
                boundary_state
            )
            
            # Update metrics
            self._update_metrics()
            
            # Log processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > self.max_processing_history:
                self.processing_times.pop(0)
                
            log.debug(f"Performed boundary maintenance (state={boundary_state}, energy_used={energy_available:.3f})")
            return {
                "routine": routine_result,
                "stress": stress_result,
                "crisis": crisis_result,
                "identity": identity_result,
                "total_energy_used": (
                    routine_result["energy_used"] +
                    stress_result["energy_used"] +
                    crisis_result["energy_used"] +
                    identity_result["energy_used"]
                ),
                "boundary_state": boundary_state.value
            }
            
        except Exception as e:
            log.error(f"Error maintaining boundary: {str(e)}", exc_info=True)
            return {
                "routine": {"actions": {}, "energy_used": 0.0},
                "stress": {"actions": {}, "energy_used": 0.0},
                "crisis": {"actions": {}, "energy_used": 0.0},
                "identity": {"actions": {}, "energy_used": 0.0},
                "total_energy_used": 0.0,
                "boundary_state": BoundaryState.HEALTHY.value
            }
    
    def _perform_routine_maintenance(self, energy_available: float) -> Dict[str, Any]:
        """Perform routine boundary maintenance operations"""
        # Repair membrane
        repair_amount = self.membrane.repair(energy_available * 0.7)
        
        # Adjust thickness and permeability for stability
        if self.tick_count % self.config.routine_check_interval == 0:
            self.membrane.adjust_thickness(0.01)  # Slight increase for stability
            self.membrane.adjust_permeability(-0.01)  # Slight decrease for identity preservation
        
        return {
            "actions": {
                "repair": repair_amount,
                "thickness_adjustment": 0.01,
                "permeability_adjustment": -0.01
            },
            "energy_used": energy_available * 0.7
        }
    
    def _perform_stress_response(
        self,
        energy_available: float,
        system_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform stress response operations"""
        # Increase membrane thickness for protection
        self.membrane.adjust_thickness(0.05)
        
        # Decrease permeability to reduce stress impact
        self.membrane.adjust_permeability(-0.05)
        
        # Repair membrane with higher priority
        repair_amount = self.membrane.repair(energy_available * self.config.repair_priority)
        
        return {
            "actions": {
                "thickness_adjustment": 0.05,
                "permeability_adjustment": -0.05,
                "repair": repair_amount
            },
            "energy_used": energy_available
        }
    
    def _perform_crisis_response(
        self,
        energy_available: float,
        system_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform crisis response operations for boundary failure"""
        # Aggressive membrane thickening
        self.membrane.adjust_thickness(0.1)
        
        # Significantly reduce permeability
        self.membrane.adjust_permeability(-0.1)
        
        # Emergency repair with all available energy
        repair_amount = self.membrane.repair(energy_available)
        
        return {
            "actions": {
                "thickness_adjustment": 0.1,
                "permeability_adjustment": -0.1,
                "repair": repair_amount
            },
            "energy_used": energy_available
        }
    
    def _perform_identity_maintenance(
        self,
        energy_available: float,
        system_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform operations to maintain system identity"""
        # Adjust permeability based on identity preservation needs
        identity_preservation = self.membrane.get_metrics()["identity_preservation"]
        
        if identity_preservation < 0.6:
            # Increase identity preservation by reducing permeability
            self.membrane.adjust_permeability(-0.05)
        
        return {
            "actions": {
                "identity_preservation": identity_preservation,
                "permeability_adjustment": -0.05 if identity_preservation < 0.6 else 0.0
            },
            "energy_used": 0.0  # Identity maintenance uses minimal energy
        }
    
    def _record_maintenance(
        self,
        routine_result: Dict[str, Any],
        stress_result: Dict[str, Any],
        crisis_result: Dict[str, Any],
        identity_result: Dict[str, Any],
        boundary_state: BoundaryState
    ):
        """Record maintenance operation for learning and adaptation"""
        # Create record
        record = MaintenanceRecord(
            protocol=boundary_state.value,
            actions={
                "routine": routine_result["actions"],
                "stress": stress_result["actions"],
                "crisis": crisis_result["actions"],
                "identity": identity_result["actions"]
            },
            outcome={
                "integrity": self.membrane.integrity,
                "thickness": self.membrane.thickness,
                "permeability": self.membrane.permeability
            },
            success=self.membrane.integrity > self.config.stress_response_threshold,
            energy_used=(
                routine_result["energy_used"] +
                stress_result["energy_used"] +
                crisis_result["energy_used"] +
                identity_result["energy_used"]
            )
        )
        
        # Add to history
        self.maintenance_history.append(record)
        
        # Keep history bounded
        if len(self.maintenance_history) > self.config.max_history:
            self.maintenance_history.pop(0)
    
    def _update_metrics(self):
        """Update boundary maintenance metrics based on history"""
        # Update maintenance efficiency
        if len(self.maintenance_history) >= 10:
            # Calculate integrity improvement per energy unit
            improvements = []
            for i in range(1, len(self.maintenance_history)):
                prev = self.maintenance_history[i-1]
                curr = self.maintenance_history[i]
                integrity_improvement = (
                    curr.outcome["integrity"] - 
                    prev.outcome["integrity"]
                )
                improvements.append(integrity_improvement / (curr.energy_used + 1e-8))
            
            self.metrics.maintenance_efficiency = max(0.0, min(1.0, np.mean(improvements)))
        
        # Update crisis response rate
        crisis_events = sum(1 for m in self.maintenance_history 
                           if m.protocol in [BoundaryState.DAMAGED.value, BoundaryState.CRITICAL.value])
        total_events = len(self.maintenance_history)
        self.metrics.crisis_response_rate = crisis_events / total_events if total_events > 0 else 0.0
        
        # Update repair success rate
        successful_repairs = sum(1 for m in self.maintenance_history if m.success)
        self.metrics.repair_success_rate = successful_repairs / total_events if total_events > 0 else 0.0
        
        # Update identity preservation
        self.metrics.identity_preservation = self.membrane.get_metrics()["identity_preservation"]
        
        # Update processing time
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0.0
        self.metrics.processing_time_ms = avg_processing_time * 1000
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current boundary maintenance metrics for monitoring and adaptation"""
        return {
            "maintenance_efficiency": self.metrics.maintenance_efficiency,
            "crisis_response_rate": self.metrics.crisis_response_rate,
            "repair_success_rate": self.metrics.repair_success_rate,
            "identity_preservation": self.metrics.identity_preservation,
            "processing_time_ms": self.metrics.processing_time_ms,
            "maintenance_history_size": len(self.maintenance_history),
            "circuit_breaker_state": self.circuit_breaker.get_state()
        }
    
    def get_boundary_snapshot(self) -> MembraneSnapshot:
        """Get current boundary snapshot for telemetry and reproduction"""
        return self.membrane.get_snapshot()
    
    def get_ssp_integration_metrics(self) -> Dict[str, float]:
        """
        Get metrics suitable for SSP integration.
        
        Returns boundary maintenance metrics in a format SSP can use for reward shaping.
        """
        metrics = self.get_metrics()
        membrane_metrics = self.membrane.get_metrics()
        
        return {
            "boundary_integrity": self.membrane.integrity,
            "boundary_stability": membrane_metrics["integrity_stability"],
            "maintenance_efficiency": metrics["maintenance_efficiency"],
            "crisis_response_rate": metrics["crisis_response_rate"],
            "repair_success_rate": metrics["repair_success_rate"],
            "identity_preservation": metrics["identity_preservation"],
            "processing_efficiency": 1.0 / (1.0 + metrics["processing_time_ms"])
        }
    
    def evaluate_maintenance_success(
        self,
        maintenance_id: str,
        expected_outcome: Dict[str, Any],
        actual_outcome: Dict[str, Any]
    ) -> bool:
        """
        Evaluate whether maintenance operation was successful.
        
        Args:
            maintenance_id: ID of the maintenance operation to evaluate
            expected_outcome: Expected outcome based on maintenance
            actual_outcome: Actual outcome that occurred
            
        Returns:
            True if maintenance was successful, False otherwise
        """
        # Find the maintenance record
        record = next((m for m in self.maintenance_history if m.id == maintenance_id), None)
        if not record:
            log.warning(f"Maintenance record not found: {maintenance_id}")
            return False
        
        # Determine success based on outcome match
        expected_integrity = expected_outcome.get("integrity", 0.5)
        actual_integrity = actual_outcome.get("integrity", 0.5)
        
        # Maintenance is successful if actual outcome matches expected outcome
        success = abs(actual_integrity - expected_integrity) < 0.1
        
        # Update record
        record.success = success
        
        # Update metrics
        if success:
            self.metrics.maintenance_efficiency = min(1.0, self.metrics.maintenance_efficiency + 0.01)
        else:
            self.metrics.maintenance_efficiency = max(0.0, self.metrics.maintenance_efficiency - 0.01)
        
        log.debug(f"Maintenance evaluation: {maintenance_id} - success={success}")
        return success
    
    def get_recent_maintenance(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get recent maintenance operations for analysis or reporting"""
        return [
            {
                "id": m.id,
                "timestamp": m.timestamp,
                "protocol": m.protocol,
                "actions": m.actions,
                "outcome": m.outcome,
                "success": m.success,
                "energy_used": m.energy_used
            }
            for m in self.maintenance_history[-n:]
        ]
    
    def get_optimal_maintenance_profile(
        self,
        boundary_state: BoundaryState
    ) -> Dict[str, float]:
        """
        Get optimal maintenance profile for given boundary state.
        
        Args:
            boundary_state: Current boundary state
            
        Returns:
            Optimal maintenance parameters for the state
        """
        if len(self.maintenance_history) < 10:
            return {
                "repair_priority": self.config.repair_priority,
                "thickness_adjustment": 0.01,
                "permeability_adjustment": -0.01
            }
        
        # Find similar boundary states
        similar = [
            m for m in self.maintenance_history
            if m.protocol == boundary_state.value
        ]
        
        # If no similar states, return default profile
        if not similar:
            return {
                "repair_priority": self.config.repair_priority,
                "thickness_adjustment": 0.01,
                "permeability_adjustment": -0.01
            }
        
        # Calculate weighted average of maintenance parameters
        repair_priority = np.mean([m.actions["routine"]["repair"] / (m.energy_used + 1e-8) for m in similar])
        thickness_adjustment = np.mean([
            m.actions["routine"]["thickness_adjustment"] +
            (m.actions["stress"]["thickness_adjustment"] if "stress" in m.actions else 0) +
            (m.actions["crisis"]["thickness_adjustment"] if "crisis" in m.actions else 0)
            for m in similar
        ])
        permeability_adjustment = np.mean([
            m.actions["routine"]["permeability_adjustment"] +
            (m.actions["stress"]["permeability_adjustment"] if "stress" in m.actions else 0) +
            (m.actions["crisis"]["permeability_adjustment"] if "crisis" in m.actions else 0)
            for m in similar
        ])
        
        return {
            "repair_priority": repair_priority,
            "thickness_adjustment": thickness_adjustment,
            "permeability_adjustment": permeability_adjustment
        } 