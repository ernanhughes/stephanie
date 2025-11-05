# stephanie/components/jitter/regulation/apoptosis.py
"""
apoptosis.py
============
Implementation of programmed cell death for the Jitter Autopoietic System.

This module implements the apoptosis system that handles graceful termination
of Jitter organisms when they can no longer maintain autopoiesis.

Key Features:
- Crisis-based initiation of apoptosis
- Legacy preservation for future generations
- Configuration validation with Pydantic
- Circuit breaker pattern for resilience
- Comprehensive telemetry and monitoring
- SSP integration hooks
- Performance optimizations
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field, validator

log = logging.getLogger("stephanie.jitter.regulation.apoptosis")

class ApoptosisConfig(BaseModel):
    """Validated configuration for ApoptosisSystem"""
    boundary_threshold: float = Field(0.1, ge=0.01, le=0.3, description="Boundary integrity threshold for apoptosis")
    energy_threshold: float = Field(1.0, ge=0.1, le=5.0, description="Energy level threshold for apoptosis")
    max_crisis_ticks: int = Field(50, ge=10, le=200, description="Maximum ticks in crisis before apoptosis")
    recovery_window: float = Field(30.0, ge=5.0, le=120.0, description="Window for crisis recovery")
    legacy_preservation: bool = Field(True, description="Whether to preserve legacy on apoptosis")
    max_history: int = Field(100, ge=50, le=500, description="Maximum history length for apoptosis tracking")
    
    @validator('boundary_threshold')
    def validate_thresholds(cls, v, values):
        if 'energy_threshold' in values and v >= values['energy_threshold']:
            raise ValueError('boundary_threshold must be less than energy_threshold')
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
    def should_initiate(core):
        # Decision logic here
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
class ApoptosisRecord:
    """Record of an apoptosis event for legacy and learning"""
    timestamp: float = field(default_factory=time.time)
    cause: str = ""
    system_state: Dict[str, Any] = field(default_factory=dict)
    legacy_artifact: Optional[Dict[str, Any]] = None
    resolution_time: Optional[float] = None
    success: bool = False

@dataclass
class ApoptosisMetrics:
    """Metrics for apoptosis system performance"""
    apoptosis_rate: float = 0.0
    legacy_preservation_rate: float = 0.0
    crisis_detection_efficiency: float = 0.0
    recovery_rate: float = 0.0
    processing_time_ms: float = 0.0

class ApoptosisSystem:
    """
    Implementation of programmed cell death for the Jitter Autopoietic System.
    
    Key Features:
    - Crisis-based initiation of apoptosis
    - Legacy preservation for future generations
    - Configuration validation with Pydantic
    - Circuit breaker pattern for resilience
    - Comprehensive telemetry and monitoring
    - SSP integration hooks
    - Performance optimizations
    """
    
    def __init__(self, cfg: Dict[str, Any]):
        try:
            # Validate configuration
            self.config = ApoptosisConfig(**cfg)
            log.info("ApoptosisSystem configuration validated successfully")
        except Exception as e:
            log.error(f"Configuration validation failed: {str(e)}")
            # Use safe defaults
            self.config = ApoptosisConfig()
        
        # Initialize tracking variables
        self.crisis_counter = 0
        self.crisis_start_time = 0.0
        self.last_recovery_check = 0.0
        self.initiated = False
        
        # Initialize history
        self.apoptosis_history: List[ApoptosisRecord] = []
        
        # Initialize metrics
        self.metrics = ApoptosisMetrics()
        
        # Initialize performance tracking
        self.processing_times = []
        self.max_processing_history = 100
        
        # Initialize circuit breaker
        self.circuit_breaker = CircuitBreaker()
        
        log.info("ApoptosisSystem initialized with crisis detection and legacy preservation")
    
    @CircuitBreaker()
    def should_initiate(self, core, homeostasis=None) -> bool:
        """
        Determine if apoptosis should be initiated based on system state.
        
        Args:
            core: The AutopoieticCore instance to check
            homeostasis: Homeostasis system (optional)
            
        Returns:
            True if apoptosis should be initiated, False otherwise
        """
        start_time = time.time()
        
        try:
            # Check if already initiated
            if self.initiated:
                return True
            
            # Get current system state
            energy_balance = core.energy.level("metabolic") + core.energy.level("cognitive")
            boundary_integrity = core.membrane.integrity
            
            # Check boundary integrity threshold
            if boundary_integrity < self.config.boundary_threshold:
                self.crisis_counter += 1
                if self.crisis_counter > self.config.max_crisis_ticks:
                    log.warning("Boundary integrity below threshold - initiating apoptosis")
                    self.initiated = True
                    return True
            
            # Check energy threshold
            if energy_balance < self.config.energy_threshold:
                self.crisis_counter += 1
                if self.crisis_counter > self.config.max_crisis_ticks:
                    log.warning("Energy levels below threshold - initiating apoptosis")
                    self.initiated = True
                    return True
            
            # Check for crisis recovery window
            if self.crisis_counter > 0:
                current_time = time.time()
                if current_time - self.last_recovery_check > self.config.recovery_window:
                    # Reset crisis counter if recovery period has passed
                    self.crisis_counter = max(0, self.crisis_counter - 1)
                    self.last_recovery_check = current_time
            
            # Reset crisis counter if system is stabilizing
            if boundary_integrity > self.config.boundary_threshold * 1.5:
                self.crisis_counter = 0
            
            # Log processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > self.max_processing_history:
                self.processing_times.pop(0)
                
            return False
            
        except Exception as e:
            log.error(f"Error determining apoptosis initiation: {str(e)}", exc_info=True)
            return False
    
    def initiate(self, core, homeostasis=None, cause: str = "unknown") -> Dict[str, Any]:
        """
        Initiate apoptosis process with legacy preservation.
        
        Args:
            core: The AutopoieticCore instance to terminate
            homeostasis: Homeostasis system (optional)
            cause: Reason for apoptosis
            
        Returns:
            Dictionary with initiation results
        """
        start_time = time.time()
        
        try:
            # Mark apoptosis as initiated
            self.initiated = True
            
            # Create system state snapshot
            system_state = {
                "timestamp": time.time(),
                "boundary_integrity": core.membrane.integrity,
                "energy_levels": {
                    "cognitive": core.energy.level("cognitive"),
                    "metabolic": core.energy.level("metabolic"),
                    "reserve": core.energy.level("reserve")
                },
                "cognitive_state": {
                    "integrated": 0.0,  # Would be populated from triune
                    "threat_level": 0.0,
                    "emotional_valence": 0.0
                },
                "homeostasis_state": homeostasis.get_telemetry() if homeostasis else {},
                "generation": getattr(core, 'generation', 0),
                "id": getattr(core, 'id', 'unknown')
            }
            
            # Create legacy artifact
            legacy_artifact = self._create_legacy_artifact(core, system_state, cause)
            
            # Record apoptosis event
            self._record_apoptosis_event(cause, system_state, legacy_artifact)
            
            # Update metrics
            self._update_metrics()
            
            # Log processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > self.max_processing_history:
                self.processing_times.pop(0)
                
            log.info(f"Apoptosis initiated (cause={cause})")
            return {
                "success": True,
                "legacy_artifact": legacy_artifact,
                "cause": cause,
                "timestamp": time.time()
            }
            
        except Exception as e:
            log.error(f"Error initiating apoptosis: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "cause": cause,
                "timestamp": time.time()
            }
    
    def _create_legacy_artifact(
        self, 
        core, 
        system_state: Dict[str, Any], 
        cause: str
    ) -> Dict[str, Any]:
        """Create a legacy artifact for future generations"""
        try:
            # Create JAF-like artifact
            artifact = {
                "spec": "jas/legacy/v1",
                "timestamp": time.time(),
                "cause": cause,
                "system_state": system_state,
                "boundary_integrity": system_state["boundary_integrity"],
                "energy_levels": system_state["energy_levels"],
                "generation": system_state["generation"],
                "id": system_state["id"],
                "crisis_duration": self.crisis_counter
            }
            
            # Add cognitive state if available
            if hasattr(core, 'triune') and core.triune:
                cognitive_states = core.triune.get_recent_states(5)
                artifact["cognitive_states"] = [
                    {
                        "integrated": s.integrated,
                        "threat_level": s.threat_level,
                        "emotional_valence": s.emotional_valence
                    } for s in cognitive_states
                ]
            
            return artifact
            
        except Exception as e:
            log.error(f"Error creating legacy artifact: {str(e)}", exc_info=True)
            return {
                "spec": "jas/legacy/error",
                "timestamp": time.time(),
                "cause": cause,
                "error": str(e)
            }
    
    def _record_apoptosis_event(
        self, 
        cause: str, 
        system_state: Dict[str, Any], 
        legacy_artifact: Dict[str, Any]
    ):
        """Record apoptosis event for learning and analysis"""
        # Create record
        record = ApoptosisRecord(
            cause=cause,
            system_state=system_state,
            legacy_artifact=legacy_artifact
        )
        
        # Add to history
        self.apoptosis_history.append(record)
        
        # Keep history bounded
        if len(self.apoptosis_history) > self.config.max_history:
            self.apoptosis_history.pop(0)
    
    def _update_metrics(self):
        """Update apoptosis metrics based on history"""
        # Update apoptosis rate
        if len(self.apoptosis_history) > 0:
            self.metrics.apoptosis_rate = len(self.apoptosis_history) / max(1, len(self.apoptosis_history))
        
        # Update legacy preservation rate
        legacy_preserved = sum(1 for c in self.apoptosis_history if c.legacy_artifact is not None)
        self.metrics.legacy_preservation_rate = legacy_preserved / max(1, len(self.apoptosis_history))
        
        # Update processing time
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0.0
        self.metrics.processing_time_ms = avg_processing_time * 1000
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current apoptosis metrics for monitoring and adaptation"""
        return {
            "apoptosis_rate": self.metrics.apoptosis_rate,
            "legacy_preservation_rate": self.metrics.legacy_preservation_rate,
            "crisis_detection_efficiency": self.metrics.crisis_detection_efficiency,
            "recovery_rate": self.metrics.recovery_rate,
            "processing_time_ms": self.metrics.processing_time_ms,
            "history_size": len(self.apoptosis_history),
            "circuit_breaker_state": self.circuit_breaker.get_state(),
            "initiated": self.initiated,
            "crisis_counter": self.crisis_counter
        }
    
    def get_ssp_integration_metrics(self) -> Dict[str, float]:
        """
        Get metrics suitable for SSP integration.
        
        Returns apoptosis metrics in a format SSP can use for reward shaping.
        """
        metrics = self.get_metrics()
        
        return {
            "apoptosis_rate": metrics["apoptosis_rate"],
            "legacy_preservation_rate": metrics["legacy_preservation_rate"],
            "crisis_detection_efficiency": metrics["crisis_detection_efficiency"],
            "processing_efficiency": 1.0 / (1.0 + metrics["processing_time_ms"]),
            "initiated": float(metrics["initiated"]),
            "crisis_counter": metrics["crisis_counter"]
        }
    
    def get_apoptosis_history(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get recent apoptosis events for analysis or reporting"""
        return [
            {
                "timestamp": c.timestamp,
                "cause": c.cause,
                "system_state": c.system_state,
                "legacy_artifact": c.legacy_artifact,
                "resolution_time": c.resolution_time,
                "success": c.success
            }
            for c in self.apoptosis_history[-n:]
        ]
    
    def reset(self):
        """Reset apoptosis system state"""
        self.crisis_counter = 0
        self.crisis_start_time = 0.0
        self.last_recovery_check = 0.0
        self.initiated = False
        self.metrics = ApoptosisMetrics()
        log.info("ApoptosisSystem reset")
    
    def get_cause(self) -> str:
        """Get the cause of the current apoptosis attempt"""
        if self.apoptosis_history:
            return self.apoptosis_history[-1].cause
        return "unknown"
    
    def is_initiated(self) -> bool:
        """Check if apoptosis has been initiated"""
        return self.initiated
    
    def get_crisis_duration(self) -> int:
        """Get current crisis duration in ticks"""
        return self.crisis_counter