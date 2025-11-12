# stephanie/components/jitter/regulation/reproduction/reproduction_system.py
"""
reproduction_system.py
======================
Implementation of the core reproduction system for the Jitter Autopoietic System.

This module implements the fundamental reproduction logic that allows Jitter
organisms to create offspring while maintaining the autopoietic system's
organizationally closed production.

Key Features:
- Reproduction readiness detection based on energy and health
- Controlled variation for genetic diversity
- Configuration validation with Pydantic
- Circuit breaker pattern for resilience
- Comprehensive telemetry and monitoring
- SSP integration hooks
- Performance optimizations
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field, validator

log = logging.getLogger("stephanie.jitter.regulation.reproduction.system")

class ReproductionConfig(BaseModel):
    """Validated configuration for ReproductionSystem"""
    ready_threshold: float = Field(80.0, ge=50.0, le=100.0, description="Energy threshold for reproduction readiness")
    variation_rate: float = Field(0.1, ge=0.01, le=0.5, description="Rate of genetic variation")
    reproduction_interval: int = Field(1000, ge=100, le=10000, description="Minimum ticks between reproduction")
    energy_threshold: float = Field(80.0, ge=50.0, le=100.0, description="Minimum energy for reproduction")
    health_threshold: float = Field(0.7, ge=0.5, le=0.95, description="Minimum health for reproduction")
    max_offspring: int = Field(5, ge=1, le=20, description="Maximum offspring per organism")
    
    @validator('ready_threshold', 'energy_threshold')
    def validate_thresholds(cls, v, values):
        if 'health_threshold' in values and v < values['health_threshold']:
            raise ValueError('ready_threshold must be greater than or equal to health_threshold')
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
    def can_reproduce(core):
        # Reproduction logic here
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
class ReproductionRecord:
    """Record of a reproduction event for learning and analysis"""
    timestamp: float = field(default_factory=time.time)
    parent_id: str = ""
    offspring_id: str = ""
    genetic_material: Dict[str, Any] = field(default_factory=dict)
    reproduction_quality: float = 0.0
    energy_used: float = 0.0
    success: bool = False
    cause: str = ""

@dataclass
class ReproductionMetrics:
    """Metrics for reproduction system performance"""
    reproduction_rate: float = 0.0
    offspring_quality: float = 0.0
    energy_efficiency: float = 0.0
    genetic_diversity: float = 0.0
    processing_time_ms: float = 0.0

class ReproductionSystem:
    """
    Implementation of the core reproduction system for the Jitter Autopoietic System.
    
    Key Features:
    - Reproduction readiness detection based on energy and health
    - Controlled variation for genetic diversity
    - Configuration validation with Pydantic
    - Circuit breaker pattern for resilience
    - Comprehensive telemetry and monitoring
    - SSP integration hooks
    - Performance optimizations
    """
    
    def __init__(self, cfg: Dict[str, Any]):
        try:
            # Validate configuration
            self.config = ReproductionConfig(**cfg)
            log.info("ReproductionSystem configuration validated successfully")
        except Exception as e:
            log.error(f"Configuration validation failed: {str(e)}")
            # Use safe defaults
            self.config = ReproductionConfig()
        
        # Initialize tracking variables
        self.reproduction_counter = 0
        self.last_reproduction = 0
        self.offspring_count = 0
        self.ready = False
        
        # Initialize history
        self.reproduction_history: List[ReproductionRecord] = []
        
        # Initialize metrics
        self.metrics = ReproductionMetrics()
        
        # Initialize performance tracking
        self.processing_times = []
        self.max_processing_history = 100
        
        # Initialize circuit breaker
        self.circuit_breaker = CircuitBreaker()
        
        log.info("ReproductionSystem initialized with controlled variation")
    
    @CircuitBreaker()
    def can_reproduce(self, core) -> bool:
        """
        Determine if the organism is ready for reproduction.
        
        Args:
            core: The AutopoieticCore instance to check
            
        Returns:
            True if reproduction is ready, False otherwise
        """
        start_time = time.time()
        
        try:
            # Check if enough time has passed since last reproduction
            if time.time() - self.last_reproduction < self.config.reproduction_interval:
                return False
            
            # Get current system state
            energy_balance = core.energy.level("metabolic") + core.energy.level("cognitive")
            health = core.homeostasis.get_telemetry().get("health", 0.5)
            
            # Check energy threshold
            if energy_balance < self.config.energy_threshold:
                return False
            
            # Check health threshold
            if health < self.config.health_threshold:
                return False
            
            # Check ready threshold
            if energy_balance < self.config.ready_threshold:
                return False
            
            # Set ready flag
            self.ready = True
            
            # Log processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > self.max_processing_history:
                self.processing_times.pop(0)
                
            log.debug(f"Reproduction readiness check (energy={energy_balance:.2f}, health={health:.2f})")
            return True
            
        except Exception as e:
            log.error(f"Error checking reproduction readiness: {str(e)}", exc_info=True)
            return False
    
    @CircuitBreaker()
    def reproduce(self, core) -> Optional[Dict[str, Any]]:
        """
        Create offspring through reproduction process.
        
        Args:
            core: The AutopoieticCore instance to reproduce from
            
        Returns:
            Dictionary with offspring genetic material or None if reproduction failed
        """
        start_time = time.time()
        
        try:
            # Check if reproduction is ready
            if not self.can_reproduce(core):
                log.warning("Reproduction not ready")
                return None
            
            # Create offspring genetic material
            offspring_genetics = self._create_offspring_genetics(core)
            
            # Record reproduction event
            self._record_reproduction_event(
                core.id,
                offspring_genetics,
                core.energy.level("metabolic") + core.energy.level("cognitive")
            )
            
            # Update metrics
            self._update_metrics()
            
            # Reset ready flag and update last reproduction time
            self.ready = False
            self.last_reproduction = time.time()
            self.offspring_count += 1
            
            # Log processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > self.max_processing_history:
                self.processing_times.pop(0)
                
            log.info(f"Reproduction successful (offspring #{self.offspring_count})")
            return offspring_genetics
            
        except Exception as e:
            log.error(f"Error during reproduction: {str(e)}", exc_info=True)
            return None
    
    def _create_offspring_genetics(self, parent_core) -> Dict[str, Any]:
        """Create offspring genetic material with controlled variation"""
        # Create base genetic material from parent
        parent_genetics = {
            "id": f"offspring_{int(time.time())}_{uuid.uuid4().hex[:8]}",
            "parent_id": parent_core.id,
            "generation": getattr(parent_core, 'generation', 0) + 1,
            "timestamp": time.time(),
            "energy_levels": {
                "cognitive": parent_core.energy.level("cognitive"),
                "metabolic": parent_core.energy.level("metabolic"),
                "reserve": parent_core.energy.level("reserve")
            },
            "boundary_integrity": parent_core.membrane.integrity,
            "boundary_thickness": parent_core.membrane.thickness,
            "mutation_rate": self.config.variation_rate,
            "heritage": [parent_core.id]  # Initial heritage
        }
        
        # Apply controlled variation
        mutated_genetics = self._apply_mutation(parent_genetics)
        
        return mutated_genetics
    
    def _apply_mutation(self, genetics: Dict[str, Any]) -> Dict[str, Any]:
        """Apply controlled mutation to genetic material"""
        # Create copy to avoid modifying original
        mutated = {k: v for k, v in genetics.items()}
        
        # Mutate energy levels
        for pool in ["cognitive", "metabolic", "reserve"]:
            if pool in mutated["energy_levels"]:
                # Apply random variation within bounds
                variation = np.random.normal(0, self.config.variation_rate)
                mutated["energy_levels"][pool] = max(
                    0.0, 
                    min(100.0, mutated["energy_levels"][pool] * (1 + variation))
                )
        
        # Mutate boundary properties
        if "boundary_integrity" in mutated:
            variation = np.random.normal(0, self.config.variation_rate)
            mutated["boundary_integrity"] = max(
                0.0, 
                min(1.0, mutated["boundary_integrity"] * (1 + variation))
            )
        
        if "boundary_thickness" in mutated:
            variation = np.random.normal(0, self.config.variation_rate)
            mutated["boundary_thickness"] = max(
                0.0, 
                min(1.0, mutated["boundary_thickness"] * (1 + variation))
            )
        
        # Add new heritage entry
        if "heritage" in mutated:
            mutated["heritage"].append(mutated["id"])
            # Keep heritage bounded
            if len(mutated["heritage"]) > 10:
                mutated["heritage"] = mutated["heritage"][-10:]
        
        return mutated
    
    def _record_reproduction_event(
        self,
        parent_id: str,
        offspring_genetics: Dict[str, Any],
        energy_used: float
    ):
        """Record reproduction event for learning and analysis"""
        # Create record
        record = ReproductionRecord(
            parent_id=parent_id,
            offspring_id=offspring_genetics.get("id", ""),
            genetic_material=offspring_genetics,
            energy_used=energy_used,
            success=True,
            cause="reproduction_successful"
        )
        
        # Add to history
        self.reproduction_history.append(record)
        
        # Keep history bounded
        if len(self.reproduction_history) > 1000:
            self.reproduction_history.pop(0)
    
    def _update_metrics(self):
        """Update reproduction metrics based on history"""
        # Update reproduction rate
        if len(self.reproduction_history) > 0:
            self.metrics.reproduction_rate = len(self.reproduction_history) / max(1, len(self.reproduction_history))
        
        # Update offspring quality (simplified)
        if len(self.reproduction_history) > 0:
            # Calculate average energy levels
            total_energy = sum(r.genetic_material.get("energy_levels", {}).get("cognitive", 0) 
                              for r in self.reproduction_history[-10:])
            self.metrics.offspring_quality = total_energy / max(1, len(self.reproduction_history[-10:]))
        
        # Update energy efficiency
        if len(self.reproduction_history) > 0:
            total_energy_used = sum(r.energy_used for r in self.reproduction_history)
            self.metrics.energy_efficiency = total_energy_used / max(1, len(self.reproduction_history))
        
        # Update processing time
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0.0
        self.metrics.processing_time_ms = avg_processing_time * 1000
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current reproduction metrics for monitoring and adaptation"""
        return {
            "reproduction_rate": self.metrics.reproduction_rate,
            "offspring_quality": self.metrics.offspring_quality,
            "energy_efficiency": self.metrics.energy_efficiency,
            "genetic_diversity": self.metrics.genetic_diversity,
            "processing_time_ms": self.metrics.processing_time_ms,
            "history_size": len(self.reproduction_history),
            "circuit_breaker_state": self.circuit_breaker.get_state(),
            "offspring_count": self.offspring_count,
            "ready": self.ready
        }
    
    def get_ssp_integration_metrics(self) -> Dict[str, float]:
        """
        Get metrics suitable for SSP integration.
        
        Returns reproduction metrics in a format SSP can use for reward shaping.
        """
        metrics = self.get_metrics()
        
        return {
            "reproduction_rate": metrics["reproduction_rate"],
            "offspring_quality": metrics["offspring_quality"],
            "energy_efficiency": metrics["energy_efficiency"],
            "genetic_diversity": metrics["genetic_diversity"],
            "processing_efficiency": 1.0 / (1.0 + metrics["processing_time_ms"]),
            "offspring_count": metrics["offspring_count"],
            "ready": float(metrics["ready"])
        }
    
    def get_reproduction_history(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get recent reproduction events for analysis or reporting"""
        return [
            {
                "timestamp": r.timestamp,
                "parent_id": r.parent_id,
                "offspring_id": r.offspring_id,
                "genetic_material": r.genetic_material,
                "reproduction_quality": r.reproduction_quality,
                "energy_used": r.energy_used,
                "success": r.success,
                "cause": r.cause
            }
            for r in self.reproduction_history[-n:]
        ]
    
    def reset(self):
        """Reset reproduction system state"""
        self.reproduction_counter = 0
        self.last_reproduction = 0
        self.offspring_count = 0
        self.ready = False
        self.metrics = ReproductionMetrics()
        log.info("ReproductionSystem reset")
    
    def get_offspring_count(self) -> int:
        """Get total number of offspring produced"""
        return self.offspring_count
    
    def is_ready(self) -> bool:
        """Check if reproduction is currently ready"""
        return self.ready
    
    def get_reproduction_interval(self) -> int:
        """Get minimum reproduction interval"""
        return self.config.reproduction_interval