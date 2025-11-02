# stephanie/components/jitter/regulation/homeostasis/adaptive_setpoints.py
"""
adaptive_setpoints.py
=====================
Implementation of adaptive setpoints for homeostatic regulation.

This module implements the adaptive setpoint system that learns and adjusts
target values based on system performance and environmental conditions.

Key Features:
- Adaptive setpoint learning from performance metrics
- Historical performance analysis for setpoint adjustment
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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field, validator

log = logging.getLogger("stephanie.jitter.regulation.homeostasis.adaptive")

class SetpointConfig(BaseModel):
    """Validated configuration for AdaptiveSetpoints"""
    learning_rate: float = Field(0.01, ge=0.001, le=0.1, description="Learning rate for setpoint adaptation")
    max_setpoint: float = Field(2.0, ge=1.0, le=5.0, description="Maximum setpoint value")
    min_setpoint: float = Field(0.1, ge=0.01, le=1.0, description="Minimum setpoint value")
    adaptation_smoothing: float = Field(0.9, ge=0.7, le=0.99, description="Smoothing factor for adaptation")
    max_history: int = Field(1000, ge=100, le=10000, description="Maximum history length")
    
    @validator('min_setpoint')
    def validate_min_max_setpoints(cls, v, values):
        if 'max_setpoint' in values and v >= values['max_setpoint']:
            raise ValueError('min_setpoint must be less than max_setpoint')
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
    def update_setpoint(name, value):
        # Setpoint update logic here
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
class SetpointRecord:
    """Record of a setpoint update for learning and adaptation"""
    timestamp: float = field(default_factory=time.time)
    name: str = ""
    old_value: float = 0.0
    new_value: float = 0.0
    performance: float = 0.0
    context: Dict[str, Any] = field(default_factory=dict)
    success: bool = False

@dataclass
class AdaptiveMetrics:
    """Metrics for adaptive setpoint performance"""
    adaptation_rate: float = 0.0
    setpoint_stability: float = 0.5
    learning_efficiency: float = 0.5
    performance_trend: float = 0.0
    processing_time_ms: float = 0.0

class AdaptiveSetpoints:
    """
    Implementation of adaptive setpoints for homeostatic regulation.
    
    Key Features:
    - Adaptive setpoint learning from performance metrics
    - Historical performance analysis for setpoint adjustment
    - Configuration validation with Pydantic
    - Circuit breaker pattern for resilience
    - Comprehensive telemetry and monitoring
    - SSP integration hooks
    - Performance optimizations
    """
    
    def __init__(self, cfg: Dict[str, Any]):
        try:
            # Validate configuration
            self.config = SetpointConfig(**cfg)
            log.info("AdaptiveSetpoints configuration validated successfully")
        except Exception as e:
            log.error(f"Configuration validation failed: {str(e)}")
            # Use safe defaults
            self.config = SetpointConfig()
        
        # Initialize setpoints
        self.setpoints = {
            "energy_balance": self.config.max_setpoint * 0.5,  # Default to middle value
            "boundary_integrity": self.config.max_setpoint * 0.8,
            "cognitive_flow": self.config.max_setpoint * 0.6,
            "vpm_diversity": self.config.max_setpoint * 0.5
        }
        
        # Initialize history
        self.history: List[SetpointRecord] = []
        
        # Initialize metrics
        self.metrics = AdaptiveMetrics()
        
        # Initialize performance tracking
        self.processing_times = []
        self.max_processing_history = 100
        
        # Initialize circuit breaker
        self.circuit_breaker = CircuitBreaker()
        
        log.info("AdaptiveSetpoints initialized with learning capabilities")
    
    @CircuitBreaker()
    def update_setpoint(self, name: str, new_value: float, performance: float, context: Optional[Dict[str, Any]] = None) -> float:
        """
        Update a setpoint based on performance and context.
        
        Args:
            name: Name of the setpoint to update
            new_value: New value for the setpoint
            performance: Current system performance (0-1)
            context: Additional context for the update
            
        Returns:
            Updated setpoint value
        """
        start_time = time.time()
        
        try:
            # Store current state for history
            old_value = self.setpoints.get(name, self.config.max_setpoint * 0.5)
            
            # Apply adaptive update
            updated_value = self._adaptive_update(name, old_value, new_value, performance)
            
            # Update setpoint
            self.setpoints[name] = updated_value
            
            # Record for history
            self._record_setpoint_update(
                name,
                old_value,
                updated_value,
                performance,
                context or {}
            )
            
            # Update metrics
            self._update_metrics()
            
            # Log processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > self.max_processing_history:
                self.processing_times.pop(0)
                
            log.debug(f"Updated setpoint {name} (old={old_value:.3f}, new={updated_value:.3f}, "
                     f"performance={performance:.3f})")
            return updated_value
            
        except Exception as e:
            log.error(f"Error updating setpoint: {str(e)}", exc_info=True)
            return self.setpoints.get(name, self.config.max_setpoint * 0.5)
    
    def _adaptive_update(self, name: str, old_value: float, new_value: float, performance: float) -> float:
        """Apply adaptive update to setpoint based on performance"""
        # If performance is good, adjust setpoint closer to desired value
        if performance > 0.7:
            # Move towards target value with learning rate
            target_value = new_value
            updated_value = old_value + self.config.learning_rate * (target_value - old_value)
        # If performance is poor, adjust setpoint to encourage better performance
        elif performance < 0.3:
            # Move away from current value to explore new settings
            updated_value = old_value + self.config.learning_rate * (0.5 - old_value)
        else:
            # Moderate adjustment
            updated_value = old_value + self.config.learning_rate * (new_value - old_value) * 0.5
        
        # Apply bounds
        updated_value = max(
            self.config.min_setpoint,
            min(self.config.max_setpoint, updated_value)
        )
        
        return updated_value
    
    def _record_setpoint_update(
        self,
        name: str,
        old_value: float,
        new_value: float,
        performance: float,
        context: Dict[str, Any]
    ):
        """Record setpoint update for learning and adaptation"""
        # Create record
        record = SetpointRecord(
            name=name,
            old_value=old_value,
            new_value=new_value,
            performance=performance,
            context=context,
            success=True  # Will be updated later if needed
        )
        
        # Add to history
        self.history.append(record)
        
        # Keep history bounded
        if len(self.history) > self.config.max_history:
            self.history.pop(0)
    
    def _update_metrics(self):
        """Update adaptive setpoint metrics based on history"""
        # Update adaptation rate
        if len(self.history) >= 10:
            # Calculate average change in setpoints
            changes = [
                abs(r.new_value - r.old_value) for r in self.history[-10:]
            ]
            self.metrics.adaptation_rate = np.mean(changes)
        
        # Update setpoint stability
        if len(self.history) >= 10:
            # Calculate variance of setpoint values
            setpoint_values = [r.new_value for r in self.history[-10:]]
            self.metrics.setpoint_stability = 1.0 / (1.0 + np.var(setpoint_values))
        
        # Update learning efficiency
        if len(self.history) >= 5:
            # Calculate performance trend
            performances = [r.performance for r in self.history[-5:]]
            if len(performances) > 1:
                self.metrics.performance_trend = performances[-1] - performances[0]
        
        # Update processing time
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0.0
        self.metrics.processing_time_ms = avg_processing_time * 1000
    
    def get_setpoints(self) -> Dict[str, float]:
        """Get current setpoints"""
        return self.setpoints.copy()
    
    def get_setpoint(self, name: str) -> float:
        """Get specific setpoint value"""
        return self.setpoints.get(name, self.config.max_setpoint * 0.5)
    
    def set_setpoint(self, name: str, value: float):
        """Set specific setpoint value with bounds checking"""
        # Apply bounds
        bounded_value = max(
            self.config.min_setpoint,
            min(self.config.max_setpoint, value)
        )
        
        # Update setpoint
        self.setpoints[name] = bounded_value
        log.debug(f"Setpoint {name} set to {bounded_value:.3f}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current adaptive setpoint metrics for monitoring and adaptation"""
        return {
            "adaptation_rate": self.metrics.adaptation_rate,
            "setpoint_stability": self.metrics.setpoint_stability,
            "learning_efficiency": self.metrics.learning_efficiency,
            "performance_trend": self.metrics.performance_trend,
            "processing_time_ms": self.metrics.processing_time_ms,
            "history_size": len(self.history),
            "circuit_breaker_state": self.circuit_breaker.get_state()
        }
    
    def get_ssp_integration_metrics(self) -> Dict[str, float]:
        """
        Get metrics suitable for SSP integration.
        
        Returns adaptive setpoint metrics in a format SSP can use for reward shaping.
        """
        metrics = self.get_metrics()
        
        return {
            "adaptation_rate": metrics["adaptation_rate"],
            "setpoint_stability": metrics["setpoint_stability"],
            "learning_efficiency": metrics["learning_efficiency"],
            "performance_trend": metrics["performance_trend"],
            "processing_efficiency": 1.0 / (1.0 + metrics["processing_time_ms"])
        }
    
    def evaluate_setpoint_success(
        self,
        setpoint_id: str,
        expected_performance: float,
        actual_performance: float
    ) -> bool:
        """
        Evaluate whether setpoint adjustment was successful.
        
        Args:
            setpoint_id: ID of the setpoint update to evaluate
            expected_performance: Expected performance after adjustment
            actual_performance: Actual performance after adjustment
            
        Returns:
            True if adjustment was successful, False otherwise
        """
        # Find the setpoint record
        record = next((r for r in self.history if r.name == setpoint_id), None)
        if not record:
            log.warning(f"Setpoint record not found: {setpoint_id}")
            return False
        
        # Determine success based on performance improvement
        success = actual_performance > expected_performance
        
        # Update record
        record.success = success
        
        # Update metrics
        if success:
            self.metrics.learning_efficiency = min(1.0, self.metrics.learning_efficiency + 0.01)
        else:
            self.metrics.learning_efficiency = max(0.0, self.metrics.learning_efficiency - 0.01)
        
        log.debug(f"Setpoint evaluation: {setpoint_id} - success={success}")
        return success
    
    def get_recent_updates(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get recent setpoint updates for analysis or reporting"""
        return [
            {
                "timestamp": r.timestamp,
                "name": r.name,
                "old_value": r.old_value,
                "new_value": r.new_value,
                "performance": r.performance,
                "context": r.context,
                "success": r.success
            }
            for r in self.history[-n:]
        ]
    
    def get_optimal_setpoint_profile(
        self,
        context: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Get optimal setpoint profile for given context based on historical data.
        
        Args:
            context: Current context for which to determine optimal profile
            
        Returns:
            Optimal setpoint values for the context
        """
        if len(self.history) < 10:
            return self.setpoints.copy()
        
        # Find similar contexts
        similar = []
        for record in self.history:
            # Simple similarity measure (would be more sophisticated in production)
            context_match = 0.0
            for key in context:
                if key in record.context:
                    # Numeric context
                    if isinstance(context[key], (int, float)):
                        context_match += 1.0 - min(1.0, abs(context[key] - record.context[key]))
                    # Categorical context
                    else:
                        context_match += 1.0 if context[key] == record.context[key] else 0.0
            
            # Normalize
            context_match = context_match / max(1, len(context))
            if context_match > 0.5:  # Minimum similarity
                similar.append((record, context_match))
        
        # If no similar contexts, return current setpoints
        if not similar:
            return self.setpoints.copy()
        
        # Calculate weighted average of setpoint values
        total_weight = 0.0
        avg_setpoints = {name: 0.0 for name in self.setpoints}
        
        for record, weight in similar:
            total_weight += weight
            for name in avg_setpoints:
                avg_setpoints[name] += record.new_value * weight
        
        # Normalize
        if total_weight > 0:
            for name in avg_setpoints:
                avg_setpoints[name] /= total_weight
        
        # Ensure setpoints are within bounds
        for name in avg_setpoints:
            avg_setpoints[name] = max(
                self.config.min_setpoint,
                min(self.config.max_setpoint, avg_setpoints[name])
            )
        
        return avg_setpoints