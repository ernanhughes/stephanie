"""
controller.py
=============
Implementation of PID controllers for homeostatic regulation.

This module implements PID (Proportional-Integral-Derivative) controllers for
precise regulation of physiological parameters in the autopoietic system.

Key Features:
- PID controller with anti-windup protection
- Configurable setpoints and tuning parameters
- Adaptive PID parameters based on system state
- Configuration validation with Pydantic
- Circuit breaker pattern for resilience
- Comprehensive telemetry and monitoring
- SSP integration hooks
- Performance optimizations
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import logging
import time
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator
from enum import Enum
from functools import wraps

log = logging.getLogger("stephanie.jitter.regulation.homeostasis.controller")

class ControllerConfig(BaseModel):
    """Validated configuration for PIDController"""
    kp: float = Field(1.0, ge=0.0, le=10.0, description="Proportional gain")
    ki: float = Field(0.1, ge=0.0, le=1.0, description="Integral gain")
    kd: float = Field(0.05, ge=0.0, le=1.0, description="Derivative gain")
    setpoint: float = Field(1.0, ge=0.0, le=2.0, description="Target setpoint")
    output_limits: Tuple[float, float] = Field((-0.5, 0.5), description="Output limits")
    anti_windup: bool = Field(True, description="Enable anti-windup protection")
    
    @validator('output_limits')
    def validate_output_limits(cls, v):
        if v[0] >= v[1]:
            raise ValueError('output_limits must be (min, max) with min < max')
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
    def update(measurement):
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
class ControllerMetrics:
    """Metrics for controller performance"""
    performance: float = 0.5
    error_variance: float = 0.0
    integral_term: float = 0.0
    derivative_term: float = 0.0
    processing_time_ms: float = 0.0

class PIDController:
    """
    Implementation of a PID (Proportional-Integral-Derivative) controller for homeostatic regulation.
    
    Key Features:
    - PID controller with anti-windup protection
    - Configurable setpoints and tuning parameters
    - Adaptive PID parameters based on system state
    - Configuration validation with Pydantic
    - Circuit breaker pattern for resilience
    - Comprehensive telemetry and monitoring
    - SSP integration hooks
    - Performance optimizations
    """
    
    def __init__(self, cfg: Dict[str, Any]):
        try:
            # Validate configuration
            self.config = ControllerConfig(**cfg)
            log.info("PIDController configuration validated successfully")
        except Exception as e:
            log.error(f"Configuration validation failed: {str(e)}")
            # Use safe defaults
            self.config = ControllerConfig()
        
        # Initialize controller state variables
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None
        
        # Initialize metrics
        self.metrics = ControllerMetrics()
        
        # Initialize performance tracking
        self.processing_times = []
        self.max_processing_history = 100
        
        # Initialize circuit breaker
        self.circuit_breaker = CircuitBreaker()
        
        log.info("PIDController initialized with configurable parameters")
    
    @CircuitBreaker()
    def update(self, measurement: float, dt: Optional[float] = None) -> float:
        """
        Update PID controller with new measurement.
        
        Args:
            measurement: Current measurement from the system
            dt: Time step (if None, uses last known time)
            
        Returns:
            Control output (action to take)
        """
        start_time = time.time()
        
        try:
            # Get current time
            current_time = time.time()
            
            # Calculate time difference
            if dt is None and self.prev_time is not None:
                dt = current_time - self.prev_time
            elif dt is None:
                dt = 0.1  # Default time step
            
            # Calculate error
            error = self.config.setpoint - measurement
            
            # Proportional term
            p_term = self.config.kp * error
            
            # Integral term
            self.integral += self.config.ki * error * dt
            
            # Anti-windup: limit integral to output range
            if self.config.anti_windup:
                min_out, max_out = self.config.output_limits
                # Prevent integral from driving output beyond limits
                integral_limit = (max_out - min_out) / (self.config.ki + 1e-8)
                self.integral = np.clip(
                    self.integral, 
                    -integral_limit, 
                    integral_limit
                )
            
            # Derivative term
            d_term = 0.0
            if self.prev_time is not None:
                d_term = self.config.kd * (error - self.prev_error) / dt
            
            # Calculate output
            output = p_term + self.integral + d_term
            
            # Apply output limits
            output = np.clip(output, *self.config.output_limits)
            
            # Update state
            self.prev_error = error
            self.prev_time = current_time
            
            # Update metrics
            self._update_metrics(error, p_term, self.integral, d_term)
            
            # Log processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > self.max_processing_history:
                self.processing_times.pop(0)
                
            log.debug(f"PID update (measurement={measurement:.3f}, output={output:.3f})")
            return float(output)
            
        except Exception as e:
            log.error(f"Error updating PID controller: {str(e)}", exc_info=True)
            # Return safe default
            return 0.0
    
    def _update_metrics(self, error: float, p_term: float, integral_term: float, d_term: float):
        """Update controller metrics based on current state"""
        # Update performance (inverse of error variance)
        self.metrics.error_variance = error ** 2
        self.metrics.performance = 1.0 / (1.0 + self.metrics.error_variance)
        
        # Update terms
        self.metrics.integral_term = integral_term
        self.metrics.derivative_term = d_term
        
        # Update processing time
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0.0
        self.metrics.processing_time_ms = avg_processing_time * 1000
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current controller metrics for monitoring and adaptation"""
        return {
            "performance": self.metrics.performance,
            "error_variance": self.metrics.error_variance,
            "integral_term": self.metrics.integral_term,
            "derivative_term": self.metrics.derivative_term,
            "processing_time_ms": self.metrics.processing_time_ms,
            "circuit_breaker_state": self.circuit_breaker.get_state()
        }
    
    def get_ssp_integration_metrics(self) -> Dict[str, float]:
        """
        Get metrics suitable for SSP integration.
        
        Returns controller metrics in a format SSP can use for reward shaping.
        """
        metrics = self.get_metrics()
        
        return {
            "controller_performance": metrics["performance"],
            "error_variance": metrics["error_variance"],
            "integral_term": metrics["integral_term"],
            "derivative_term": metrics["derivative_term"],
            "processing_efficiency": 1.0 / (1.0 + metrics["processing_time_ms"])
        }
    
    def reset(self):
        """Reset controller state variables"""
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None
        self.metrics = ControllerMetrics()
        log.info("PIDController state reset")
    
    def adapt_parameters(self, performance: float, system_state: Dict[str, Any]):
        """
        Adapt PID parameters based on performance and system state.
        
        Args:
            performance: System performance (0-1)
            system_state: Current system state for context
        """
        # If performance is poor, increase gains to respond more strongly
        if performance < 0.3:
            self.config.kp = min(10.0, self.config.kp * 1.1)
            self.config.ki = min(1.0, self.config.ki * 1.1)
            self.config.kd = min(1.0, self.config.kd * 1.1)
        # If performance is good, reduce gains to avoid overcorrection
        elif performance > 0.7:
            self.config.kp = max(0.1, self.config.kp * 0.9)
            self.config.ki = max(0.01, self.config.ki * 0.9)
            self.config.kd = max(0.01, self.config.kd * 0.9)

        # Ensure parameters stay within bounds
        self.config.kp = np.clip(self.config.kp, 0.0, 10.0)
        self.config.ki = np.clip(self.config.ki, 0.0, 1.0)
        self.config.kd = np.clip(self.config.kd, 0.0, 1.0)

        log.debug(f"Adapted PID parameters (performance={performance:.3f}, "
                 f"kp={self.config.kp:.3f}, ki={self.config.ki:.3f}, kd={self.config.kd:.3f})")