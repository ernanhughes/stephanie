# stephanie/components/jitter/regulation/homeostasis/crisis_detector.py
"""
crisis_detector.py
==================
Implementation of crisis detection for homeostatic regulation.

This module implements the crisis detection system that identifies when
the autopoietic system is approaching failure or critical conditions.

Key Features:
- Multi-level crisis detection based on system metrics
- Crisis response protocols for different severity levels
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
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field, validator

log = logging.getLogger("stephanie.jitter.regulation.homeostasis.crisis")

class CrisisLevel(str, Enum):
    """Levels of crisis severity"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class CrisisConfig(BaseModel):
    """Validated configuration for CrisisDetector"""
    low_threshold: float = Field(0.3, ge=0.1, le=0.5, description="Low crisis threshold")
    medium_threshold: float = Field(0.6, ge=0.4, le=0.8, description="Medium crisis threshold")
    high_threshold: float = Field(0.8, ge=0.6, le=0.95, description="High crisis threshold")
    critical_threshold: float = Field(0.95, ge=0.9, le=0.99, description="Critical crisis threshold")
    max_history: int = Field(100, ge=50, le=500, description="Maximum history length for crisis analysis")
    response_timeout: float = Field(30.0, ge=10.0, le=120.0, description="Timeout for crisis response")
    
    @validator('low_threshold', 'medium_threshold', 'high_threshold', 'critical_threshold')
    def validate_thresholds(cls, v, values):
        thresholds = ['low_threshold', 'medium_threshold', 'high_threshold', 'critical_threshold']
        if len(values) >= 4:
            for i, threshold in enumerate(thresholds):
                if threshold in values and i < len(thresholds) - 1:
                    if values[threshold] >= values[thresholds[i+1]]:
                        raise ValueError(f'{threshold} must be less than {thresholds[i+1]}')
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
    def detect_crisis(metrics):
        # Crisis detection logic here
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
class CrisisRecord:
    """Record of a crisis event for learning and response"""
    timestamp: float = field(default_factory=time.time)
    crisis_level: str = CrisisLevel.NONE.value
    metrics: Dict[str, float] = field(default_factory=dict)
    response_actions: List[str] = field(default_factory=list)
    resolution_time: Optional[float] = None
    success: bool = False

@dataclass
class CrisisMetrics:
    """Metrics for crisis detection performance"""
    crisis_frequency: float = 0.0
    response_efficiency: float = 0.0
    resolution_rate: float = 0.0
    crisis_trend: float = 0.0
    processing_time_ms: float = 0.0

class CrisisDetector:
    """
    Implementation of crisis detection for homeostatic regulation.
    
    Key Features:
    - Multi-level crisis detection based on system metrics
    - Crisis response protocols for different severity levels
    - Configuration validation with Pydantic
    - Circuit breaker pattern for resilience
    - Comprehensive telemetry and monitoring
    - SSP integration hooks
    - Performance optimizations
    """
    
    def __init__(self, cfg: Dict[str, Any]):
        try:
            # Validate configuration
            self.config = CrisisConfig(**cfg)
            log.info("CrisisDetector configuration validated successfully")
        except Exception as e:
            log.error(f"Configuration validation failed: {str(e)}")
            # Use safe defaults
            self.config = CrisisConfig()
        
        # Initialize crisis levels
        self.crisis_levels = {
            CrisisLevel.NONE: 0.0,
            CrisisLevel.LOW: self.config.low_threshold,
            CrisisLevel.MEDIUM: self.config.medium_threshold,
            CrisisLevel.HIGH: self.config.high_threshold,
            CrisisLevel.CRITICAL: self.config.critical_threshold
        }
        
        # Initialize history
        self.crisis_history: List[CrisisRecord] = []
        
        # Initialize metrics
        self.metrics = CrisisMetrics()
        
        # Initialize performance tracking
        self.processing_times = []
        self.max_processing_history = 100
        
        # Initialize circuit breaker
        self.circuit_breaker = CircuitBreaker()
        
        # Initialize crisis tracking
        self.current_crisis_level = CrisisLevel.NONE
        self.crisis_start_time = 0.0
        self.crisis_response_time = 0.0
        
        log.info("CrisisDetector initialized with multi-level crisis detection")
    
    @CircuitBreaker()
    def detect_crisis(self, metrics: Dict[str, Any]) -> CrisisLevel:
        """
        Detect crisis level based on current system metrics.
        
        Args:
            metrics: Current system metrics
            
        Returns:
            Current crisis level
        """
        start_time = time.time()
        
        try:
            # Calculate crisis level based on metrics
            crisis_level = self._calculate_crisis_level(metrics)
            
            # Record crisis event if level changed
            if crisis_level != self.current_crisis_level:
                self._record_crisis_event(crisis_level, metrics)
            
            # Update current crisis level
            self.current_crisis_level = crisis_level
            
            # Update metrics
            self._update_metrics()
            
            # Log processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > self.max_processing_history:
                self.processing_times.pop(0)
                
            log.debug(f"Crisis detected (level={crisis_level.value})")
            return crisis_level
            
        except Exception as e:
            log.error(f"Error detecting crisis: {str(e)}", exc_info=True)
            return CrisisLevel.NONE
    
    def _calculate_crisis_level(self, metrics: Dict[str, Any]) -> CrisisLevel:
        """Calculate crisis level based on system metrics"""
        # Get key metrics
        energy_balance = metrics.get("energy_balance", 0.5)
        boundary_integrity = metrics.get("boundary_integrity", 0.5)
        cognitive_flow = metrics.get("cognitive_flow", 0.5)
        vpm_diversity = metrics.get("vpm_diversity", 0.5)
        
        # Calculate combined crisis score
        # Weighted average of system health indicators
        crisis_score = (
            (1.0 - energy_balance) * 0.3 +
            (1.0 - boundary_integrity) * 0.3 +
            (1.0 - cognitive_flow) * 0.2 +
            (1.0 - vpm_diversity) * 0.2
        )
        
        # Determine crisis level
        if crisis_score >= self.config.critical_threshold:
            return CrisisLevel.CRITICAL
        elif crisis_score >= self.config.high_threshold:
            return CrisisLevel.HIGH
        elif crisis_score >= self.config.medium_threshold:
            return CrisisLevel.MEDIUM
        elif crisis_score >= self.config.low_threshold:
            return CrisisLevel.LOW
        else:
            return CrisisLevel.NONE
    
    def _record_crisis_event(self, crisis_level: CrisisLevel, metrics: Dict[str, Any]):
        """Record crisis event for learning and response"""
        # Create record
        record = CrisisRecord(
            crisis_level=crisis_level.value,
            metrics=metrics.copy(),
            response_actions=[]
        )
        
        # Add to history
        self.crisis_history.append(record)
        
        # Keep history bounded
        if len(self.crisis_history) > self.config.max_history:
            self.crisis_history.pop(0)
    
    def _update_metrics(self):
        """Update crisis detection metrics based on history"""
        # Update crisis frequency
        if len(self.crisis_history) > 0:
            # Count crisis events in recent history
            recent_crisis = sum(1 for c in self.crisis_history[-50:] 
                               if c.crisis_level != CrisisLevel.NONE.value)
            self.metrics.crisis_frequency = recent_crisis / max(1, len(self.crisis_history[-50:]))
        
        # Update crisis trend
        if len(self.crisis_history) >= 10:
            recent_levels = [
                self.crisis_levels[CrisisLevel(c.crisis_level)] 
                for c in self.crisis_history[-10:]
            ]
            if len(recent_levels) > 1:
                self.metrics.crisis_trend = recent_levels[-1] - recent_levels[0]
        
        # Update processing time
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0.0
        self.metrics.processing_time_ms = avg_processing_time * 1000
    
    def get_crisis_metrics(self) -> Dict[str, Any]:
        """Get current crisis detection metrics for monitoring and adaptation"""
        return {
            "crisis_frequency": self.metrics.crisis_frequency,
            "response_efficiency": self.metrics.response_efficiency,
            "resolution_rate": self.metrics.resolution_rate,
            "crisis_trend": self.metrics.crisis_trend,
            "processing_time_ms": self.metrics.processing_time_ms,
            "history_size": len(self.crisis_history),
            "circuit_breaker_state": self.circuit_breaker.get_state(),
            "current_crisis_level": self.current_crisis_level.value
        }
    
    def get_ssp_integration_metrics(self) -> Dict[str, float]:
        """
        Get metrics suitable for SSP integration.
        
        Returns crisis detection metrics in a format SSP can use for reward shaping.
        """
        metrics = self.get_crisis_metrics()
        
        return {
            "crisis_frequency": metrics["crisis_frequency"],
            "crisis_trend": metrics["crisis_trend"],
            "current_crisis_level": self.crisis_levels[CrisisLevel(self.current_crisis_level.value)],
            "processing_efficiency": 1.0 / (1.0 + metrics["processing_time_ms"])
        }
    
    def get_crisis_response_plan(self, crisis_level: CrisisLevel) -> List[str]:
        """
        Get response plan for a specific crisis level.
        
        Args:
            crisis_level: Current crisis level
            
        Returns:
            List of response actions
        """
        response_plans = {
            CrisisLevel.NONE: [],
            CrisisLevel.LOW: [
                "monitor_closely",
                "adjust_homeostasis",
                "optimize_energy_allocation"
            ],
            CrisisLevel.MEDIUM: [
                "conserve_energy",
                "fortify_boundary",
                "reduce_cognitive_load",
                "alert_monitoring"
            ],
            CrisisLevel.HIGH: [
                "initiate_apoptosis",
                "preserve_legacy",
                "alert_emergency",
                "emergency_repair"
            ],
            CrisisLevel.CRITICAL: [
                "immediate_apoptosis",
                "preserve_legacy",
                "emergency_shutdown",
                "alert_critical"
            ]
        }
        
        return response_plans.get(crisis_level, [])
    
    def evaluate_crisis_response(
        self,
        crisis_id: str,
        response_actions: List[str],
        outcome: Dict[str, Any]
    ) -> bool:
        """
        Evaluate whether crisis response was successful.
        
        Args:
            crisis_id: ID of the crisis event to evaluate
            response_actions: Actions taken during response
            outcome: Outcome of the crisis response
            
        Returns:
            True if response was successful, False otherwise
        """
        # Find the crisis record
        record = next((c for c in self.crisis_history if c.crisis_level == crisis_id), None)
        if not record:
            log.warning(f"Crisis record not found: {crisis_id}")
            return False
        
        # Determine success based on outcome
        success = outcome.get("resolved", False)
        
        # Update record
        record.response_actions = response_actions
        record.resolution_time = time.time()
        record.success = success
        
        # Update metrics
        if success:
            self.metrics.resolution_rate = min(1.0, self.metrics.resolution_rate + 0.01)
        else:
            self.metrics.resolution_rate = max(0.0, self.metrics.resolution_rate - 0.01)
        
        log.debug(f"Crisis response evaluation: {crisis_id} - success={success}")
        return success
    
    def get_recent_crisis_events(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get recent crisis events for analysis or reporting"""
        return [
            {
                "timestamp": c.timestamp,
                "crisis_level": c.crisis_level,
                "metrics": c.metrics,
                "response_actions": c.response_actions,
                "resolution_time": c.resolution_time,
                "success": c.success
            }
            for c in self.crisis_history[-n:]
        ]
    
    def get_crisis_trend_analysis(self) -> Dict[str, Any]:
        """
        Analyze recent crisis trends for predictive capabilities.
        
        Returns:
            Analysis of crisis trends and patterns
        """
        if len(self.crisis_history) < 10:
            return {
                "trend": "stable",
                "frequency": 0.0,
                "severity_distribution": {
                    CrisisLevel.NONE.value: 0.0,
                    CrisisLevel.LOW.value: 0.0,
                    CrisisLevel.MEDIUM.value: 0.0,
                    CrisisLevel.HIGH.value: 0.0,
                    CrisisLevel.CRITICAL.value: 0.0
                }
            }
        
        # Analyze recent trends
        recent = self.crisis_history[-20:]
        total_events = len(recent)
        
        # Calculate severity distribution
        severity_counts = {level.value: 0 for level in CrisisLevel}
        for c in recent:
            severity_counts[c.crisis_level] += 1
        
        # Normalize
        severity_distribution = {
            level: count / total_events for level, count in severity_counts.items()
        }
        
        # Calculate trend
        if len(recent) >= 5:
            recent_severity = [self.crisis_levels[CrisisLevel(c.crisis_level)] for c in recent]
            trend = recent_severity[-1] - recent_severity[0]
            trend_direction = "increasing" if trend > 0.1 else "decreasing" if trend < -0.1 else "stable"
        else:
            trend_direction = "stable"
        
        return {
            "trend": trend_direction,
            "frequency": len(recent) / 20.0,
            "severity_distribution": severity_distribution
        }