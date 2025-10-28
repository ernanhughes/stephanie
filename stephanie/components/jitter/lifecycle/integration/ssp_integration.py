# stephanie/components/jitter/lifecycle/integration/ssp_integration.py
"""
ssp_integration.py
==================
Integration with the Stephanie Planning System (SSP).

This module implements the bidirectional communication bridge between
Jitter and SSP, enabling:
- Context sharing for planning episodes
- Reward signal propagation
- Status reporting
- Feedback integration

Key Features:
- Lightweight context snapshots for SSP
- Bidirectional communication patterns
- Feedback integration for learning
- Configuration validation with Pydantic
- Circuit breaker pattern for resilience
- Comprehensive telemetry and monitoring
- SSP integration hooks
- Performance optimizations
"""
from __future__ import annotations

from typing import Dict, Any, List
import time
import logging
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator
from functools import wraps

log = logging.getLogger("stephanie.jitter.integration.ssp")

class SSPIntegrationConfig(BaseModel):
    """Validated configuration for SSPIntegration"""
    context_refresh_interval: float = Field(10.0, ge=1.0, le=60.0, description="Context refresh interval in seconds")
    enable_reward_shaping: bool = Field(True, description="Enable reward shaping from SSP")
    max_context_history: int = Field(100, ge=10, le=1000, description="Maximum context history size")
    feedback_timeout: float = Field(30.0, ge=5.0, le=120.0, description="Feedback timeout in seconds")
    
    @validator('context_refresh_interval', 'feedback_timeout')
    def validate_time_intervals(cls, v):
        if v <= 0:
            raise ValueError('Time intervals must be positive')
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
    def get_context_snapshot():
        # Context retrieval logic here
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
class ContextSnapshot:
    """Context snapshot for SSP planning episodes"""
    timestamp: float = field(default_factory=time.time)
    jitter_health: float = 0.0
    cognitive_state: Dict[str, Any] = field(default_factory=dict)
    energy_status: Dict[str, float] = field(default_factory=dict)
    crisis_level: float = 0.0
    system_state: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FeedbackRecord:
    """Record of SSP feedback for learning"""
    timestamp: float = field(default_factory=time.time)
    episode_id: str = ""
    feedback_type: str = ""
    feedback_data: Dict[str, Any] = field(default_factory=dict)
    reward: float = 0.0
    success: bool = False

class SSPIntegration:
    """
    Integration with the Stephanie Planning System (SSP).
    
    Key Features:
    - Lightweight context snapshots for SSP
    - Bidirectional communication patterns
    - Feedback integration for learning
    - Configuration validation with Pydantic
    - Circuit breaker pattern for resilience
    - Comprehensive telemetry and monitoring
    - SSP integration hooks
    - Performance optimizations
    """
    
    def __init__(self, jitter_agent, ssp_client=None, cfg: Dict[str, Any] = None):
        try:
            # Validate configuration
            self.config = SSPIntegrationConfig(**(cfg or {}))
            log.info("SSPIntegration configuration validated successfully")
        except Exception as e:
            log.error(f"Configuration validation failed: {str(e)}")
            # Use safe defaults
            self.config = SSPIntegrationConfig()
        
        self.jitter = jitter_agent
        self.ssp_client = ssp_client
        self.integration_enabled = ssp_client is not None
        
        # Initialize history
        self.context_history: List[ContextSnapshot] = []
        self.feedback_history: List[FeedbackRecord] = []
        
        # Initialize performance tracking
        self.processing_times = []
        self.max_processing_history = 100
        
        # Initialize circuit breaker
        self.circuit_breaker = CircuitBreaker()
        
        log.info("SSPIntegration initialized with bidirectional communication")
    
    @CircuitBreaker()
    def get_context_snapshot(self) -> ContextSnapshot:
        """
        Get lightweight context snapshot for SSP planning episodes.
        
        Returns:
            ContextSnapshot object with current system state
        """
        start_time = time.time()
        
        try:
            # Get core system state
            if not self.jitter.core:
                return ContextSnapshot()
            
            # Get latest cognitive state
            latest_state = self.jitter.triune.state_history[-1] if self.jitter.triune.state_history else None
            
            # Create context snapshot
            snapshot = ContextSnapshot(
                jitter_health=self.jitter.homeostasis.get_telemetry().get("health", 0.5),
                cognitive_state={
                    "integrated": latest_state.integrated if latest_state else 0.5,
                    "veto_layer": latest_state.layer_veto if latest_state else "none",
                    "threat_level": latest_state.threat_level if latest_state else 0.5
                },
                energy_status={
                    "metabolic": self.jitter.core.energy.level("metabolic"),
                    "cognitive": self.jitter.core.energy.level("cognitive")
                },
                crisis_level=self.jitter.homeostasis.get_telemetry().get("crisis_level", 0.0),
                system_state={
                    "tick": self.jitter.tick,
                    "generation": getattr(self.jitter.core, 'generation', 0),
                    "alive": self.jitter.running
                }
            )
            
            # Add to history
            self.context_history.append(snapshot)
            if len(self.context_history) > self.config.max_context_history:
                self.context_history.pop(0)
            
            # Log processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > self.max_processing_history:
                self.processing_times.pop(0)
                
            log.debug(f"Generated context snapshot (health={snapshot.jitter_health:.3f})")
            return snapshot
            
        except Exception as e:
            log.error(f"Error generating context snapshot: {str(e)}", exc_info=True)
            # Return minimal snapshot as fallback
            return ContextSnapshot()
    
    def apply_ssp_feedback(self, feedback: Dict[str, Any]):
        """
        Apply feedback from SSP episodes to adjust Jitter behavior.
        
        Args:
            feedback: Dictionary containing SSP feedback metrics
        """
        start_time = time.time()
        
        try:
            # Record feedback
            feedback_record = FeedbackRecord(
                episode_id=feedback.get("episode_id", ""),
                feedback_type=feedback.get("type", "unknown"),
                feedback_data=feedback,
                reward=feedback.get("reward", 0.0),
                success=feedback.get("success", False)
            )
            
            self.feedback_history.append(feedback_record)
            if len(self.feedback_history) > self.config.max_context_history:
                self.feedback_history.pop(0)
            
            # Apply feedback based on type
            if "episode_quality" in feedback:
                quality = feedback["episode_quality"]
                # Adjust attention weights based on task performance
                if quality > 0.7:
                    # Increase cognitive attention for good performance
                    self.jitter.triune.update_attention_weights("primate", 0.05)
                elif quality < 0.3:
                    # Increase reptilian attention for poor performance
                    self.jitter.triune.update_attention_weights("reptilian", 0.05)
            
            # Adjust homeostasis based on task complexity
            if "task_complexity" in feedback:
                complexity = feedback["task_complexity"]
                # Increase cognitive flow setpoint for complex tasks
                self.jitter.homeostasis.set_setpoint("cognitive_flow", 0.5 + complexity * 0.3)
            
            # Log processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > self.max_processing_history:
                self.processing_times.pop(0)
                
            log.debug(f"Applied SSP feedback (type={feedback.get('type', 'unknown')})")
            
        except Exception as e:
            log.error(f"Error applying SSP feedback: {str(e)}", exc_info=True)
    
    def get_ssp_integration_metrics(self) -> Dict[str, float]:
        """
        Get metrics suitable for SSP integration.
        
        Returns:
            Integration metrics in a format SSP can use for reward shaping.
        """
        # Calculate processing time
        avg_processing_time = 0.0
        if self.processing_times:
            avg_processing_time = sum(self.processing_times) / len(self.processing_times)
        
        # Calculate context freshness
        context_freshness = 0.0
        if self.context_history:
            latest_timestamp = self.context_history[-1].timestamp
            context_freshness = time.time() - latest_timestamp
        
        return {
            "context_freshness": context_freshness,
            "processing_efficiency": 1.0 / (1.0 + avg_processing_time * 1000),
            "integration_enabled": float(self.integration_enabled),
            "context_history_size": len(self.context_history),
            "feedback_history_size": len(self.feedback_history)
        }
    
    def get_recent_context(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get recent context snapshots for analysis or reporting"""
        return [
            {
                "timestamp": c.timestamp,
                "jitter_health": c.jitter_health,
                "cognitive_state": c.cognitive_state,
                "energy_status": c.energy_status,
                "crisis_level": c.crisis_level,
                "system_state": c.system_state
            }
            for c in self.context_history[-n:]
        ]
    
    def get_recent_feedback(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get recent feedback records for analysis or reporting"""
        return [
            {
                "timestamp": f.timestamp,
                "episode_id": f.episode_id,
                "feedback_type": f.feedback_type,
                "feedback_data": f.feedback_data,
                "reward": f.reward,
                "success": f.success
            }
            for f in self.feedback_history[-n:]
        ]
    
    def reset(self):
        """Reset integration state"""
        self.context_history.clear()
        self.feedback_history.clear()
        self.processing_times.clear()
        log.info("SSPIntegration reset")