# stephanie/components/jitter/lifecycle/integration/reward_shaper.py
"""
reward_shaper.py
================
Reward shaping system for learning and adaptation in the Jitter Autopoietic System.

This module implements the reward shaping mechanism that:
- Transforms system performance into meaningful rewards
- Integrates with SSP feedback for learning
- Enables reinforcement learning in Jitter's cognitive system
- Provides adaptive reward scaling based on system state
- Supports hierarchical reward structures

Key Features:
- Multi-dimensional reward calculation
- Hierarchical reward structure
- Adaptive reward scaling
- Feedback integration from SSP
- Configuration validation with Pydantic
- Circuit breaker pattern for resilience
- Comprehensive telemetry and monitoring
- SSP integration hooks
- Performance optimizations
"""
from __future__ import annotations

from typing import Dict, Any, List, Optional
import time
import logging
import numpy as np
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator
from enum import Enum
from functools import wraps

log = logging.getLogger("stephanie.jitter.integration.reward")

class RewardConfig(BaseModel):
    """Validated configuration for RewardShaper"""
    base_reward_weight: float = Field(0.3, ge=0.0, le=1.0, description="Weight for base rewards")
    health_reward_weight: float = Field(0.25, ge=0.0, le=1.0, description="Weight for health rewards")
    energy_reward_weight: float = Field(0.25, ge=0.0, le=1.0, description="Weight for energy rewards")
    boundary_reward_weight: float = Field(0.2, ge=0.0, le=1.0, description="Weight for boundary rewards")
    crisis_penalty_weight: float = Field(0.1, ge=0.0, le=1.0, description="Weight for crisis penalties")
    learning_rate: float = Field(0.01, ge=0.001, le=0.1, description="Learning rate for reward adaptation")
    max_history: int = Field(1000, ge=100, le=10000, description="Maximum history length")
    
    @validator('base_reward_weight', 'health_reward_weight', 'energy_reward_weight', 
              'boundary_reward_weight', 'crisis_penalty_weight')
    def validate_weights_sum_to_one(cls, v, values):
        weights = [
            values.get('base_reward_weight', 0.3),
            values.get('health_reward_weight', 0.25),
            values.get('energy_reward_weight', 0.25),
            values.get('boundary_reward_weight', 0.2),
            values.get('crisis_penalty_weight', 0.1)
        ]
        if abs(sum(weights) - 1.0) > 0.01:
            raise ValueError('Weights must sum to approximately 1.0')
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
    def calculate_reward(feedback):
        # Reward calculation logic here
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
class RewardRecord:
    """Record of reward calculation for learning and analysis"""
    timestamp: float = field(default_factory=time.time)
    base_reward: float = 0.0
    health_reward: float = 0.0
    energy_reward: float = 0.0
    boundary_reward: float = 0.0
    crisis_penalty: float = 0.0
    total_reward: float = 0.0
    feedback: Dict[str, Any] = field(default_factory=dict)
    success: bool = False

@dataclass
class RewardMetrics:
    """Metrics for reward shaping performance"""
    reward_efficiency: float = 0.0
    reward_trend: float = 0.0
    learning_rate: float = 0.0
    adaptation_rate: float = 0.0
    processing_time_ms: float = 0.0
    history_size: int = 0

class RewardShaper:
    """
    Reward shaping system for learning and adaptation in the Jitter Autopoietic System.
    
    Key Features:
    - Multi-dimensional reward calculation
    - Hierarchical reward structure
    - Adaptive reward scaling
    - Feedback integration from SSP
    - Configuration validation with Pydantic
    - Circuit breaker pattern for resilience
    - Comprehensive telemetry and monitoring
    - SSP integration hooks
    - Performance optimizations
    """
    
    def __init__(self, jitter_agent, cfg: Dict[str, Any] = None):
        try:
            # Validate configuration
            self.config = RewardConfig(**(cfg or {}))
            log.info("RewardShaper configuration validated successfully")
        except Exception as e:
            log.error(f"Configuration validation failed: {str(e)}")
            # Use safe defaults
            self.config = RewardConfig()
        
        self.jitter = jitter_agent
        
        # Initialize history
        self.reward_history: List[RewardRecord] = []
        
        # Initialize metrics
        self.metrics = RewardMetrics()
        
        # Initialize performance tracking
        self.processing_times = []
        self.max_processing_history = 100
        
        # Initialize circuit breaker
        self.circuit_breaker = CircuitBreaker()
        
        # Initialize adaptive parameters
        self.adaptive_weights = {
            "health_reward_weight": self.config.health_reward_weight,
            "energy_reward_weight": self.config.energy_reward_weight,
            "boundary_reward_weight": self.config.boundary_reward_weight,
            "crisis_penalty_weight": self.config.crisis_penalty_weight
        }
        
        log.info("RewardShaper initialized with adaptive reward system")
    
    @CircuitBreaker()
    def calculate_reward(self, feedback: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculate reward based on system performance and feedback.
        
        Args:
            feedback: Optional feedback from SSP or other systems
            
        Returns:
            Reward value (0-1)
        """
        start_time = time.time()
        
        try:
            # Get current system state
            if not self.jitter.core:
                return 0.0
            
            # Base reward (system stability)
            base_reward = 0.5  # Neutral base reward
            
            # Health-based reward
            health = self.jitter.homeostasis.get_telemetry().get("health", 0.5)
            health_reward = health * self.config.health_reward_weight
            
            # Energy-based reward
            energy_balance = (
                self.jitter.core.energy.level("metabolic") + 
                self.jitter.core.energy.level("cognitive") + 
                self.jitter.core.energy.level("reserve")
            ) / 3.0
            energy_reward = min(1.0, energy_balance / 100.0) * self.config.energy_reward_weight
            
            # Boundary integrity reward
            boundary = self.jitter.core.membrane.integrity
            boundary_reward = boundary * self.config.boundary_reward_weight
            
            # Crisis penalty
            crisis_level = self.jitter.homeostasis.get_telemetry().get("crisis_level", 0.0)
            crisis_penalty = crisis_level * self.config.crisis_penalty_weight
            
            # Calculate total reward
            total_reward = (
                base_reward * self.config.base_reward_weight +
                health_reward +
                energy_reward +
                boundary_reward -
                crisis_penalty
            )
            
            # Normalize to 0-1 range
            total_reward = max(0.0, min(1.0, total_reward))
            
            # Apply feedback adjustments if available
            if feedback:
                total_reward = self._apply_feedback_adjustments(total_reward, feedback)
            
            # Record reward
            reward_record = RewardRecord(
                base_reward=base_reward,
                health_reward=health_reward,
                energy_reward=energy_reward,
                boundary_reward=boundary_reward,
                crisis_penalty=crisis_penalty,
                total_reward=total_reward,
                feedback=feedback or {},
                success=True
            )
            
            self.reward_history.append(reward_record)
            if len(self.reward_history) > self.config.max_history:
                self.reward_history.pop(0)
            
            # Update metrics
            self._update_metrics(total_reward)
            
            # Apply adaptive learning
            self._adapt_rewards(feedback)
            
            # Log processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > self.max_processing_history:
                self.processing_times.pop(0)
                
            log.debug(f"Calculated reward (total={total_reward:.3f}, health={health:.3f})")
            return total_reward
            
        except Exception as e:
            log.error(f"Error calculating reward: {str(e)}", exc_info=True)
            return 0.0
    
    def _apply_feedback_adjustments(self, base_reward: float, feedback: Dict[str, Any]) -> float:
        """Apply adjustments based on feedback from SSP or other systems"""
        # Adjust reward based on episode quality
        if "episode_quality" in feedback:
            quality = feedback["episode_quality"]
            # Scale reward based on quality
            quality_factor = min(1.0, max(0.1, quality))  # Clamp between 0.1 and 1.0
            base_reward *= quality_factor
        
        # Adjust reward based on task difficulty
        if "task_difficulty" in feedback:
            difficulty = feedback["task_difficulty"]
            # Make reward more challenging for difficult tasks
            difficulty_factor = 1.0 + (difficulty * 0.2)  # Add up to 20% bonus
            base_reward *= difficulty_factor
        
        # Apply any direct reward adjustments
        if "direct_reward_adjustment" in feedback:
            adjustment = feedback["direct_reward_adjustment"]
            base_reward = max(0.0, min(1.0, base_reward + adjustment))
        
        return base_reward
    
    def _update_metrics(self, reward: float):
        """Update reward shaping metrics based on recent rewards"""
        # Update reward efficiency
        if len(self.reward_history) >= 10:
            recent_rewards = [r.total_reward for r in self.reward_history[-10:]]
            self.metrics.reward_efficiency = np.mean(recent_rewards)
        
        # Update reward trend
        if len(self.reward_history) >= 5:
            recent_rewards = [r.total_reward for r in self.reward_history[-5:]]
            if len(recent_rewards) > 1:
                self.metrics.reward_trend = recent_rewards[-1] - recent_rewards[0]
        
        # Update processing time
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0.0
        self.metrics.processing_time_ms = avg_processing_time * 1000
        self.metrics.history_size = len(self.reward_history)
    
    def _adapt_rewards(self, feedback: Optional[Dict[str, Any]] = None):
        """Adapt reward weights based on performance and feedback"""
        # Only adapt if we have enough history
        if len(self.reward_history) < 20:
            return
        
        # Calculate recent performance trends
        recent_rewards = [r.total_reward for r in self.reward_history[-20:]]
        recent_performance = np.mean(recent_rewards)
        
        # Adjust weights based on performance
        if recent_performance > 0.7:
            # Good performance - increase weights for positive factors
            self.adaptive_weights["health_reward_weight"] = min(
                1.0, 
                self.adaptive_weights["health_reward_weight"] * 1.05
            )
            self.adaptive_weights["energy_reward_weight"] = min(
                1.0, 
                self.adaptive_weights["energy_reward_weight"] * 1.05
            )
            self.adaptive_weights["boundary_reward_weight"] = min(
                1.0, 
                self.adaptive_weights["boundary_reward_weight"] * 1.05
            )
        elif recent_performance < 0.3:
            # Poor performance - reduce weights for negative factors
            self.adaptive_weights["crisis_penalty_weight"] = min(
                1.0, 
                self.adaptive_weights["crisis_penalty_weight"] * 1.05
            )
        
        # Apply learning rate to smooth adaptation
        for key in self.adaptive_weights:
            self.adaptive_weights[key] = (
                self.adaptive_weights[key] * (1.0 - self.config.learning_rate) +
                self.config.__dict__[key] * self.config.learning_rate
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current reward shaping metrics for monitoring and adaptation"""
        return {
            "reward_efficiency": self.metrics.reward_efficiency,
            "reward_trend": self.metrics.reward_trend,
            "learning_rate": self.metrics.learning_rate,
            "adaptation_rate": self.metrics.adaptation_rate,
            "processing_time_ms": self.metrics.processing_time_ms,
            "history_size": self.metrics.history_size,
            "circuit_breaker_state": self.circuit_breaker.get_state(),
            "adaptive_weights": self.adaptive_weights
        }
    
    def get_reward_history(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get recent reward calculations for analysis or reporting"""
        return [
            {
                "timestamp": r.timestamp,
                "base_reward": r.base_reward,
                "health_reward": r.health_reward,
                "energy_reward": r.energy_reward,
                "boundary_reward": r.boundary_reward,
                "crisis_penalty": r.crisis_penalty,
                "total_reward": r.total_reward,
                "feedback": r.feedback,
                "success": r.success
            }
            for r in self.reward_history[-n:]
        ]
    
    def get_adaptive_weights(self) -> Dict[str, float]:
        """Get current adaptive reward weights"""
        return self.adaptive_weights.copy()
    
    def get_ssp_integration_metrics(self) -> Dict[str, float]:
        """
        Get metrics suitable for SSP integration.
        
        Returns:
            Reward shaping metrics in a format SSP can use for reward shaping.
        """
        metrics = self.get_metrics()
        
        return {
            "reward_efficiency": metrics["reward_efficiency"],
            "reward_trend": metrics["reward_trend"],
            "learning_rate": metrics["learning_rate"],
            "adaptation_rate": metrics["adaptation_rate"],
            "processing_efficiency": 1.0 / (1.0 + metrics["processing_time_ms"]),
            "adaptive_weights": metrics["adaptive_weights"]
        }
    
    def reset(self):
        """Reset reward shaping state"""
        self.reward_history.clear()
        self.metrics = RewardMetrics()
        self.processing_times.clear()
        self.adaptive_weights = {
            "health_reward_weight": self.config.health_reward_weight,
            "energy_reward_weight": self.config.energy_reward_weight,
            "boundary_reward_weight": self.config.boundary_reward_weight,
            "crisis_penalty_weight": self.config.crisis_penalty_weight
        }
        log.info("RewardShaper reset")