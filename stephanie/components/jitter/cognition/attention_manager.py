# stephanie/components/jitter/cognition/attention_manager.py
"""
attention_manager.py
====================
Dynamic attention weight management for the triune cognitive architecture.

This module implements:
- Adaptive attention allocation based on context and performance
- Reinforcement learning for attention weight optimization
- Crisis-responsive attention reallocation
- Configuration validation with Pydantic
- Circuit breaker pattern for resilience
- Comprehensive telemetry and monitoring
- SSP integration hooks
- Performance optimizations
"""

from __future__ import annotations


from typing import Dict, Any, List, Optional
import numpy as np
import logging
import time
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator
from enum import Enum
from functools import wraps

log = logging.getLogger("stephanie.jitter.cognition.attention")


class AttentionLayer(str, Enum):
    """Cognitive layers that receive attention allocation"""

    REPTILIAN = "reptilian"
    MAMMALIAN = "mammalian"
    PRIMATE = "primate"


class AttentionConfig(BaseModel):
    """Validated configuration for AttentionManager"""

    min_attention: float = Field(
        0.1, ge=0.05, le=0.3, description="Minimum attention allocation"
    )
    max_attention: float = Field(
        0.8, ge=0.6, le=0.95, description="Maximum attention allocation"
    )
    learning_rate: float = Field(
        0.01,
        ge=0.001,
        le=0.05,
        description="Learning rate for attention updates",
    )
    crisis_response_factor: float = Field(
        1.5, ge=1.2, le=2.0, description="Factor for crisis response"
    )
    max_history: int = Field(
        100, ge=50, le=500, description="Maximum history length for learning"
    )

    @validator("min_attention")
    def validate_min_max_attention(cls, v, values):
        if "max_attention" in values and v >= values["max_attention"]:
            raise ValueError("min_attention must be less than max_attention")
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
    def update_attention(layer, delta):
        # Update logic here
        pass
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_attempts: int = 3,
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
                if (
                    time.time() - self.last_failure_time
                    > self.recovery_timeout
                ):
                    self.logger.info(
                        "Circuit breaker transitioning to HALF_OPEN state"
                    )
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.half_open_successes = 0
                else:
                    self.logger.warning(
                        "Circuit breaker is OPEN - skipping call"
                    )
                    return None

            try:
                # Call the function
                result = func(*args, **kwargs)

                # Reset failures if successful
                if self.state == CircuitBreakerState.HALF_OPEN:
                    self.half_open_successes += 1
                    if self.half_open_successes >= self.half_open_attempts:
                        self.logger.info(
                            "Circuit breaker transitioning to CLOSED state"
                        )
                        self.state = CircuitBreakerState.CLOSED
                        self.failures = 0
                        self.half_open_successes = 0

                return result

            except Exception as e:
                # Record failure
                self.failures += 1
                self.last_failure_time = time.time()
                self.logger.error(
                    f"Service failure: {str(e)}, failures: {self.failures}"
                )

                # Transition to OPEN state if threshold reached
                if self.failures >= self.failure_threshold:
                    self.logger.warning(
                        "Circuit breaker transitioning to OPEN state"
                    )
                    self.state = CircuitBreakerState.OPEN

                raise

        return wrapper

    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state for monitoring"""
        return {
            "state": self.state,
            "failures": self.failures,
            "last_failure_time": self.last_failure_time,
            "half_open_successes": self.half_open_successes,
        }


@dataclass
class AttentionRecord:
    """Record of an attention allocation decision for learning"""

    id: str = field(
        default_factory=lambda: f"attn_{int(time.time())}_{np.random.randint(1000)}"
    )
    timestamp: float = field(default_factory=time.time)
    attention_weights: Dict[str, float] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    outcome: Dict[str, Any] = field(default_factory=dict)
    reward: float = 0.0
    success: bool = False


@dataclass
class AttentionMetrics:
    """Metrics for attention management performance"""

    adaptation_rate: float = 0.0
    stability: float = 0.5
    crisis_response_count: int = 0
    learning_efficiency: float = 0.5
    processing_time_ms: float = 0.0


class AttentionManager:
    """
    Manages dynamic attention allocation for the triune cognitive architecture.

    Key Features:
    - Adaptive attention allocation based on context and performance
    - Reinforcement learning for attention weight optimization
    - Crisis-responsive attention reallocation
    - Configuration validation with Pydantic
    - Circuit breaker pattern for resilience
    - Comprehensive telemetry and monitoring
    - SSP integration hooks
    - Performance optimizations
    """

    def __init__(self, cfg: Dict[str, Any], initial_weights: Dict[str, float]):
        try:
            # Validate configuration
            self.config = AttentionConfig(**cfg)
            log.info("AttentionManager configuration validated successfully")
        except Exception as e:
            log.error(f"Configuration validation failed: {str(e)}")
            # Use safe defaults
            self.config = AttentionConfig()

        # Initialize attention weights
        self.attention_weights = initial_weights.copy()

        # Validate initial weights
        self._validate_weights()

        # Initialize history
        self.attention_history: List[AttentionRecord] = []

        # Initialize circuit breaker
        self.circuit_breaker = CircuitBreaker()

        # Initialize metrics
        self.metrics = AttentionMetrics()

        # Initialize performance tracking
        self.processing_times = []
        self.max_processing_history = 100

        log.info(
            "AttentionManager initialized with dynamic allocation features"
        )

    @CircuitBreaker()
    def update_attention(
        self,
        layer: str,
        delta: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Update attention weights based on performance or external signals.

        Args:
            layer: Which layer to adjust (reptilian, mammalian, primate)
            delta: Change amount (-1.0 to 1.0)
            context: Additional context for the update

        Returns:
            Updated attention weights
        """
        start_time = time.time()

        try:
            # Validate layer
            if layer not in self.attention_weights:
                log.warning(f"Invalid layer for attention update: {layer}")
                return self.attention_weights.copy()

            # Store current state for history
            old_weights = self.attention_weights.copy()

            # Apply delta with bounds checking
            self.attention_weights[layer] = max(
                self.config.min_attention,
                min(
                    self.config.max_attention,
                    self.attention_weights[layer] + delta,
                ),
            )

            # Normalize to sum to 1
            total = sum(self.attention_weights.values())
            for l in self.attention_weights:
                self.attention_weights[l] /= total

            # Record for learning
            self._record_attention_update(old_weights, context or {}, delta)

            # Update metrics
            self._update_metrics(old_weights, delta)

            # Log processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > self.max_processing_history:
                self.processing_times.pop(0)

            log.debug(
                f"Updated attention for {layer} (delta={delta:.3f}): {self.attention_weights}"
            )
            return self.attention_weights.copy()

        except Exception as e:
            log.error(f"Error updating attention: {str(e)}", exc_info=True)
            return self.attention_weights.copy()

    def _validate_weights(self):
        """Validate that attention weights are within bounds and sum to 1"""
        # Check bounds
        for layer, weight in self.attention_weights.items():
            if (
                weight < self.config.min_attention
                or weight > self.config.max_attention
            ):
                log.warning(
                    f"Attention weight for {layer} ({weight}) outside bounds, clamping"
                )
                self.attention_weights[layer] = max(
                    self.config.min_attention,
                    min(self.config.max_attention, weight),
                )

        # Check sum
        total = sum(self.attention_weights.values())
        if abs(total - 1.0) > 0.01:
            log.warning(
                f"Attention weights don't sum to 1 ({total}), normalizing"
            )
            for layer in self.attention_weights:
                self.attention_weights[layer] /= total

    def _record_attention_update(
        self,
        old_weights: Dict[str, float],
        context: Dict[str, Any],
        delta: float,
    ):
        """Record attention update for learning and adaptation"""
        # Calculate outcome (simplified - would be more sophisticated in production)
        outcome = {"delta": delta, "context": context}

        # Create record
        record = AttentionRecord(
            attention_weights=self.attention_weights.copy(),
            context=context,
            outcome=outcome,
            reward=0.0,  # Will be updated later
            success=False,  # Will be updated later
        )

        # Add to history
        self.attention_history.append(record)

        # Keep history bounded
        if len(self.attention_history) > self.config.max_history:
            self.attention_history.pop(0)

    def _update_metrics(self, old_weights: Dict[str, float], delta: float):
        """Update attention management metrics based on update"""
        # Calculate adaptation rate
        changes = [
            abs(self.attention_weights[l] - old_weights[l])
            for l in self.attention_weights
        ]
        self.metrics.adaptation_rate = np.mean(changes)

        # Update stability (how much weights change over time)
        if len(self.attention_history) >= 10:
            recent_changes = [
                abs(
                    self.attention_history[-i].attention_weights[l]
                    - self.attention_history[-i - 1].attention_weights[l]
                )
                for i in range(1, min(10, len(self.attention_history)))
                for l in self.attention_weights
            ]
            self.metrics.stability = 1.0 - np.mean(recent_changes)

        # Update processing time
        avg_processing_time = (
            np.mean(self.processing_times) if self.processing_times else 0.0
        )
        self.metrics.processing_time_ms = avg_processing_time * 1000

    def get_attention_metrics(self) -> Dict[str, Any]:
        """Get current attention metrics for monitoring and adaptation"""
        return {
            "adaptation_rate": self.metrics.adaptation_rate,
            "stability": self.metrics.stability,
            "crisis_response_count": self.metrics.crisis_response_count,
            "learning_efficiency": self.metrics.learning_efficiency,
            "processing_time_ms": self.metrics.processing_time_ms,
            "attention_history_size": len(self.attention_history),
            "circuit_breaker_state": self.circuit_breaker.get_state(),
        }

    def apply_crisis_response(self, crisis_level: float) -> Dict[str, float]:
        """
        Apply crisis response by reallocating attention.

        Args:
            crisis_level: Current crisis level (0-1)

        Returns:
            Updated attention weights
        """
        start_time = time.time()

        try:
            # Store current state
            old_weights = self.attention_weights.copy()

            # Increase reptilian attention for stability
            reptilian_delta = (
                crisis_level * self.config.crisis_response_factor * 0.1
            )
            self.attention_weights[AttentionLayer.REPTILIAN.value] = min(
                self.config.max_attention,
                self.attention_weights[AttentionLayer.REPTILIAN.value]
                + reptilian_delta,
            )

            # Decrease primate attention (less abstract reasoning during crisis)
            primate_delta = (
                -crisis_level * self.config.crisis_response_factor * 0.1
            )
            self.attention_weights[AttentionLayer.PRIMATE.value] = max(
                self.config.min_attention,
                self.attention_weights[AttentionLayer.PRIMATE.value]
                + primate_delta,
            )

            # Normalize to sum to 1
            total = sum(self.attention_weights.values())
            for l in self.attention_weights:
                self.attention_weights[l] /= total

            # Record for learning
            self._record_attention_update(
                old_weights, {"crisis_level": crisis_level}, reptilian_delta
            )

            # Update metrics
            self.metrics.crisis_response_count += 1

            # Log processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > self.max_processing_history:
                self.processing_times.pop(0)

            log.info(
                f"Applied crisis response (level={crisis_level:.3f}): {self.attention_weights}"
            )
            return self.attention_weights.copy()

        except Exception as e:
            log.error(
                f"Error applying crisis response: {str(e)}", exc_info=True
            )
            return self.attention_weights.copy()

    def get_ssp_integration_metrics(self) -> Dict[str, float]:
        """
        Get metrics suitable for SSP integration.

        Returns attention metrics in a format SSP can use for reward shaping.
        """
        metrics = self.get_attention_metrics()

        return {
            "adaptation_rate": metrics["adaptation_rate"],
            "attention_stability": metrics["stability"],
            "crisis_response_count": metrics["crisis_response_count"],
            "learning_efficiency": metrics["learning_efficiency"],
            "processing_efficiency": 1.0
            / (1.0 + metrics["processing_time_ms"]),
            "attention_reptilian": self.attention_weights[
                AttentionLayer.REPTILIAN.value
            ],
            "attention_mammalian": self.attention_weights[
                AttentionLayer.MAMMALIAN.value
            ],
            "attention_primate": self.attention_weights[
                AttentionLayer.PRIMATE.value
            ],
        }

    def evaluate_update_success(
        self,
        attention_id: str,
        performance_before: float,
        performance_after: float,
    ) -> bool:
        """
        Evaluate whether an attention update was successful.

        Args:
            attention_id: ID of the attention update to evaluate
            performance_before: Performance before update
            performance_after: Performance after update

        Returns:
            True if update was successful, False otherwise
        """
        # Find the attention record
        record = next(
            (a for a in self.attention_history if a.id == attention_id), None
        )
        if not record:
            log.warning(f"Attention record not found: {attention_id}")
            return False

        # Determine success based on performance improvement
        success = performance_after > performance_before
        reward = performance_after - performance_before

        # Update record
        record.reward = reward
        record.success = success

        # Update learning efficiency
        if success:
            self.metrics.learning_efficiency = min(
                1.0, self.metrics.learning_efficiency + 0.01
            )
        else:
            self.metrics.learning_efficiency = max(
                0.0, self.metrics.learning_efficiency - 0.01
            )

        log.debug(
            f"Attention update evaluation: {attention_id} - success={success}, reward={reward:.3f}"
        )
        return success

    def get_recent_updates(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get recent attention updates for analysis or reporting"""
        return [
            {
                "id": a.id,
                "timestamp": a.timestamp,
                "attention_weights": a.attention_weights,
                "context": a.context,
                "outcome": a.outcome,
                "reward": a.reward,
                "success": a.success,
            }
            for a in self.attention_history[-n:]
        ]

    def get_optimal_attention_profile(
        self, context: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Get optimal attention profile for given context based on historical data.

        Args:
            context: Current context for which to determine optimal profile

        Returns:
            Optimal attention weights for the context
        """
        if len(self.attention_history) < 10:
            return self.attention_weights.copy()

        # Find similar contexts
        similar = []
        for record in self.attention_history:
            # Simple similarity measure (would be more sophisticated in production)
            context_match = 0.0
            for key in context:
                if key in record.context:
                    # Numeric context
                    if isinstance(context[key], (int, float)):
                        context_match += 1.0 - min(
                            1.0, abs(context[key] - record.context[key])
                        )
                    # Categorical context
                    else:
                        context_match += (
                            1.0 if context[key] == record.context[key] else 0.0
                        )

            # Normalize
            context_match = context_match / max(1, len(context))
            if context_match > 0.5:  # Minimum similarity
                similar.append((record, context_match))

        # If no similar contexts, return current weights
        if not similar:
            return self.attention_weights.copy()

        # Calculate weighted average of attention weights
        total_weight = 0.0
        avg_weights = {l: 0.0 for l in self.attention_weights}

        for record, weight in similar:
            total_weight += weight
            for layer in avg_weights:
                avg_weights[layer] += (
                    record.attention_weights.get(layer, 0.0) * weight
                )

        # Normalize
        if total_weight > 0:
            for layer in avg_weights:
                avg_weights[layer] /= total_weight

        # Ensure weights sum to 1 and are within bounds
        total = sum(avg_weights.values())
        if total > 0:
            for layer in avg_weights:
                avg_weights[layer] /= total

        for layer in avg_weights:
            avg_weights[layer] = max(
                self.config.min_attention,
                min(self.config.max_attention, avg_weights[layer]),
            )

        return avg_weights
