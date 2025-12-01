# stephanie/components/jitter/cognition/sense_making.py
"""
sense_making.py
===============
Implementation of cognition as sense-making rather than information processing.

This module implements the enactive approach to cognition where meaning is created
through the organism's interactions with its environment, not through representation.

Key Features:
- Meaning arises from sensorimotor loops, not internal representations
- Cognitive structures are shaped by patterns of interaction
- Learning happens through successful interactions (not just error correction)
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
from typing import Any, Dict, List, Tuple

import numpy as np
from pydantic import BaseModel, Field, validator

from stephanie.utils.similarity_utils import cosine

log = logging.getLogger("stephanie.jitter.cognition.sense_making")

class MeaningType(str, Enum):
    """Types of meaning created through interaction"""
    AFFORDANCE = "affordance"  # Environment offers opportunity
    CONSTRAINT = "constraint"  # Environment imposes limitation
    NEUTRAL = "neutral"        # No strong meaning
    ERROR = "error"            # Error in meaning creation

class SenseMakingConfig(BaseModel):
    """Validated configuration for SenseMakingEngine"""
    max_buffer_size: int = Field(100, ge=10, le=500, description="Sensory/action buffer size")
    meaning_threshold: float = Field(0.6, ge=0.3, le=0.9, description="Threshold for strong meaning")
    resonance_decay: float = Field(0.95, ge=0.8, le=0.99, description="Decay factor for resonance")
    min_interaction_similarity: float = Field(0.3, ge=0.1, le=0.5, description="Min similarity threshold")
    max_history: int = Field(1000, ge=100, le=10000, description="Maximum history length")
    familiarity_weight: float = Field(0.4, ge=0.2, le=0.6, description="Weight for familiarity")
    outcome_weight: float = Field(0.6, ge=0.4, le=0.8, description="Weight for outcome quality")
    
    def validate_min_max_similarity(cls, v, values):
        if 'meaning_threshold' in values and v >= values['meaning_threshold']:
            raise ValueError('min_interaction_similarity must be less than meaning_threshold')
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
    def process_interaction(sensory_input, action, outcome):
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
class InteractionRecord:
    """Record of an interaction cycle for learning and adaptation"""
    id: str = field(default_factory=lambda: f"interact_{int(time.time())}_{np.random.randint(1000)}")
    timestamp: float = field(default_factory=time.time)
    sensory_input: Dict[str, Any] = field(default_factory=dict)
    action: Dict[str, Any] = field(default_factory=dict)
    outcome: Dict[str, Any] = field(default_factory=dict)
    meaning: Dict[str, Any] = field(default_factory=dict)
    similarity_score: float = 0.0
    success: bool = False

@dataclass
class MeaningMetrics:
    """Metrics for sense-making performance"""
    meaning_stability: float = 0.5
    resonance_score: float = 0.5
    familiarity_rate: float = 0.5
    affordance_rate: float = 0.3
    constraint_rate: float = 0.3
    neutral_rate: float = 0.4
    processing_time_ms: float = 0.0

class SenseMakingEngine:
    """
    Engine for creating meaning through interaction with the environment with enhanced features.
    
    Key Features:
    - Meaning arises from sensorimotor loops, not internal representations
    - Cognitive structures are shaped by patterns of interaction
    - Learning happens through successful interactions (not just error correction)
    - Configuration validation with Pydantic
    - Circuit breaker pattern for resilience
    - Comprehensive telemetry and monitoring
    - SSP integration hooks
    - Performance optimizations
    """
    
    def __init__(self, cfg: Dict[str, Any]):
        try:
            # Validate configuration
            self.config = SenseMakingConfig(**cfg)
            log.info("SenseMakingEngine configuration validated successfully")
        except Exception as e:
            log.error(f"Configuration validation failed: {str(e)}")
            # Use safe defaults
            self.config = SenseMakingConfig()
        
        # Initialize buffers
        self.sensory_buffer = []
        self.action_buffer = []
        
        # Initialize history
        self.interaction_history: List[InteractionRecord] = []
        
        # Initialize circuit breaker
        self.circuit_breaker = CircuitBreaker()
        
        # Initialize metrics
        self.metrics = MeaningMetrics()
        
        # Initialize performance tracking
        self.processing_times = []
        self.max_processing_history = 100
        
        log.info("SenseMakingEngine initialized with enactive cognition features")
    
    @CircuitBreaker()
    def process_interaction(
        self, 
        sensory_input: Dict[str, Any], 
        action: Dict[str, Any],
        outcome: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process an interaction cycle and create meaning.
        
        Args:
            sensory_input: What the system perceived
            action: What the system did in response
            outcome: Result of the action
            
        Returns:
            Dictionary containing cognitive outputs:
            - meaning: How the system interprets this pattern
            - familiarity: How familiar this pattern is (0-1)
            - resonance: How well this pattern fits with existing meaning (0-1)
        """
        start_time = time.time()
        
        try:
            # Store in buffers
            self._store_in_buffers(sensory_input, action)
            
            # Find similar past interactions
            similar_interactions = self._find_similar_interactions(sensory_input, action)
            
            # Create meaning from the interaction pattern
            meaning = self._create_meaning(sensory_input, action, outcome, similar_interactions)
            
            # Record for learning
            self._record_interaction(sensory_input, action, outcome, meaning, similar_interactions)
            
            # Update metrics
            self._update_metrics(meaning, similar_interactions)
            
            # Log processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > self.max_processing_history:
                self.processing_times.pop(0)
                
            log.debug(f"Processed interaction (type={meaning['meaning_type']}, "
                     f"resonance={meaning['resonance']:.3f}, "
                     f"familiarity={meaning['familiarity']:.3f})")
            return meaning
            
        except Exception as e:
            log.error(f"Error processing interaction: {str(e)}", exc_info=True)
            # Return safe fallback meaning
            return {
                "meaning_type": MeaningType.ERROR.value,
                "outcome_quality": 0.0,
                "resonance": 0.0,
                "familiarity": 0.0,
                "similar_count": 0
            }
    
    def _store_in_buffers(self, sensory_input: Dict[str, Any], action: Dict[str, Any]):
        """Store interaction in buffers for short-term processing"""
        self.sensory_buffer.append(sensory_input)
        self.action_buffer.append(action)
        
        # Keep buffers bounded
        if len(self.sensory_buffer) > self.config.max_buffer_size:
            self.sensory_buffer.pop(0)
        if len(self.action_buffer) > self.config.max_buffer_size:
            self.action_buffer.pop(0)
    
    def _find_similar_interactions(
        self, 
        sensory_input: Dict[str, Any], 
        action: Dict[str, Any]
    ) -> List[Tuple[InteractionRecord, float]]:
        """Find similar past interactions and their outcomes"""
        similar = []
        
        for interaction in self.interaction_history:
            # Calculate similarity
            sensory_sim = self._calculate_sensory_similarity(
                sensory_input, 
                interaction.sensory_input
            )
            action_sim = self._calculate_action_similarity(
                action, 
                interaction.action
            )
            similarity = 0.7 * sensory_sim + 0.3 * action_sim
            
            if similarity > self.config.min_interaction_similarity:  # Minimum similarity threshold
                similar.append((interaction, similarity))
        
        # Sort by similarity
        similar.sort(key=lambda x: x[1], reverse=True)
        return similar[:5]  # Return top 5 most similar
    
    def _calculate_sensory_similarity(
        self, 
        sensory_a: Dict[str, Any], 
        sensory_b: Dict[str, Any]
    ) -> float:
        """Calculate similarity between two sensory inputs"""
        # For VPM embeddings
        if "vpm_embedding" in sensory_a and "vpm_embedding" in sensory_b:
            return cosine(
                sensory_a["vpm_embedding"], 
                sensory_b["vpm_embedding"]
            )
        
        # For other sensory types
        return 0.5
    
    def _calculate_action_similarity(
        self, 
        action_a: Dict[str, Any], 
        action_b: Dict[str, Any]
    ) -> float:
        """Calculate similarity between two actions"""
        # Simple implementation - would be more sophisticated in production
        if action_a.get("type") == action_b.get("type"):
            return 0.8  # High similarity if same action type
        return 0.2  # Low similarity if different action types
    
    def _create_meaning(
        self, 
        sensory_input: Dict[str, Any], 
        action: Dict[str, Any],
        outcome: Dict[str, Any],
        similar_interactions: List[Tuple[InteractionRecord, float]]
    ) -> Dict[str, Any]:
        """Create meaning from the interaction pattern"""
        # Base meaning on outcome quality
        outcome_quality = outcome.get("quality", 0.5)
        
        # Incorporate resonance with existing meaning
        resonance = 0.0
        if similar_interactions:
            # Weighted average of resonance scores
            total_weight = 0.0
            for interaction, similarity in similar_interactions:
                weight = similarity * interaction.meaning.get("resonance", 0.5)
                resonance += weight
                total_weight += similarity
            
            if total_weight > 0:
                resonance = resonance / total_weight
            else:
                resonance = 0.5
        
        # Update resonance with decay for stability
        resonance = resonance * self.config.resonance_decay + outcome_quality * (1.0 - self.config.resonance_decay)
        
        # Determine meaning based on outcome and resonance
        if outcome_quality > self.config.meaning_threshold and resonance > self.config.meaning_threshold:
            meaning_type = MeaningType.AFFORDANCE.value  # Environment offers opportunity
        elif outcome_quality < (1.0 - self.config.meaning_threshold) and resonance > self.config.meaning_threshold:
            meaning_type = MeaningType.CONSTRAINT.value  # Environment imposes limitation
        else:
            meaning_type = MeaningType.NEUTRAL.value  # No strong meaning yet
        
        return {
            "meaning_type": meaning_type,
            "outcome_quality": outcome_quality,
            "resonance": resonance,
            "familiarity": len(similar_interactions) / 10.0,  # Scale to 0-1
            "similar_count": len(similar_interactions)
        }
    
    def _record_interaction(
        self, 
        sensory_input: Dict[str, Any], 
        action: Dict[str, Any],
        outcome: Dict[str, Any],
        meaning: Dict[str, Any],
        similar_interactions: List[Tuple[InteractionRecord, float]]
    ):
        """Record interaction for future learning"""
        # Calculate similarity to most similar interaction
        similarity_score = similar_interactions[0][1] if similar_interactions else 0.0
        
        # Determine success
        success = outcome.get("quality", 0.5) > 0.5
        
        # Create record
        record = InteractionRecord(
            sensory_input={k: v for k, v in sensory_input.items() if k != "vpm_embedding"},
            action=action,
            outcome=outcome,
            meaning=meaning,
            similarity_score=similarity_score,
            success=success
        )
        
        # Add to history
        self.interaction_history.append(record)
        
        # Keep history bounded
        if len(self.interaction_history) > self.config.max_history:
            self.interaction_history.pop(0)
    
    def _update_metrics(self, meaning: Dict[str, Any], similar_interactions: List[Tuple[InteractionRecord, float]]):
        """Update sense-making metrics based on interaction"""
        # Update meaning stability
        if len(self.interaction_history) >= 10:
            # Check consistency of meaning for similar interaction patterns
            stable_count = 0
            total_count = 0
            
            for i in range(len(self.interaction_history) - 5):
                base = self.interaction_history[i]
                for j in range(i+1, min(i+6, len(self.interaction_history))):
                    comparison = self.interaction_history[j]
                    
                    # Calculate interaction similarity
                    sensory_sim = self._calculate_sensory_similarity(
                        base.sensory_input, 
                        comparison.sensory_input
                    )
                    action_sim = self._calculate_action_similarity(
                        base.action, 
                        comparison.action
                    )
                    similarity = 0.7 * sensory_sim + 0.3 * action_sim
                    
                    if similarity > 0.6:  # Significant similarity
                        total_count += 1
                        # Check if meaning types match
                        if base.meaning["meaning_type"] == comparison.meaning["meaning_type"]:
                            stable_count += 1
            
            self.metrics.meaning_stability = stable_count / total_count if total_count > 0 else 0.5
        
        # Update resonance score
        self.metrics.resonance_score = (
            self.metrics.resonance_score * 0.9 + 
            meaning["resonance"] * 0.1
        )
        
        # Update familiarity rate
        self.metrics.familiarity_rate = (
            self.metrics.familiarity_rate * 0.95 + 
            meaning["familiarity"] * 0.05
        )
        
        # Update meaning type rates
        total = len(self.interaction_history)
        if total > 0:
            self.metrics.affordance_rate = sum(1 for i in self.interaction_history 
                                            if i.meaning["meaning_type"] == MeaningType.AFFORDANCE.value) / total
            self.metrics.constraint_rate = sum(1 for i in self.interaction_history 
                                            if i.meaning["meaning_type"] == MeaningType.CONSTRAINT.value) / total
            self.metrics.neutral_rate = sum(1 for i in self.interaction_history 
                                        if i.meaning["meaning_type"] == MeaningType.NEUTRAL.value) / total
        
        # Update processing time
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0.0
        self.metrics.processing_time_ms = avg_processing_time * 1000
    
    def get_meaning_metrics(self) -> Dict[str, Any]:
        """Get current meaning metrics for monitoring and adaptation"""
        return {
            "meaning_stability": self.metrics.meaning_stability,
            "resonance_score": self.metrics.resonance_score,
            "familiarity_rate": self.metrics.familiarity_rate,
            "affordance_rate": self.metrics.affordance_rate,
            "constraint_rate": self.metrics.constraint_rate,
            "neutral_rate": self.metrics.neutral_rate,
            "processing_time_ms": self.metrics.processing_time_ms,
            "interaction_history_size": len(self.interaction_history),
            "circuit_breaker_state": self.circuit_breaker.get_state()
        }
    
    def get_recent_meaning_trends(self, n: int = 20) -> Dict[str, float]:
        """
        Analyze recent meaning trends to identify shifts in environmental perception.
        
        Returns frequencies of meaning types in recent interactions.
        """
        if not self.interaction_history:
            return {
                MeaningType.AFFORDANCE.value: 0.3,
                MeaningType.CONSTRAINT.value: 0.3,
                MeaningType.NEUTRAL.value: 0.4
            }
            
        recent = self.interaction_history[-n:]
        meaning_types = [i.meaning["meaning_type"] for i in recent]
        
        counts = {
            MeaningType.AFFORDANCE.value: meaning_types.count(MeaningType.AFFORDANCE.value),
            MeaningType.CONSTRAINT.value: meaning_types.count(MeaningType.CONSTRAINT.value),
            MeaningType.NEUTRAL.value: meaning_types.count(MeaningType.NEUTRAL.value)
        }
        
        total = len(recent)
        return {k: v / total for k, v in counts.items()}
    
    def get_ssp_integration_metrics(self) -> Dict[str, float]:
        """
        Get metrics suitable for SSP integration.
        
        Returns meaning metrics in a format SSP can use for reward shaping.
        """
        metrics = self.get_meaning_metrics()
        
        return {
            "meaning_stability": metrics["meaning_stability"],
            "resonance_score": metrics["resonance_score"],
            "familiarity_rate": metrics["familiarity_rate"],
            "affordance_rate": metrics["affordance_rate"],
            "constraint_rate": metrics["constraint_rate"],
            "neutral_rate": metrics["neutral_rate"],
            "processing_efficiency": 1.0 / (1.0 + metrics["processing_time_ms"])
        }
    
    def get_recent_interactions(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get recent interactions for analysis or reporting"""
        return [
            {
                "id": i.id,
                "timestamp": i.timestamp,
                "sensory_input": i.sensory_input,
                "action": i.action,
                "outcome": i.outcome,
                "meaning": i.meaning,
                "similarity_score": i.similarity_score,
                "success": i.success
            }
            for i in self.interaction_history[-n:]
        ]
    
    def evaluate_meaning_success(
        self,
        interaction_id: str,
        expected_outcome: Dict[str, Any],
        actual_outcome: Dict[str, Any]
    ) -> bool:
        """
        Evaluate whether meaning creation was successful.
        
        Args:
            interaction_id: ID of the interaction to evaluate
            expected_outcome: Expected outcome based on meaning
            actual_outcome: Actual outcome that occurred
            
        Returns:
            True if meaning creation was successful, False otherwise
        """
        # Find the interaction record
        record = next((i for i in self.interaction_history if i.id == interaction_id), None)
        if not record:
            log.warning(f"Interaction record not found: {interaction_id}")
            return False
        
        # Determine success based on outcome match
        expected_quality = expected_outcome.get("quality", 0.5)
        actual_quality = actual_outcome.get("quality", 0.5)
        
        # Meaning is successful if actual outcome matches expected outcome
        success = abs(actual_quality - expected_quality) < 0.2
        
        # Update record
        record.success = success
        
        # Update metrics
        if success:
            self.metrics.meaning_stability = min(1.0, self.metrics.meaning_stability + 0.01)
        else:
            self.metrics.meaning_stability = max(0.0, self.metrics.meaning_stability - 0.01)
        
        log.debug(f"Meaning evaluation: {interaction_id} - success={success}")
        return success