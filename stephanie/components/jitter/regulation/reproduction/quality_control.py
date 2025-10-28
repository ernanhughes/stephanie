"""
quality_control.py
==================
Implementation of quality control for offspring reproduction.

This module implements quality assessment mechanisms to ensure that
offspring meet minimum standards before being accepted into the population.

Key Features:
- Quality assessment of potential offspring
- Genetic diversity preservation
- Controlled variation with quality gates
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

log = logging.getLogger("stephanie.jitter.regulation.reproduction.quality")

class QualityConfig(BaseModel):
    """Validated configuration for QualityControlledReproduction"""
    quality_threshold: float = Field(0.6, ge=0.3, le=0.9, description="Minimum quality threshold for offspring")
    diversity_weight: float = Field(0.4, ge=0.1, le=0.7, description="Weight for genetic diversity")
    fitness_weight: float = Field(0.6, ge=0.3, le=0.9, description="Weight for parent fitness")
    mutation_quality_weight: float = Field(0.3, ge=0.1, le=0.5, description="Weight for mutation quality")
    heritage_preservation: bool = Field(True, description="Whether to preserve heritage")
    max_history: int = Field(100, ge=50, le=500, description="Maximum history length for learning")
    
    @validator('diversity_weight', 'fitness_weight', 'mutation_quality_weight')
    def validate_weights_sum_to_one(cls, v, values):
        weights = [
            values.get('diversity_weight', 0.4),
            values.get('fitness_weight', 0.6),
            values.get('mutation_quality_weight', 0.3)
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
    def assess_offspring_quality(genetics):
        # Quality assessment logic here
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
class QualityAssessmentRecord:
    """Record of a quality assessment for learning and adaptation"""
    timestamp: float = field(default_factory=time.time)
    offspring_id: str = ""
    quality_score: float = 0.0
    assessment_details: Dict[str, Any] = field(default_factory=dict)
    success: bool = False
    reason: str = ""

@dataclass
class QualityMetrics:
    """Metrics for quality control performance"""
    quality_assessment_rate: float = 0.0
    quality_trend: float = 0.0
    diversity_preservation: float = 0.0
    fitness_correlation: float = 0.0
    processing_time_ms: float = 0.0

class QualityControlledReproduction:
    """
    Implementation of quality control for offspring reproduction.
    
    Key Features:
    - Quality assessment of potential offspring
    - Genetic diversity preservation
    - Controlled variation with quality gates
    - Configuration validation with Pydantic
    - Circuit breaker pattern for resilience
    - Comprehensive telemetry and monitoring
    - SSP integration hooks
    - Performance optimizations
    """
    
    def __init__(self, cfg: Dict[str, Any]):
        try:
            # Validate configuration
            self.config = QualityConfig(**cfg)
            log.info("QualityControlledReproduction configuration validated successfully")
        except Exception as e:
            log.error(f"Configuration validation failed: {str(e)}")
            # Use safe defaults
            self.config = QualityConfig()
        
        # Initialize history
        self.quality_history: List[QualityAssessmentRecord] = []
        
        # Initialize metrics
        self.metrics = QualityMetrics()
        
        # Initialize performance tracking
        self.processing_times = []
        self.max_processing_history = 100
        
        # Initialize circuit breaker
        self.circuit_breaker = CircuitBreaker()
        
        log.info("QualityControlledReproduction initialized with quality gates")
    
    @CircuitBreaker()
    def assess_offspring_quality(self, parent_core, offspring_genetics: Dict[str, Any]) -> float:
        """
        Assess quality of potential offspring.
        
        Args:
            parent_core: The parent AutopoieticCore instance
            offspring_genetics: Genetic material of offspring to assess
            
        Returns:
            Quality score between 0 and 1
        """
        start_time = time.time()
        
        try:
            # Calculate quality score based on multiple factors
            quality_score = self._calculate_quality_score(
                parent_core, 
                offspring_genetics
            )
            
            # Record assessment
            self._record_quality_assessment(
                offspring_genetics.get("id", ""),
                quality_score,
                offspring_genetics
            )
            
            # Update metrics
            self._update_metrics(quality_score)
            
            # Log processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > self.max_processing_history:
                self.processing_times.pop(0)
                
            log.debug(f"Assessed offspring quality (score={quality_score:.3f})")
            return quality_score
            
        except Exception as e:
            log.error(f"Error assessing offspring quality: {str(e)}", exc_info=True)
            return 0.0
    
    def _calculate_quality_score(self, parent_core, offspring_genetics: Dict[str, Any]) -> float:
        """Calculate quality score for offspring"""
        # 1. Genetic diversity score
        diversity_score = self._calculate_diversity_score(parent_core, offspring_genetics)
        
        # 2. Parent fitness inheritance
        parent_fitness = self._calculate_parent_fitness(parent_core)
        
        # 3. Mutation quality (beneficial vs detrimental)
        mutation_quality = self._assess_mutation_quality(offspring_genetics)
        
        # 4. Heritage preservation (if enabled)
        heritage_score = 1.0
        if self.config.heritage_preservation:
            heritage_score = self._calculate_heritage_score(parent_core, offspring_genetics)
        
        # Weighted combination
        quality_score = (
            self.config.diversity_weight * diversity_score +
            self.config.fitness_weight * parent_fitness +
            self.config.mutation_quality_weight * mutation_quality +
            0.1 * heritage_score  # Smaller weight for heritage
        )
        
        # Ensure score stays within bounds
        return max(0.0, min(1.0, quality_score))
    
    def _calculate_diversity_score(self, parent_core, offspring_genetics: Dict[str, Any]) -> float:
        """Calculate genetic diversity score"""
        # Simple implementation - would be more sophisticated in production
        # For now, assume moderate diversity based on variation rate
        variation_rate = offspring_genetics.get("mutation_rate", 0.1)
        return 1.0 - variation_rate  # Lower variation = higher diversity
    
    def _calculate_parent_fitness(self, parent_core) -> float:
        """Calculate parent fitness from core system state"""
        # Get core metrics
        health = parent_core.homeostasis.get_telemetry().get("health", 0.5)
        boundary_integrity = parent_core.membrane.integrity
        energy_balance = parent_core.energy.level("metabolic") + parent_core.energy.level("cognitive")
        
        # Combine metrics into fitness score
        fitness = (
            health * 0.4 +
            boundary_integrity * 0.3 +
            energy_balance / 100.0 * 0.3  # Normalize to 0-1
        )
        
        return max(0.0, min(1.0, fitness))
    
    def _assess_mutation_quality(self, offspring_genetics: Dict[str, Any]) -> float:
        """Assess whether mutations are beneficial or detrimental"""
        # Simple implementation - would be more sophisticated in production
        # For now, assume random mutation quality based on mutation rate
        mutation_rate = offspring_genetics.get("mutation_rate", 0.1)
        base_quality = 1.0 - mutation_rate
        
        # Penalize if quality score decreased significantly
        # (This would require comparing to parent in a full implementation)
        return base_quality
    
    def _calculate_heritage_score(self, parent_core, offspring_genetics: Dict[str, Any]) -> float:
        """Calculate score based on heritage preservation"""
        # Check if heritage was properly inherited
        heritage = offspring_genetics.get("heritage", [])
        if not heritage:
            return 0.5  # Neutral score if no heritage
        
        # Check if parent is in heritage
        parent_id = parent_core.id
        heritage_match = parent_id in heritage
        
        # Return score based on heritage match
        return 0.8 if heritage_match else 0.3
    
    def _record_quality_assessment(
        self,
        offspring_id: str,
        quality_score: float,
        genetics: Dict[str, Any]
    ):
        """Record quality assessment for learning and adaptation"""
        # Create record
        record = QualityAssessmentRecord(
            offspring_id=offspring_id,
            quality_score=quality_score,
            assessment_details={
                "genetics": genetics,
                "timestamp": time.time()
            },
            success=quality_score >= self.config.quality_threshold,
            reason="quality_threshold_met" if quality_score >= self.config.quality_threshold else "below_threshold"
        )
        
        # Add to history
        self.quality_history.append(record)
        
        # Keep history bounded
        if len(self.quality_history) > self.config.max_history:
            self.quality_history.pop(0)
    
    def _update_metrics(self, quality_score: float):
        """Update quality control metrics based on recent assessments"""
        # Update quality assessment rate
        if len(self.quality_history) >= 10:
            # Calculate recent quality trends
            recent_scores = [r.quality_score for r in self.quality_history[-10:]]
            self.metrics.quality_assessment_rate = np.mean(recent_scores)
        
        # Update quality trend
        if len(self.quality_history) >= 5:
            recent_scores = [r.quality_score for r in self.quality_history[-5:]]
            if len(recent_scores) > 1:
                self.metrics.quality_trend = recent_scores[-1] - recent_scores[0]
        
        # Update processing time
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0.0
        self.metrics.processing_time_ms = avg_processing_time * 1000
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current quality control metrics for monitoring and adaptation"""
        return {
            "quality_assessment_rate": self.metrics.quality_assessment_rate,
            "quality_trend": self.metrics.quality_trend,
            "diversity_preservation": self.metrics.diversity_preservation,
            "fitness_correlation": self.metrics.fitness_correlation,
            "processing_time_ms": self.metrics.processing_time_ms,
            "history_size": len(self.quality_history),
            "circuit_breaker_state": self.circuit_breaker.get_state()
        }
    
    def get_ssp_integration_metrics(self) -> Dict[str, float]:
        """
        Get metrics suitable for SSP integration.
        
        Returns quality control metrics in a format SSP can use for reward shaping.
        """
        metrics = self.get_metrics()
        
        return {
            "quality_assessment_rate": metrics["quality_assessment_rate"],
            "quality_trend": metrics["quality_trend"],
            "diversity_preservation": metrics["diversity_preservation"],
            "fitness_correlation": metrics["fitness_correlation"],
            "processing_efficiency": 1.0 / (1.0 + metrics["processing_time_ms"])
        }
    
    def is_quality_acceptable(self, quality_score: float) -> bool:
        """Check if quality score meets threshold"""
        return quality_score >= self.config.quality_threshold
    
    def get_recent_assessments(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get recent quality assessments for analysis or reporting"""
        return [
            {
                "timestamp": r.timestamp,
                "offspring_id": r.offspring_id,
                "quality_score": r.quality_score,
                "assessment_details": r.assessment_details,
                "success": r.success,
                "reason": r.reason
            }
            for r in self.quality_history[-n:]
        ]
    
    def get_optimal_quality_profile(
        self,
        context: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Get optimal quality assessment profile for given context.
        
        Args:
            context: Current context for which to determine optimal profile
            
        Returns:
            Optimal quality parameters for the context
        """
        if len(self.quality_history) < 10:
            return {
                "quality_threshold": self.config.quality_threshold,
                "diversity_weight": self.config.diversity_weight,
                "fitness_weight": self.config.fitness_weight,
                "mutation_quality_weight": self.config.mutation_quality_weight
            }
        
        # Find similar contexts
        similar = []
        for record in self.quality_history:
            # Simple similarity measure (would be more sophisticated in production)
            context_match = 0.0
            for key in context:
                if key in record.assessment_details:
                    # Numeric context
                    if isinstance(context[key], (int, float)):
                        context_match += 1.0 - min(1.0, abs(context[key] - record.assessment_details[key]))
                    # Categorical context
                    else:
                        context_match += 1.0 if context[key] == record.assessment_details[key] else 0.0
            
            # Normalize
            context_match = context_match / max(1, len(context))
            if context_match > 0.5:  # Minimum similarity
                similar.append((record, context_match))
        
        # If no similar contexts, return current configuration
        if not similar:
            return {
                "quality_threshold": self.config.quality_threshold,
                "diversity_weight": self.config.diversity_weight,
                "fitness_weight": self.config.fitness_weight,
                "mutation_quality_weight": self.config.mutation_quality_weight
            }
        
        # Calculate weighted average of quality parameters
        total_weight = 0.0
        avg_params = {
            "quality_threshold": 0.0,
            "diversity_weight": 0.0,
            "fitness_weight": 0.0,
            "mutation_quality_weight": 0.0
        }
        
        for record, weight in similar:
            total_weight += weight
            avg_params["quality_threshold"] += record.quality_score * weight
        
        # Normalize
        if total_weight > 0:
            avg_params["quality_threshold"] /= total_weight
        
        return avg_params