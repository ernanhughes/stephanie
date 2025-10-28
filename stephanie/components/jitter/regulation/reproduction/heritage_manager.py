"""
heritage_manager.py
===================
Implementation of heritage preservation for offspring reproduction.

This module implements the heritage management system that tracks and preserves
genetic lineage information across generations to support evolutionary learning.

Key Features:
- Heritage tracking for genetic lineage
- Ancestral information preservation
- Heritage diversity analysis
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
import uuid

log = logging.getLogger("stephanie.jitter.regulation.reproduction.heritage")

class HeritageConfig(BaseModel):
    """Validated configuration for HeritageManager"""
    max_heritage_length: int = Field(10, ge=5, le=50, description="Maximum heritage lineage length")
    diversity_threshold: float = Field(0.5, ge=0.1, le=0.9, description="Threshold for heritage diversity")
    preservation_rate: float = Field(0.9, ge=0.7, le=0.99, description="Rate of heritage preservation")
    max_history: int = Field(1000, ge=100, le=10000, description="Maximum history length for heritage tracking")
    
    @validator('max_heritage_length')
    def validate_max_heritage_length(cls, v):
        if v < 1:
            raise ValueError('max_heritage_length must be at least 1')
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
    def track_heritage(genetics):
        # Heritage tracking logic here
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
class HeritageRecord:
    """Record of a heritage event for tracking and analysis"""
    timestamp: float = field(default_factory=time.time)
    offspring_id: str = ""
    parent_id: str = ""
    heritage: List[str] = field(default_factory=list)
    diversity_score: float = 0.0
    preservation_status: str = "preserved"
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HeritageMetrics:
    """Metrics for heritage management performance"""
    heritage_preservation_rate: float = 0.0
    diversity_conservation: float = 0.0
    lineage_depth: float = 0.0
    heritage_completeness: float = 0.0
    processing_time_ms: float = 0.0

class HeritageManager:
    """
    Implementation of heritage preservation for offspring reproduction.
    
    Key Features:
    - Heritage tracking for genetic lineage
    - Ancestral information preservation
    - Heritage diversity analysis
    - Configuration validation with Pydantic
    - Circuit breaker pattern for resilience
    - Comprehensive telemetry and monitoring
    - SSP integration hooks
    - Performance optimizations
    """
    
    def __init__(self, cfg: Dict[str, Any]):
        try:
            # Validate configuration
            self.config = HeritageConfig(**cfg)
            log.info("HeritageManager configuration validated successfully")
        except Exception as e:
            log.error(f"Configuration validation failed: {str(e)}")
            # Use safe defaults
            self.config = HeritageConfig()
        
        # Initialize heritage tracking
        self.heritage_records: List[HeritageRecord] = []
        
        # Initialize metrics
        self.metrics = HeritageMetrics()
        
        # Initialize performance tracking
        self.processing_times = []
        self.max_processing_history = 100
        
        # Initialize circuit breaker
        self.circuit_breaker = CircuitBreaker()
        
        log.info("HeritageManager initialized with lineage tracking")
    
    @CircuitBreaker()
    def track_heritage(self, offspring_genetics: Dict[str, Any], parent_id: str) -> Dict[str, Any]:
        """
        Track heritage lineage for offspring.
        
        Args:
            offspring_genetics: Genetic material of offspring
            parent_id: ID of parent organism
            
        Returns:
            Updated genetics with heritage information
        """
        start_time = time.time()
        
        try:
            # Create heritage record
            heritage_record = self._create_heritage_record(
                offspring_genetics,
                parent_id
            )
            
            # Update heritage records
            self._update_heritage_records(heritage_record)
            
            # Update metrics
            self._update_metrics()
            
            # Log processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > self.max_processing_history:
                self.processing_times.pop(0)
                
            log.debug(f"Tracked heritage for offspring {offspring_genetics.get('id', 'unknown')}")
            return offspring_genetics
            
        except Exception as e:
            log.error(f"Error tracking heritage: {str(e)}", exc_info=True)
            return offspring_genetics
    
    def _create_heritage_record(
        self, 
        offspring_genetics: Dict[str, Any], 
        parent_id: str
    ) -> HeritageRecord:
        """Create heritage record for tracking"""
        # Get existing heritage from parent
        heritage = offspring_genetics.get("heritage", [])
        
        # Add parent to heritage if not already present
        if parent_id not in heritage:
            heritage.append(parent_id)
        
        # Keep heritage bounded
        if len(heritage) > self.config.max_heritage_length:
            heritage = heritage[-self.config.max_heritage_length:]
        
        # Calculate diversity score (simplified)
        diversity_score = self._calculate_diversity_score(heritage)
        
        # Create record
        record = HeritageRecord(
            offspring_id=offspring_genetics.get("id", ""),
            parent_id=parent_id,
            heritage=heritage,
            diversity_score=diversity_score,
            metadata={
                "mutation_rate": offspring_genetics.get("mutation_rate", 0.1),
                "generation": offspring_genetics.get("generation", 0)
            }
        )
        
        return record
    
    def _calculate_diversity_score(self, heritage: List[str]) -> float:
        """Calculate diversity score based on heritage lineage"""
        if len(heritage) <= 1:
            return 0.5  # Neutral score for small lineage
        
        # Simple diversity calculation (would be more sophisticated in production)
        # For now, return inverse of lineage length (more diverse = shorter lineage)
        return 1.0 / max(1, len(heritage) / 2.0)
    
    def _update_heritage_records(self, record: HeritageRecord):
        """Update heritage records with new information"""
        # Add to history
        self.heritage_records.append(record)
        
        # Keep history bounded
        if len(self.heritage_records) > self.config.max_history:
            self.heritage_records.pop(0)
    
    def _update_metrics(self):
        """Update heritage management metrics based on history"""
        # Update heritage preservation rate
        if len(self.heritage_records) > 0:
            preserved = sum(1 for r in self.heritage_records if r.preservation_status == "preserved")
            self.metrics.heritage_preservation_rate = preserved / len(self.heritage_records)
        
        # Update diversity conservation
        if len(self.heritage_records) > 0:
            diversity_scores = [r.diversity_score for r in self.heritage_records]
            self.metrics.diversity_conservation = np.mean(diversity_scores)
        
        # Update lineage depth
        if len(self.heritage_records) > 0:
            lineage_lengths = [len(r.heritage) for r in self.heritage_records]
            self.metrics.lineage_depth = np.mean(lineage_lengths)
        
        # Update heritage completeness
        if len(self.heritage_records) > 0:
            # Calculate average heritage completeness (simplified)
            self.metrics.heritage_completeness = 1.0  # Would be more complex in production
        
        # Update processing time
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0.0
        self.metrics.processing_time_ms = avg_processing_time * 1000
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current heritage management metrics for monitoring and adaptation"""
        return {
            "heritage_preservation_rate": self.metrics.heritage_preservation_rate,
            "diversity_conservation": self.metrics.diversity_conservation,
            "lineage_depth": self.metrics.lineage_depth,
            "heritage_completeness": self.metrics.heritage_completeness,
            "processing_time_ms": self.metrics.processing_time_ms,
            "history_size": len(self.heritage_records),
            "circuit_breaker_state": self.circuit_breaker.get_state()
        }
    
    def get_ssp_integration_metrics(self) -> Dict[str, float]:
        """
        Get metrics suitable for SSP integration.
        
        Returns heritage management metrics in a format SSP can use for reward shaping.
        """
        metrics = self.get_metrics()
        
        return {
            "heritage_preservation_rate": metrics["heritage_preservation_rate"],
            "diversity_conservation": metrics["diversity_conservation"],
            "lineage_depth": metrics["lineage_depth"],
            "heritage_completeness": metrics["heritage_completeness"],
            "processing_efficiency": 1.0 / (1.0 + metrics["processing_time_ms"])
        }
    
    def get_heritage_diversity(self, heritage: List[str]) -> float:
        """
        Calculate diversity of a specific heritage lineage.
        
        Args:
            heritage: List of IDs in the heritage lineage
            
        Returns:
            Diversity score (0-1)
        """
        return self._calculate_diversity_score(heritage)
    
    def get_family_tree(self, ancestor_id: str) -> Dict[str, Any]:
        """
        Generate family tree information for an ancestor.
        
        Args:
            ancestor_id: ID of ancestor to generate family tree for
            
        Returns:
            Family tree information
        """
        # Find all descendants of this ancestor
        descendants = []
        for record in self.heritage_records:
            if ancestor_id in record.heritage:
                descendants.append({
                    "offspring_id": record.offspring_id,
                    "parent_id": record.parent_id,
                    "diversity_score": record.diversity_score,
                    "generation": record.metadata.get("generation", 0)
                })
        
        return {
            "ancestor_id": ancestor_id,
            "descendants": descendants,
            "total_descendants": len(descendants),
            "average_diversity": np.mean([d["diversity_score"] for d in descendants]) if descendants else 0.0
        }
    
    def get_heritage_statistics(self) -> Dict[str, Any]:
        """Get comprehensive heritage statistics"""
        if not self.heritage_records:
            return {
                "total_records": 0,
                "average_diversity": 0.0,
                "average_lineage_depth": 0.0,
                "preservation_rate": 0.0,
                "diversity_distribution": {}
            }
        
        # Calculate statistics
        diversity_scores = [r.diversity_score for r in self.heritage_records]
        lineage_lengths = [len(r.heritage) for r in self.heritage_records]
        
        # Distribution of diversity scores
        diversity_bins = np.linspace(0, 1, 6)  # 5 bins
        diversity_counts = np.histogram(diversity_scores, bins=diversity_bins)[0]
        diversity_distribution = {
            f"bin_{i}": count for i, count in enumerate(diversity_counts)
        }
        
        return {
            "total_records": len(self.heritage_records),
            "average_diversity": np.mean(diversity_scores),
            "average_lineage_depth": np.mean(lineage_lengths),
            "preservation_rate": self.metrics.heritage_preservation_rate,
            "diversity_distribution": diversity_distribution,
            "max_lineage_depth": max(lineage_lengths) if lineage_lengths else 0,
            "min_lineage_depth": min(lineage_lengths) if lineage_lengths else 0
        }
    
    def get_recent_heritage(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get recent heritage tracking records for analysis or reporting"""
        return [
            {
                "timestamp": r.timestamp,
                "offspring_id": r.offspring_id,
                "parent_id": r.parent_id,
                "heritage": r.heritage,
                "diversity_score": r.diversity_score,
                "preservation_status": r.preservation_status,
                "metadata": r.metadata
            }
            for r in self.heritage_records[-n:]
        ]
    
    def reset(self):
        """Reset heritage management state"""
        self.heritage_records.clear()
        self.metrics = HeritageMetrics()
        log.info("HeritageManager reset")
    
    def get_heritage_length(self, offspring_id: str) -> int:
        """
        Get the length of heritage lineage for a specific offspring.
        
        Args:
            offspring_id: ID of offspring to check
            
        Returns:
            Length of heritage lineage
        """
        for record in self.heritage_records:
            if record.offspring_id == offspring_id:
                return len(record.heritage)
        return 0
    
    def is_diverse_lineage(self, heritage: List[str]) -> bool:
        """
        Check if a heritage lineage is sufficiently diverse.
        
        Args:
            heritage: List of IDs in the heritage lineage
            
        Returns:
            True if lineage is diverse, False otherwise
        """
        diversity_score = self._calculate_diversity_score(heritage)
        return diversity_score >= self.config.diversity_threshold