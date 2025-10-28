"""
energy_optimizer.py
===================
Implementation of energy optimization for performance and efficiency.

This module implements the energy optimization system with:
- Adaptive energy allocation based on system state
- Crisis-responsive energy reallocation
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

from .energy import EnergyPools, EnergyPool
from .metabolic_pathways import MetabolicPathways, PathwayType

log = logging.getLogger("stephanie.jitter.metabolism.optimizer")

class OptimizationGoal(str, Enum):
    """Types of energy optimization goals"""
    STABILITY = "stability"
    PERFORMANCE = "performance"
    CRISIS_RESPONSE = "crisis_response"
    REPRODUCTION = "reproduction"

class EnergyOptimizerConfig(BaseModel):
    """Validated configuration for EnergyOptimizer"""
    stability_threshold: float = Field(0.3, ge=0.1, le=0.5, description="Threshold for stability optimization")
    performance_threshold: float = Field(0.7, ge=0.5, le=0.9, description="Threshold for performance optimization")
    crisis_threshold: float = Field(0.2, ge=0.1, le=0.4, description="Threshold for crisis response")
    reproduction_threshold: float = Field(0.8, ge=0.6, le=0.95, description="Threshold for reproduction")
    allocation_smoothing: float = Field(0.9, ge=0.7, le=0.95, description="Smoothing factor for allocation")
    max_history: int = Field(100, ge=50, le=500, description="Maximum history length for learning")
    
    @validator('stability_threshold', 'crisis_threshold')
    def validate_stability_thresholds(cls, v, values):
        if 'performance_threshold' in values and v >= values['performance_threshold']:
            raise ValueError('stability_threshold must be less than performance_threshold')
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
    def optimize_allocation():
        # Optimization logic here
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
class AllocationRecord:
    """Record of an energy allocation decision for learning"""
    id: str = field(default_factory=lambda: f"alloc_{int(time.time())}_{np.random.randint(1000)}")
    timestamp: float = field(default_factory=time.time)
    allocation: Dict[str, float] = field(default_factory=dict)
    goal: str = OptimizationGoal.STABILITY.value
    context: Dict[str, Any] = field(default_factory=dict)
    outcome: Dict[str, Any] = field(default_factory=dict)
    reward: float = 0.0
    success: bool = False

@dataclass
class EnergyOptimizerMetrics:
    """Metrics for energy optimization performance"""
    allocation_efficiency: float = 0.5
    goal_achievement: float = 0.5
    crisis_response_efficiency: float = 0.5
    reproduction_success_rate: float = 0.0
    processing_time_ms: float = 0.0

class EnergyOptimizer:
    """
    Manages energy optimization for performance and efficiency in the autopoietic system.
    
    Key Features:
    - Adaptive energy allocation based on system state
    - Crisis-responsive energy reallocation
    - Configuration validation with Pydantic
    - Circuit breaker pattern for resilience
    - Comprehensive telemetry and monitoring
    - SSP integration hooks
    - Performance optimizations
    """
    
    def __init__(
        self,
        cfg: Dict[str, Any],
        energy_pools: EnergyPools,
        metabolic_pathways: MetabolicPathways
    ):
        try:
            # Validate configuration
            self.config = EnergyOptimizerConfig(**cfg)
            log.info("EnergyOptimizer configuration validated successfully")
        except Exception as e:
            log.error(f"Configuration validation failed: {str(e)}")
            # Use safe defaults
            self.config = EnergyOptimizerConfig()
        
        self.energy_pools = energy_pools
        self.metabolic_pathways = metabolic_pathways
        
        # Initialize current allocation
        self.current_allocation = {
            EnergyPool.COGNITIVE.value: 0.3,
            EnergyPool.METABOLIC.value: 0.5,
            EnergyPool.RESERVE.value: 0.2
        }
        
        # Initialize history
        self.allocation_history: List[AllocationRecord] = []
        
        # Initialize circuit breaker
        self.circuit_breaker = CircuitBreaker()
        
        # Initialize metrics
        self.metrics = EnergyOptimizerMetrics()
        
        # Initialize performance tracking
        self.processing_times = []
        self.max_processing_history = 100
        
        log.info("EnergyOptimizer initialized with adaptive allocation features")
    
    @CircuitBreaker()
    def optimize_allocation(
        self,
        system_state: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Optimize energy allocation based on current system state.
        
        Args:
            system_state: Current system state for context
            
        Returns:
            Optimized allocation ratios (summing to 1.0)
        """
        start_time = time.time()
        
        try:
            # Determine optimization goal based on system state
            goal = self._determine_optimization_goal(system_state)
            
            # Calculate optimized allocation
            optimized_allocation = self._calculate_optimized_allocation(goal, system_state)
            
            # Store current state for history
            old_allocation = self.current_allocation.copy()
            
            # Update current allocation
            self.current_allocation = optimized_allocation
            
            # Record for learning
            self._record_allocation(
                old_allocation,
                goal,
                system_state
            )
            
            # Update metrics
            self._update_metrics()
            
            # Log processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > self.max_processing_history:
                self.processing_times.pop(0)
                
            log.debug(f"Optimized energy allocation (goal={goal}, allocation={optimized_allocation})")
            return optimized_allocation
            
        except Exception as e:
            log.error(f"Error optimizing allocation: {str(e)}", exc_info=True)
            return self.current_allocation.copy()
    
    def _determine_optimization_goal(self, system_state: Dict[str, Any]) -> OptimizationGoal:
        """Determine optimization goal based on system state"""
        # Extract relevant metrics
        energy_balance = system_state.get("energy_balance", 1.0)
        boundary_integrity = system_state.get("boundary_integrity", 0.8)
        cognitive_flow = system_state.get("cognitive_flow", 0.5)
        crisis_level = system_state.get("crisis_level", 0.0)
        
        # Determine goal based on metrics
        if crisis_level > self.config.crisis_threshold:
            return OptimizationGoal.CRISIS_RESPONSE
        elif energy_balance < self.config.stability_threshold:
            return OptimizationGoal.STABILITY
        elif energy_balance > self.config.performance_threshold:
            return OptimizationGoal.PERFORMANCE
        elif cognitive_flow > self.config.reproduction_threshold:
            return OptimizationGoal.REPRODUCTION
        else:
            return OptimizationGoal.STABILITY
    
    def _calculate_optimized_allocation(
        self,
        goal: OptimizationGoal,
        system_state: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate optimized allocation based on goal"""
        # Base allocation (will be adjusted based on goal)
        allocation = self.current_allocation.copy()
        
        if goal == OptimizationGoal.STABILITY:
            # Prioritize metabolic energy for stability
            allocation[EnergyPool.METABOLIC.value] = max(
                0.6, 
                allocation[EnergyPool.METABOLIC.value] * 1.1
            )
            allocation[EnergyPool.COGNITIVE.value] = min(
                0.3, 
                allocation[EnergyPool.COGNITIVE.value] * 0.9
            )
        
        elif goal == OptimizationGoal.PERFORMANCE:
            # Prioritize cognitive energy for performance
            allocation[EnergyPool.COGNITIVE.value] = max(
                0.5, 
                allocation[EnergyPool.COGNITIVE.value] * 1.1
            )
            allocation[EnergyPool.METABOLIC.value] = min(
                0.4, 
                allocation[EnergyPool.METABOLIC.value] * 0.9
            )
        
        elif goal == OptimizationGoal.CRISIS_RESPONSE:
            # Prioritize metabolic energy for crisis response
            allocation[EnergyPool.METABOLIC.value] = max(
                0.7, 
                allocation[EnergyPool.METABOLIC.value] * 1.2
            )
            allocation[EnergyPool.COGNITIVE.value] = min(
                0.2, 
                allocation[EnergyPool.COGNITIVE.value] * 0.8
            )
            allocation[EnergyPool.RESERVE.value] = max(
                0.1, 
                allocation[EnergyPool.RESERVE.value] * 1.1
            )
        
        elif goal == OptimizationGoal.REPRODUCTION:
            # Prioritize reserve energy for reproduction
            allocation[EnergyPool.RESERVE.value] = max(
                0.3, 
                allocation[EnergyPool.RESERVE.value] * 1.1
            )
            allocation[EnergyPool.METABOLIC.value] = min(
                0.4, 
                allocation[EnergyPool.METABOLIC.value] * 0.9
            )
        
        # Normalize to sum to 1
        total = sum(allocation.values())
        for pool in allocation:
            allocation[pool] /= total
        
        return allocation
    
    def _record_allocation(
        self,
        old_allocation: Dict[str, float],
        goal: OptimizationGoal,
        context: Dict[str, Any]
    ):
        """Record allocation decision for learning and adaptation"""
        # Calculate outcome (simplified - would be more sophisticated in production)
        outcome = {
            "old_allocation": old_allocation,
            "new_allocation": self.current_allocation.copy(),
            "goal": goal.value
        }
        
        # Create record
        record = AllocationRecord(
            allocation=self.current_allocation.copy(),
            goal=goal.value,
            context=context,
            outcome=outcome,
            reward=0.0,  # Will be updated later
            success=False  # Will be updated later
        )
        
        # Add to history
        self.allocation_history.append(record)
        
        # Keep history bounded
        if len(self.allocation_history) > self.config.max_history:
            self.allocation_history.pop(0)
    
    def _update_metrics(self):
        """Update optimization metrics based on history"""
        # Calculate allocation efficiency
        if len(self.allocation_history) >= 10:
            # Calculate variance of allocations
            cognitive_values = [a.allocation[EnergyPool.COGNITIVE.value] for a in self.allocation_history[-10:]]
            metabolic_values = [a.allocation[EnergyPool.METABOLIC.value] for a in self.allocation_history[-10:]]
            reserve_values = [a.allocation[EnergyPool.RESERVE.value] for a in self.allocation_history[-10:]]
            
            self.metrics.allocation_efficiency = (
                1.0 - np.var(cognitive_values) * 0.4 -
                np.var(metabolic_values) * 0.4 -
                np.var(reserve_values) * 0.2
            )
        
        # Calculate goal achievement
        if len(self.allocation_history) > 0:
            goals = [a.goal for a in self.allocation_history]
            successes = [a.success for a in self.allocation_history]
            
            if successes:
                self.metrics.goal_achievement = sum(successes) / len(successes)
        
        # Calculate crisis response efficiency
        crisis_responses = [
            a for a in self.allocation_history 
            if a.goal == OptimizationGoal.CRISIS_RESPONSE.value
        ]
        if crisis_responses:
            self.metrics.crisis_response_efficiency = sum(1 for a in crisis_responses if a.success) / len(crisis_responses)
        
        # Calculate reproduction success rate
        reproduction_attempts = [
            a for a in self.allocation_history 
            if a.goal == OptimizationGoal.REPRODUCTION.value
        ]
        if reproduction_attempts:
            self.metrics.reproduction_success_rate = sum(1 for a in reproduction_attempts if a.success) / len(reproduction_attempts)
        
        # Update processing time
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0.0
        self.metrics.processing_time_ms = avg_processing_time * 1000
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current optimization metrics for monitoring and adaptation"""
        return {
            "allocation_efficiency": self.metrics.allocation_efficiency,
            "goal_achievement": self.metrics.goal_achievement,
            "crisis_response_efficiency": self.metrics.crisis_response_efficiency,
            "reproduction_success_rate": self.metrics.reproduction_success_rate,
            "processing_time_ms": self.metrics.processing_time_ms,
            "allocation_history_size": len(self.allocation_history),
            "circuit_breaker_state": self.circuit_breaker.get_state()
        }
    
    def get_ssp_integration_metrics(self) -> Dict[str, float]:
        """
        Get metrics suitable for SSP integration.
        
        Returns optimization metrics in a format SSP can use for reward shaping.
        """
        metrics = self.get_metrics()
        
        return {
            "allocation_efficiency": metrics["allocation_efficiency"],
            "goal_achievement": metrics["goal_achievement"],
            "crisis_response_efficiency": metrics["crisis_response_efficiency"],
            "reproduction_success_rate": metrics["reproduction_success_rate"],
            "processing_efficiency": 1.0 / (1.0 + metrics["processing_time_ms"])
        }
    
    def evaluate_allocation_success(
        self,
        allocation_id: str,
        performance_before: float,
        performance_after: float
    ) -> bool:
        """
        Evaluate whether allocation decision was successful.
        
        Args:
            allocation_id: ID of the allocation decision to evaluate
            performance_before: Performance before allocation
            performance_after: Performance after allocation
            
        Returns:
            True if allocation was successful, False otherwise
        """
        # Find the allocation record
        record = next((a for a in self.allocation_history if a.id == allocation_id), None)
        if not record:
            log.warning(f"Allocation record not found: {allocation_id}")
            return False
        
        # Determine success based on performance improvement
        success = performance_after > performance_before
        reward = performance_after - performance_before
        
        # Update record
        record.reward = reward
        record.success = success
        
        # Update metrics
        if success:
            self.metrics.goal_achievement = min(1.0, self.metrics.goal_achievement + 0.01)
        else:
            self.metrics.goal_achievement = max(0.0, self.metrics.goal_achievement - 0.01)
        
        log.debug(f"Allocation evaluation: {allocation_id} - success={success}, reward={reward:.3f}")
        return success
    
    def get_recent_allocations(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get recent allocation decisions for analysis or reporting"""
        return [
            {
                "id": a.id,
                "timestamp": a.timestamp,
                "allocation": a.allocation,
                "goal": a.goal,
                "context": a.context,
                "outcome": a.outcome,
                "reward": a.reward,
                "success": a.success
            }
            for a in self.allocation_history[-n:]
        ]
    
    def get_optimal_allocation_profile(
        self,
        context: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Get optimal allocation profile for given context based on historical data.
        
        Args:
            context: Current context for which to determine optimal profile
            
        Returns:
            Optimal allocation ratios for the context
        """
        if len(self.allocation_history) < 10:
            return self.current_allocation.copy()
        
        # Find similar contexts
        similar = []
        for record in self.allocation_history:
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
        
        # If no similar contexts, return current allocation
        if not similar:
            return self.current_allocation.copy()
        
        # Calculate weighted average of allocations
        total_weight = 0.0
        avg_allocation = {p: 0.0 for p in self.current_allocation}
        
        for record, weight in similar:
            total_weight += weight
            for pool in avg_allocation:
                avg_allocation[pool] += record.allocation.get(pool, 0.0) * weight
        
        # Normalize
        if total_weight > 0:
            for pool in avg_allocation:
                avg_allocation[pool] /= total_weight
        
        # Ensure allocation sums to 1 and is within reasonable bounds
        total = sum(avg_allocation.values())
        if total > 0:
            for pool in avg_allocation:
                avg_allocation[pool] /= total
        
        for pool in avg_allocation:
            avg_allocation[pool] = max(
                0.1,  # Minimum allocation
                min(0.8, avg_allocation[pool])  # Maximum allocation
            )
        
        # Normalize again after clamping
        total = sum(avg_allocation.values())
        if total > 0:
            for pool in avg_allocation:
                avg_allocation[pool] /= total
        
        return avg_allocation
    
    def apply_optimized_allocation(self):
        """Apply optimized allocation to energy pools and pathways"""
        # This would typically adjust energy transfer rates between pools
        # based on the current allocation strategy
        
        # Example: Adjust metabolic to cognitive conversion rate
        cognitive_target = self.energy_pools.level(EnergyPool.METABOLIC.value) * self.current_allocation[EnergyPool.COGNITIVE.value]
        reserve_target = self.energy_pools.level(EnergyPool.METABOLIC.value) * self.current_allocation[EnergyPool.RESERVE.value]
        
        # Transfer energy according to allocation
        self.metabolic_pathways.convert(
            PathwayType.METABOLIC_TO_COGNITIVE.value,
            cognitive_target
        )
        self.metabolic_pathways.convert(
            PathwayType.METABOLIC_TO_RESERVE.value,
            reserve_target
        )
        
        log.debug(f"Applied optimized allocation: {self.current_allocation}")