# stephanie/components/jitter/metabolism/energy.py
"""
energy.py
=========
Implementation of energy pools for the autopoietic system.

This module implements the energy management system with:
- Cognitive, metabolic, and reserve energy pools
- Energy level monitoring and adjustment
- Configuration validation with Pydantic
- Circuit breaker pattern for resilience
- Comprehensive telemetry and monitoring
- SSP integration hooks
- Performance optimizations

Key Features:
- Three distinct energy pools with different functions
- Energy conversion between pools
- Energy maintenance costs
- Energy-based regulation of system functions
- Organizationally closed production of energy pathways
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Dict

import numpy as np
from pydantic import BaseModel, Field, validator

log = logging.getLogger("stephanie.jitter.metabolism.energy")

class EnergyPool(str, Enum):
    """Types of energy pools"""
    COGNITIVE = "cognitive"
    METABOLIC = "metabolic"
    RESERVE = "reserve"

class EnergyConfig(BaseModel):
    """Validated configuration for EnergyPools"""
    initial_cognitive: float = Field(50.0, ge=10.0, le=100.0, description="Initial cognitive energy")
    initial_metabolic: float = Field(50.0, ge=10.0, le=100.0, description="Initial metabolic energy")
    initial_reserve: float = Field(20.0, ge=0.0, le=50.0, description="Initial reserve energy")
    max_reserve: float = Field(100.0, ge=50.0, le=200.0, description="Maximum reserve capacity")
    metabolic_baseline: float = Field(0.1, ge=0.01, le=0.5, description="Baseline metabolic consumption")
    cognitive_baseline: float = Field(0.05, ge=0.01, le=0.2, description="Baseline cognitive consumption")
    reserve_transfer_rate: float = Field(0.05, ge=0.01, le=0.1, description="Rate of reserve transfer")
    
    @validator('initial_reserve')
    def validate_reserve_limits(cls, v, values):
        if 'max_reserve' in values and v > values['max_reserve']:
            raise ValueError('initial_reserve must be less than or equal to max_reserve')
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
    def level(pool):
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
class EnergySnapshot:
    """Snapshot of energy state for telemetry and reproduction"""
    cognitive: float
    metabolic: float
    reserve: float
    energy_balance: float
    timestamp: float = field(default_factory=time.time)

@dataclass
class EnergyMetrics:
    """Metrics for energy management performance"""
    energy_stability: float = 0.5
    metabolic_efficiency: float = 0.5
    reserve_utilization: float = 0.5
    energy_balance: float = 1.0
    processing_time_ms: float = 0.0

class EnergyPools:
    """
    Implementation of energy pools for the autopoietic system.
    
    Key Features:
    - Three distinct energy pools (cognitive, metabolic, reserve)
    - Energy level monitoring and adjustment
    - Configuration validation with Pydantic
    - Circuit breaker pattern for resilience
    - Comprehensive telemetry and monitoring
    - SSP integration hooks
    - Performance optimizations
    """
    
    def __init__(self, cfg: Dict[str, Any]):
        try:
            # Validate configuration
            self.config = EnergyConfig(**cfg)
            log.info("EnergyPools configuration validated successfully")
        except Exception as e:
            log.error(f"Configuration validation failed: {str(e)}")
            # Use safe defaults
            self.config = EnergyConfig()
        
        # Initialize energy pools
        self.energy_pools = {
            EnergyPool.COGNITIVE.value: float(self.config.initial_cognitive),
            EnergyPool.METABOLIC.value: float(self.config.initial_metabolic),
            EnergyPool.RESERVE.value: float(self.config.initial_reserve),
        }
        self.max_reserve = float(self.config.max_reserve)
        
        # Initialize history
        self.energy_history = []
        
        # Initialize circuit breaker
        self.circuit_breaker = CircuitBreaker()
        
        # Initialize metrics
        self.metrics = EnergyMetrics()
        
        # Initialize performance tracking
        self.processing_times = []
        self.max_processing_history = 100
        
        log.info("EnergyPools initialized with three-pool energy management")
    
    @CircuitBreaker()
    def level(self, pool: str) -> float:
        """
        Get current level of an energy pool.
        
        Args:
            pool: Energy pool to check (cognitive, metabolic, reserve)
            
        Returns:
            Current energy level
        """
        start_time = time.time()
        
        try:
            # Validate pool
            if pool not in self.energy_pools:
                log.warning(f"Invalid energy pool: {pool}")
                return 0.0
            
            # Return level
            level = self.energy_pools[pool]
            
            # Log processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > self.max_processing_history:
                self.processing_times.pop(0)
                
            return level
            
        except Exception as e:
            log.error(f"Error getting energy level: {str(e)}", exc_info=True)
            return 0.0
    
    @CircuitBreaker()
    def consume(self, pool: str, amount: float) -> float:
        """
        Consume energy from a pool.
        
        Args:
            pool: Energy pool to consume from
            amount: Amount to consume
            
        Returns:
            Actual amount consumed (may be less than requested)
        """
        start_time = time.time()
        
        try:
            # Validate pool
            if pool not in self.energy_pools:
                log.warning(f"Invalid energy pool: {pool}")
                return 0.0
            
            # Calculate actual consumption (can't consume more than available)
            actual_consumption = min(amount, self.energy_pools[pool])
            
            # Update energy pool
            self.energy_pools[pool] -= actual_consumption
            
            # Record for history
            self._record_energy_change(pool, -actual_consumption)
            
            # Update metrics
            self._update_metrics()
            
            # Log processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > self.max_processing_history:
                self.processing_times.pop(0)
                
            log.debug(f"Consumed energy from {pool} (amount={actual_consumption:.3f}, remaining={self.energy_pools[pool]:.3f})")
            return actual_consumption
            
        except Exception as e:
            log.error(f"Error consuming energy: {str(e)}", exc_info=True)
            return 0.0
    
    @CircuitBreaker()
    def replenish(self, pool: str, amount: float) -> float:
        """
        Replenish energy to a pool.
        
        Args:
            pool: Energy pool to replenish
            amount: Amount to replenish
            
        Returns:
            Actual amount replenished (may be less than requested)
        """
        start_time = time.time()
        
        try:
            # Validate pool
            if pool not in self.energy_pools:
                log.warning(f"Invalid energy pool: {pool}")
                return 0.0
            
            # Calculate maximum replenish amount based on pool
            max_amount = self.max_reserve if pool == EnergyPool.RESERVE.value else float('inf')
            current_level = self.energy_pools[pool]
            available_space = max_amount - current_level
            
            # Calculate actual replenishment
            actual_replenishment = min(amount, available_space)
            
            # Update energy pool
            self.energy_pools[pool] = min(max_amount, self.energy_pools[pool] + actual_replenishment)
            
            # Record for history
            self._record_energy_change(pool, actual_replenishment)
            
            # Update metrics
            self._update_metrics()
            
            # Log processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > self.max_processing_history:
                self.processing_times.pop(0)
                
            log.debug(f"Replenished energy to {pool} (amount={actual_replenishment:.3f}, level={self.energy_pools[pool]:.3f})")
            return actual_replenishment
            
        except Exception as e:
            log.error(f"Error replenishing energy: {str(e)}", exc_info=True)
            return 0.0
    
    @CircuitBreaker()
    def transfer(self, from_pool: str, to_pool: str, amount: float) -> float:
        """
        Transfer energy between pools.
        
        Args:
            from_pool: Source energy pool
            to_pool: Destination energy pool
            amount: Amount to transfer
            
        Returns:
            Actual amount transferred
        """
        start_time = time.time()
        
        try:
            # Validate pools
            if from_pool not in self.energy_pools or to_pool not in self.energy_pools:
                log.warning(f"Invalid energy pool transfer: {from_pool} -> {to_pool}")
                return 0.0
            
            # Consume from source pool
            actual_consumed = self.consume(from_pool, amount)
            
            # Replenish to destination pool
            actual_replenished = self.replenish(to_pool, actual_consumed)
            
            # Log processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > self.max_processing_history:
                self.processing_times.pop(0)
                
            log.debug(f"Transferred energy ({from_pool} -> {to_pool}, amount={actual_replenished:.3f})")
            return actual_replenished
            
        except Exception as e:
            log.error(f"Error transferring energy: {str(e)}", exc_info=True)
            return 0.0
    
    def _record_energy_change(self, pool: str, delta: float):
        """Record energy change for history and metrics"""
        # Create snapshot
        snapshot = {
            "timestamp": time.time(),
            "pool": pool,
            "delta": delta,
            "cognitive": self.energy_pools[EnergyPool.COGNITIVE.value],
            "metabolic": self.energy_pools[EnergyPool.METABOLIC.value],
            "reserve": self.energy_pools[EnergyPool.RESERVE.value]
        }
        
        # Add to history
        self.energy_history.append(snapshot)
        
        # Keep history bounded
        if len(self.energy_history) > 100:
            self.energy_history.pop(0)
    
    def _update_metrics(self):
        """Update energy metrics based on current state"""
        # Update energy stability
        if len(self.energy_history) >= 10:
            # Calculate variance of energy balance
            energy_balances = [
                h["cognitive"] / (h["metabolic"] + 1e-8)
                for h in self.energy_history[-10:]
            ]
            self.metrics.energy_stability = 1.0 / (1.0 + np.var(energy_balances))
        
        # Update metabolic efficiency
        if len(self.energy_history) > 1:
            # Calculate energy consumption rates
            metabolic_changes = [
                self.energy_history[i]["metabolic"] - self.energy_history[i-1]["metabolic"]
                for i in range(1, len(self.energy_history))
            ]
            self.metrics.metabolic_efficiency = 1.0 - min(1.0, np.mean(np.abs(metabolic_changes)))
        
        # Update reserve utilization
        self.metrics.reserve_utilization = self.energy_pools[EnergyPool.RESERVE.value] / self.max_reserve
        
        # Update energy balance
        cognitive = self.energy_pools[EnergyPool.COGNITIVE.value]
        metabolic = self.energy_pools[EnergyPool.METABOLIC.value]
        self.metrics.energy_balance = cognitive / (metabolic + 1e-8) if metabolic > 0 else float('inf')
        
        # Update processing time
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0.0
        self.metrics.processing_time_ms = avg_processing_time * 1000
    
    def get_snapshot(self) -> EnergySnapshot:
        """Get current energy snapshot for telemetry and reproduction"""
        return EnergySnapshot(
            cognitive=self.energy_pools[EnergyPool.COGNITIVE.value],
            metabolic=self.energy_pools[EnergyPool.METABOLIC.value],
            reserve=self.energy_pools[EnergyPool.RESERVE.value],
            energy_balance=self.metrics.energy_balance
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current energy metrics for monitoring and adaptation"""
        return {
            "energy_stability": self.metrics.energy_stability,
            "metabolic_efficiency": self.metrics.metabolic_efficiency,
            "reserve_utilization": self.metrics.reserve_utilization,
            "energy_balance": self.metrics.energy_balance,
            "processing_time_ms": self.metrics.processing_time_ms,
            "energy_history_size": len(self.energy_history),
            "circuit_breaker_state": self.circuit_breaker.get_state()
        }
    
    def get_ssp_integration_metrics(self) -> Dict[str, float]:
        """
        Get metrics suitable for SSP integration.
        
        Returns energy metrics in a format SSP can use for reward shaping.
        """
        metrics = self.get_metrics()
        
        return {
            "energy_stability": metrics["energy_stability"],
            "metabolic_efficiency": metrics["metabolic_efficiency"],
            "reserve_utilization": metrics["reserve_utilization"],
            "energy_balance": metrics["energy_balance"],
            "processing_efficiency": 1.0 / (1.0 + metrics["processing_time_ms"])
        }
    
    def apply_baseline_consumption(self):
        """Apply baseline energy consumption for system maintenance"""
        # Metabolic baseline consumption
        self.consume(EnergyPool.METABOLIC.value, self.config.metabolic_baseline)
        
        # Cognitive baseline consumption
        self.consume(EnergyPool.COGNITIVE.value, self.config.cognitive_baseline)
        
        # Transfer from reserve if needed
        if self.energy_pools[EnergyPool.METABOLIC.value] < 10.0:
            self.transfer(
                EnergyPool.RESERVE.value,
                EnergyPool.METABOLIC.value,
                self.config.reserve_transfer_rate
            )
        
        log.debug(f"Applied baseline consumption (metabolic={self.config.metabolic_baseline:.3f}, "
                 f"cognitive={self.config.cognitive_baseline:.3f})")
    
    def get_energy_balance(self) -> float:
        """Get current energy balance (cognitive/metabolic)"""
        metabolic = self.energy_pools[EnergyPool.METABOLIC.value]
        if metabolic <= 0:
            return float('inf')
        return self.energy_pools[EnergyPool.COGNITIVE.value] / metabolic
    
    def is_energy_critical(self) -> bool:
        """Check if energy levels are critically low"""
        return (
            self.energy_pools[EnergyPool.METABOLIC.value] < 5.0 or
            self.energy_pools[EnergyPool.COGNITIVE.value] < 5.0
        )