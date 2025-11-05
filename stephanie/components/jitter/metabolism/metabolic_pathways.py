# stephanie/components/jitter/metabolism/metabolic_pathways.py
"""
metabolic_pathways.py
=====================
Implementation of metabolic pathways for energy conversion.

This module implements the organizationally closed production of energy pathways:
- Conversion between cognitive, metabolic, and reserve energy pools
- Pathway efficiency and adaptation
- Configuration validation with Pydantic
- Circuit breaker pattern for resilience
- Comprehensive telemetry and monitoring
- SSP integration hooks
- Performance optimizations
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from pydantic import BaseModel, Field, validator

from .energy import EnergyPool, EnergyPools

log = logging.getLogger("stephanie.jitter.metabolism.pathways")

class PathwayType(str, Enum):
    """Types of metabolic pathways"""
    COGNITIVE_TO_METABOLIC = "cognitive_to_metabolic"
    METABOLIC_TO_RESERVE = "metabolic_to_reserve"
    RESERVE_TO_METABOLIC = "reserve_to_metabolic"
    METABOLIC_TO_COGNITIVE = "metabolic_to_cognitive"

class PathwayConfig(BaseModel):
    """Validated configuration for MetabolicPathways"""
    initial_efficiency: float = Field(0.8, ge=0.5, le=1.0, description="Initial pathway efficiency")
    learning_rate: float = Field(0.01, ge=0.001, le=0.05, description="Learning rate for adaptation")
    max_efficiency: float = Field(0.95, ge=0.8, le=0.99, description="Maximum pathway efficiency")
    min_efficiency: float = Field(0.5, ge=0.3, le=0.7, description="Minimum pathway efficiency")
    adaptation_smoothing: float = Field(0.9, ge=0.7, le=0.95, description="Smoothing factor for adaptation")
    pathway_network_size: int = Field(64, ge=32, le=128, description="Size of pathway neural network")
    
    @validator('min_efficiency')
    def validate_min_max_efficiency(cls, v, values):
        if 'max_efficiency' in values and v >= values['max_efficiency']:
            raise ValueError('min_efficiency must be less than max_efficiency')
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
    def convert(pathway, amount):
        # Conversion logic here
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

class PathwayNetwork(nn.Module):
    """Neural network for modeling a metabolic pathway"""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

@dataclass
class PathwayRecord:
    """Record of a pathway operation for learning and adaptation"""
    id: str = field(default_factory=lambda: f"path_{int(time.time())}_{np.random.randint(1000)}")
    timestamp: float = field(default_factory=time.time)
    pathway: str = PathwayType.COGNITIVE_TO_METABOLIC.value
    input_amount: float = 0.0
    output_amount: float = 0.0
    efficiency: float = 0.0
    context: Dict[str, Any] = field(default_factory=dict)
    success: bool = False

@dataclass
class PathwayMetrics:
    """Metrics for metabolic pathway performance"""
    pathway_efficiency: float = 0.5
    adaptation_rate: float = 0.0
    energy_conversion_rate: float = 0.5
    pathway_stability: float = 0.5
    processing_time_ms: float = 0.0

class MetabolicPathways:
    """
    Manages metabolic pathways for energy conversion in the autopoietic system.
    
    Key Features:
    - Organizationally closed production of energy pathways
    - Conversion between cognitive, metabolic, and reserve energy pools
    - Pathway efficiency and adaptation
    - Configuration validation with Pydantic
    - Circuit breaker pattern for resilience
    - Comprehensive telemetry and monitoring
    - SSP integration hooks
    - Performance optimizations
    """
    
    def __init__(
        self,
        cfg: Dict[str, Any],
        energy_pools: EnergyPools
    ):
        try:
            # Validate configuration
            self.config = PathwayConfig(**cfg)
            log.info("MetabolicPathways configuration validated successfully")
        except Exception as e:
            log.error(f"Configuration validation failed: {str(e)}")
            # Use safe defaults
            self.config = PathwayConfig()
        
        self.energy_pools = energy_pools
        
        # Initialize pathway efficiencies
        self.efficiencies = {
            PathwayType.COGNITIVE_TO_METABOLIC.value: self.config.initial_efficiency,
            PathwayType.METABOLIC_TO_RESERVE.value: self.config.initial_efficiency,
            PathwayType.RESERVE_TO_METABOLIC.value: self.config.initial_efficiency,
            PathwayType.METABOLIC_TO_COGNITIVE.value: self.config.initial_efficiency
        }
        
        # Initialize pathway networks
        self.pathways = {
            PathwayType.COGNITIVE_TO_METABOLIC.value: PathwayNetwork(1, self.config.pathway_network_size, 1),
            PathwayType.METABOLIC_TO_RESERVE.value: PathwayNetwork(1, self.config.pathway_network_size, 1),
            PathwayType.RESERVE_TO_METABOLIC.value: PathwayNetwork(1, self.config.pathway_network_size, 1),
            PathwayType.METABOLIC_TO_COGNITIVE.value: PathwayNetwork(1, self.config.pathway_network_size, 1)
        }
        
        # Initialize history
        self.pathway_history: List[PathwayRecord] = []
        
        # Initialize circuit breaker
        self.circuit_breaker = CircuitBreaker()
        
        # Initialize metrics
        self.metrics = PathwayMetrics()
        
        # Initialize performance tracking
        self.processing_times = []
        self.max_processing_history = 100
        
        log.info("MetabolicPathways initialized with organizationally closed production features")
    
    @CircuitBreaker()
    def convert(
        self,
        pathway_type: str,
        amount: float,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Convert energy through a metabolic pathway.
        
        Args:
            pathway_type: Type of pathway to use
            amount: Amount of energy to convert
            context: Additional context for the conversion
            
        Returns:
            Amount of energy after conversion
        """
        start_time = time.time()
        
        try:
            # Validate pathway type
            if pathway_type not in self.efficiencies:
                log.warning(f"Invalid pathway type: {pathway_type}")
                return 0.0
            
            # Get current efficiency
            efficiency = self.efficiencies[pathway_type]
            
            # Calculate output amount
            output_amount = amount * efficiency
            
            # Record for history
            self._record_pathway_operation(
                pathway_type,
                amount,
                output_amount,
                efficiency,
                context or {}
            )
            
            # Update metrics
            self._update_metrics()
            
            # Log processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > self.max_processing_history:
                self.processing_times.pop(0)
                
            log.debug(f"Converted energy ({pathway_type}, input={amount:.3f}, output={output_amount:.3f}, "
                     f"efficiency={efficiency:.3f})")
            return output_amount
            
        except Exception as e:
            log.error(f"Error converting energy: {str(e)}", exc_info=True)
            return 0.0
    
    def _record_pathway_operation(
        self,
        pathway_type: str,
        input_amount: float,
        output_amount: float,
        efficiency: float,
        context: Dict[str, Any]
    ):
        """Record pathway operation for learning and adaptation"""
        # Create record
        record = PathwayRecord(
            pathway=pathway_type,
            input_amount=input_amount,
            output_amount=output_amount,
            efficiency=efficiency,
            context=context,
            success=True  # Will be updated later if needed
        )
        
        # Add to history
        self.pathway_history.append(record)
        
        # Keep history bounded
        if len(self.pathway_history) > self.config.max_history:
            self.pathway_history.pop(0)
    
    def _update_metrics(self):
        """Update pathway metrics based on history"""
        # Update pathway efficiency
        if len(self.pathway_history) >= 10:
            efficiencies = [p.efficiency for p in self.pathway_history[-10:]]
            self.metrics.pathway_efficiency = np.mean(efficiencies)
        
        # Update adaptation rate
        if len(self.pathway_history) >= 2:
            efficiency_changes = [
                self.pathway_history[i].efficiency - self.pathway_history[i-1].efficiency
                for i in range(1, len(self.pathway_history))
            ]
            self.metrics.adaptation_rate = np.mean(np.abs(efficiency_changes))
        
        # Update energy conversion rate
        if len(self.pathway_history) > 0:
            total_input = sum(p.input_amount for p in self.pathway_history)
            total_output = sum(p.output_amount for p in self.pathway_history)
            self.metrics.energy_conversion_rate = total_output / (total_input + 1e-8)
        
        # Update pathway stability
        if len(self.pathway_history) >= 10:
            efficiency_values = [p.efficiency for p in self.pathway_history[-10:]]
            self.metrics.pathway_stability = 1.0 / (1.0 + np.var(efficiency_values))
        
        # Update processing time
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0.0
        self.metrics.processing_time_ms = avg_processing_time * 1000
    
    def adapt_pathways(self, performance: float):
        """
        Adapt pathway efficiencies based on system performance.
        
        Args:
            performance: System performance metric (0-1)
        """
        # Update all pathway efficiencies based on performance
        for pathway in self.efficiencies:
            # If performance is good, increase efficiency (with bounds)
            if performance > 0.7:
                self.efficiencies[pathway] = min(
                    self.config.max_efficiency,
                    self.efficiencies[pathway] * (1.0 + self.config.learning_rate)
                )
            # If performance is poor, decrease efficiency (with bounds)
            elif performance < 0.3:
                self.efficiencies[pathway] = max(
                    self.config.min_efficiency,
                    self.efficiencies[pathway] * (1.0 - self.config.learning_rate)
                )
            
            # Apply smoothing
            self.efficiencies[pathway] = (
                self.efficiencies[pathway] * self.config.adaptation_smoothing +
                self.config.initial_efficiency * (1.0 - self.config.adaptation_smoothing)
            )
        
        log.debug(f"Adapted pathways (performance={performance:.3f}, "
                 f"efficiencies={self.efficiencies})")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current pathway metrics for monitoring and adaptation"""
        return {
            "pathway_efficiency": self.metrics.pathway_efficiency,
            "adaptation_rate": self.metrics.adaptation_rate,
            "energy_conversion_rate": self.metrics.energy_conversion_rate,
            "pathway_stability": self.metrics.pathway_stability,
            "processing_time_ms": self.metrics.processing_time_ms,
            "pathway_history_size": len(self.pathway_history),
            "circuit_breaker_state": self.circuit_breaker.get_state()
        }
    
    def get_ssp_integration_metrics(self) -> Dict[str, float]:
        """
        Get metrics suitable for SSP integration.
        
        Returns pathway metrics in a format SSP can use for reward shaping.
        """
        metrics = self.get_metrics()
        
        return {
            "pathway_efficiency": metrics["pathway_efficiency"],
            "adaptation_rate": metrics["adaptation_rate"],
            "energy_conversion_rate": metrics["energy_conversion_rate"],
            "pathway_stability": metrics["pathway_stability"],
            "processing_efficiency": 1.0 / (1.0 + metrics["processing_time_ms"])
        }
    
    def evaluate_conversion_success(
        self,
        pathway_id: str,
        expected_output: float,
        actual_output: float
    ) -> bool:
        """
        Evaluate whether pathway conversion was successful.
        
        Args:
            pathway_id: ID of the pathway operation to evaluate
            expected_output: Expected output amount
            actual_output: Actual output amount
            
        Returns:
            True if conversion was successful, False otherwise
        """
        # Find the pathway record
        record = next((p for p in self.pathway_history if p.id == pathway_id), None)
        if not record:
            log.warning(f"Pathway record not found: {pathway_id}")
            return False
        
        # Determine success based on output match
        success = abs(actual_output - expected_output) < 0.1 * expected_output
        
        # Update record
        record.success = success
        
        # Update metrics
        if success:
            self.metrics.pathway_efficiency = min(1.0, self.metrics.pathway_efficiency + 0.01)
        else:
            self.metrics.pathway_efficiency = max(0.0, self.metrics.pathway_efficiency - 0.01)
        
        log.debug(f"Pathway evaluation: {pathway_id} - success={success}")
        return success
    
    def get_recent_conversions(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get recent pathway conversions for analysis or reporting"""
        return [
            {
                "id": p.id,
                "timestamp": p.timestamp,
                "pathway": p.pathway,
                "input_amount": p.input_amount,
                "output_amount": p.output_amount,
                "efficiency": p.efficiency,
                "context": p.context,
                "success": p.success
            }
            for p in self.pathway_history[-n:]
        ]
    
    def get_optimal_pathway_profile(
        self,
        context: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Get optimal pathway profile for given context based on historical data.
        
        Args:
            context: Current context for which to determine optimal profile
            
        Returns:
            Optimal pathway efficiencies for the context
        """
        if len(self.pathway_history) < 10:
            return self.efficiencies.copy()
        
        # Find similar contexts
        similar = []
        for record in self.pathway_history:
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
        
        # If no similar contexts, return current efficiencies
        if not similar:
            return self.efficiencies.copy()
        
        # Calculate weighted average of pathway efficiencies
        total_weight = 0.0
        avg_efficiencies = {p: 0.0 for p in self.efficiencies}
        
        for record, weight in similar:
            total_weight += weight
            avg_efficiencies[record.pathway] += record.efficiency * weight
        
        # Normalize
        if total_weight > 0:
            for pathway in avg_efficiencies:
                avg_efficiencies[pathway] /= total_weight
        
        # Ensure efficiencies are within bounds
        for pathway in avg_efficiencies:
            avg_efficiencies[pathway] = max(
                self.config.min_efficiency,
                min(self.config.max_efficiency, avg_efficiencies[pathway])
            )
        
        return avg_efficiencies
    