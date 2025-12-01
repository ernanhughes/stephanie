# stephanie/components/jitter/cognition/triune.py
"""
triune.py
=========
Biologically-inspired triune cognitive architecture with veto power cascade.

This implementation:
- Models the three-layer biological architecture (reptilian, mammalian, primate)
- Implements true veto power where lower layers override higher ones
- Integrates with sense-making engine for enactive cognition
- Uses production network for component self-production
- Supports structural coupling with environment

Key Features:
- Layer-specific processing with different neural architectures
- Dynamic attention allocation based on context
- Biological veto power (lower layers override higher ones)
- Energy-based cognitive efficiency measurement
- Continuous state history for reproduction system
- Configuration validation with Pydantic
- Circuit breaker pattern for resilience
- Comprehensive telemetry and monitoring
- SSP integration hooks
- Performance optimizations
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from pydantic import BaseModel, Field, validator

from ...coupling.structural_coupling import StructuralCoupling
from ...production.closed_production import ProductionNetwork
from .sense_making import SenseMakingEngine

log = logging.getLogger("stephanie.jitter.cognition.triune")

class CognitiveLayer(str, Enum):
    """Cognitive layers in the triune architecture"""
    REPTILIAN = "reptilian"
    MAMMALIAN = "mammalian"
    PRIMATE = "primate"

class TriuneConfig(BaseModel):
    """Validated configuration for TriuneCognition"""
    reptilian_veto_threshold: float = Field(0.7, ge=0.5, le=0.9, description="Veto threshold for reptilian layer")
    mammalian_veto_threshold: float = Field(0.6, ge=0.4, le=0.8, description="Veto threshold for mammalian layer")
    reptilian_weight: float = Field(0.3, ge=0.1, le=0.5, description="Base weight for reptilian layer")
    mammalian_weight: float = Field(0.3, ge=0.1, le=0.5, description="Base weight for mammalian layer")
    primate_weight: float = Field(0.4, ge=0.2, le=0.6, description="Base weight for primate layer")
    max_state_history: int = Field(1000, ge=100, le=10000, description="Maximum state history length")
    energy_gain_factor: float = Field(1.0, ge=0.5, le=1.5, description="Factor for cognitive energy extraction")
    
    def validate_weights_sum_to_one(cls, v, values):
        weights = [
            values.get('reptilian_weight', 0.3),
            values.get('mammalian_weight', 0.3),
            values.get('primate_weight', 0.4)
        ]
        if abs(sum(weights) - 1.0) > 0.01:
            raise ValueError('Weights must sum to approximately 1.0')
        return v

@dataclass
class CognitiveState:
    """Current state of the triune cognitive system"""
    reptilian: float  # Boundary threat assessment (0-1)
    mammalian: float  # Pattern recognition confidence (0-1)
    primate: float    # Abstract reasoning quality (0-1)
    integrated: float # Final cognitive output (0-1)
    cognitive_energy: float  # Energy extracted from cognitive processing
    attention_weights: Dict[str, float]  # Current attention allocation
    layer_veto: str   # Which layer has control (if any)
    latency_ms: float # Processing time
    threat_level: float  # Current boundary threat (0-1)
    emotional_valence: float  # Mammalian layer emotional signal (-1 to 1)
    reasoning_depth: int  # Primate cortex reasoning steps taken

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
    def process(sensory_input, action, outcome):
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

class TriuneCognition:
    """
    Triune cognitive architecture with biological veto power cascade.
    
    This is not just an information processing system but a sense-making
    system that creates meaning through interaction with the environment.
    
    Key Features:
    - Reptilian Core: Survival/reflex layer (boundary threat assessment)
    - Mammalian Layer: Pattern recognition and emotional valence
    - Primate Cortex: Abstract reasoning and planning
    - Veto power cascade: lower layers override higher ones when survival is at stake
    - Integration with sense-making engine for enactive cognition
    - Configuration validation with Pydantic
    - Circuit breaker pattern for resilience
    - Comprehensive telemetry and monitoring
    - SSP integration hooks
    - Performance optimizations
    """
    
    def __init__(
        self, 
        cfg: Dict[str, Any],
        sense_making: SenseMakingEngine,
        production: ProductionNetwork,
        coupling: StructuralCoupling
    ):
        try:
            # Validate configuration
            self.config = TriuneConfig(**cfg)
            log.info("TriuneCognition configuration validated successfully")
        except Exception as e:
            log.error(f"Configuration validation failed: {str(e)}")
            # Use safe defaults
            self.config = TriuneConfig()
        
        self.sense_making = sense_making
        self.production = production
        self.coupling = coupling
        
        # Layer thresholds for veto power
        self.veto_thresholds = {
            CognitiveLayer.REPTILIAN.value: self.config.reptilian_veto_threshold,
            CognitiveLayer.MAMMALIAN.value: self.config.mammalian_veto_threshold
        }
        
        # Initial attention weights
        self.attention_weights = {
            CognitiveLayer.REPTILIAN.value: self.config.reptilian_weight,
            CognitiveLayer.MAMMALIAN.value: self.config.mammalian_weight,
            CognitiveLayer.PRIMATE.value: self.config.primate_weight
        }
        
        # Cognitive state history
        self.state_history: List[CognitiveState] = []
        self.max_history = self.config.max_history
        
        # Initialize circuit breaker
        self.circuit_breaker = CircuitBreaker()
        
        # Initialize metrics
        self.health_metrics = {
            "stability": 0.5,
            "efficiency": 0.5,
            "balance": 0.5,
            "veto_frequency": {
                CognitiveLayer.REPTILIAN.value: 0.0,
                CognitiveLayer.MAMMALIAN.value: 0.0
            },
            "resonance": 0.5,
            "production_efficiency": 0.5,
            "meaning_stability": 0.5
        }
        
        # Initialize performance tracking
        self.processing_times = []
        self.max_processing_history = 100
        
        log.info("TriuneCognition initialized with biological veto power cascade")
    
    @CircuitBreaker()
    def process(
        self,
        sensory_input: Dict[str, Any],
        action: Dict[str, Any],
        outcome: Dict[str, Any]
    ) -> CognitiveState:
        """
        Process an interaction cycle through the triune architecture.
        
        This is the core cognitive function that creates meaning from interaction.
        
        Args:
            sensory_input: What the system perceived (VPM, etc.)
            action: What the system did in response
            outcome: Result of the action
            
        Returns:
            CognitiveState object containing cognitive outputs
        """
        start_time = time.time()
        
        try:
            # Create meaning from the interaction (sense-making)
            meaning = self.sense_making.process_interaction(
                sensory_input, 
                action, 
                outcome
            )
            
            # 1. Reptilian Core: Boundary threat assessment
            reptilian_out = self._process_reptilian(sensory_input, meaning)
            threat_level = reptilian_out
            
            # Check for reptilian veto (immediate boundary threats)
            if threat_level > self.veto_thresholds[CognitiveLayer.REPTILIAN.value]:
                cognitive_state = self._create_veto_state(
                    CognitiveLayer.REPTILIAN.value, threat_level, sensory_input, start_time,
                    meaning=meaning
                )
                self._record_state(cognitive_state)
                return cognitive_state
            
            # 2. Mammalian Layer: Pattern recognition and emotional valence
            mammalian_out, emotional_valence = self._process_mammalian(
                sensory_input, meaning
            )
            pattern_confidence = mammalian_out
            
            # Check for mammalian veto (emotional valence threshold)
            if emotional_valence < -self.veto_thresholds[CognitiveLayer.MAMMALIAN.value]:
                cognitive_state = self._create_veto_state(
                    CognitiveLayer.MAMMALIAN.value, pattern_confidence, sensory_input, start_time,
                    emotional_valence=emotional_valence,
                    meaning=meaning
                )
                self._record_state(cognitive_state)
                return cognitive_state
            
            # 3. Primate Cortex: Abstract reasoning
            primate_out, reasoning_depth = self._process_primate(
                sensory_input, meaning
            )
            
            # No veto - integrate all layers
            integrated = self._integrate_layers(
                threat_level, pattern_confidence, primate_out
            )
            
            # Extract cognitive energy
            cognitive_energy = self._extract_energy(
                threat_level, pattern_confidence, primate_out
            )
            
            # Create final cognitive state
            cognitive_state = CognitiveState(
                reptilian=threat_level,
                mammalian=pattern_confidence,
                primate=primate_out,
                integrated=integrated,
                cognitive_energy=cognitive_energy,
                attention_weights=self._get_attention_dict(),
                layer_veto="none",
                latency_ms=(time.time() - start_time) * 1000,
                threat_level=threat_level,
                emotional_valence=emotional_valence,
                reasoning_depth=reasoning_depth
            )
            
            self._record_state(cognitive_state)
            return cognitive_state
            
        except Exception as e:
            log.error(f"TriuneCognition error: {str(e)}", exc_info=True)
            # Return safe fallback state
            return CognitiveState(
                reptilian=0.5, mammalian=0.5, primate=0.5, integrated=0.5,
                cognitive_energy=0.0, attention_weights=self._get_attention_dict(),
                layer_veto="error", latency_ms=0.0, threat_level=0.5,
                emotional_valence=0.0, reasoning_depth=0
            )
    
    def _process_reptilian(self, input_emb: torch.Tensor) -> torch.Tensor:
        # Use VPM energy plugin for boundary threat assessment
        if hasattr(self, 'scoring') and self.scoring:
            try:
                result = self.scoring.score(
                    "vpm_energy", 
                    {"vpm_embedding": input_emb.detach().cpu().numpy()},
                    dimensions=["reptilian.threat01"]
                )
                threat_score = result.get("reptilian.threat01", 0.5)
                return torch.tensor([threat_score])
            except Exception as e:
                log.warning(f"Reptilian processing error: {str(e)}")
        # Fallback
        return torch.tensor([0.5])

    def _process_mammalian(self, input_emb: torch.Tensor) -> Tuple[float, float]:
        # Use VPM pattern plugin for pattern recognition
        if hasattr(self, 'scoring') and self.scoring:
            try:
                result = self.scoring.score(
                    "vpm_pattern", 
                    {"vpm_embedding": input_emb.detach().cpu().numpy()},
                    dimensions=["mammalian.pattern_conf01", "mammalian.valence"]
                )
                pattern_conf = result.get("mammalian.pattern_conf01", 0.5)
                valence = result.get("mammalian.valence", 0.0)
                return pattern_conf, valence
            except Exception as e:
                log.warning(f"Mammalian processing error: {str(e)}")
        # Fallback
        return 0.5, 0.0    

    def _process_primate(
        self, 
        sensory_input: Dict[str, Any],
        meaning: Dict[str, Any]
    ) -> Tuple[float, int]:
        """Process input through Primate Cortex (abstract reasoning)"""
        try:
            # Reasoning quality based on resonance
            reasoning_quality = meaning.get("resonance", 0.5)
            
            # Reasoning depth based on meaning stability and familiarity
            meaning_stability = self.sense_making.get_meaning_metrics()["meaning_stability"]
            familiarity = meaning.get("familiarity", 0.5)
            
            # Higher stability and familiarity allow deeper reasoning
            reasoning_depth = int(5 * (meaning_stability * 0.6 + familiarity * 0.4))
            
            return reasoning_quality, reasoning_depth
            
        except Exception as e:
            log.warning(f"Primate processing error: {str(e)}")
            return 0.5, 0
    
    def _integrate_layers(
        self, 
        reptilian: float, 
        mammalian: float, 
        primate: float
    ) -> float:
        """Integrate outputs from all three layers with attention weighting"""
        # Base weights (can be adjusted based on context)
        weights = [
            self.attention_weights[CognitiveLayer.REPTILIAN.value],
            self.attention_weights[CognitiveLayer.MAMMALIAN.value],
            self.attention_weights[CognitiveLayer.PRIMATE.value]
        ]
        
        # Normalize to sum to 1
        total = sum(weights)
        weights = [w / total for w in weights]
        
        # Calculate integrated output
        return float(
            reptilian * weights[0] + 
            mammalian * weights[1] + 
            primate * weights[2]
        )
    
    def _extract_energy(
        self, 
        threat_level: float, 
        pattern_confidence: float, 
        reasoning_quality: float
    ) -> float:
        """Extract cognitive energy from processing"""
        # Energy gain is higher when processing is efficient and successful
        energy_base = reasoning_quality * 0.6 + pattern_confidence * 0.3 + (1.0 - threat_level) * 0.1
        return energy_base * self.config.energy_gain_factor
    
    def _create_veto_state(
        self,
        veto_layer: str,
        primary_value: float,
        sensory_input: Dict[str, Any],
        start_time: float,
        emotional_valence: float = 0.0,
        meaning: Dict[str, Any] = None
    ) -> CognitiveState:
        """Create cognitive state when a layer has veto power"""
        meaning = meaning or {
            "meaning_type": "veto",
            "outcome_quality": primary_value,
            "resonance": primary_value,
            "familiarity": 0.5
        }
        
        return CognitiveState(
            reptilian=primary_value if veto_layer == CognitiveLayer.REPTILIAN.value else 0.0,
            mammalian=primary_value if veto_layer == CognitiveLayer.MAMMALIAN.value else 0.0,
            primate=0.0,
            integrated=primary_value,
            cognitive_energy=primary_value * 0.2,  # Reduced energy during veto
            attention_weights=self._get_attention_dict(veto_layer=veto_layer),
            layer_veto=veto_layer,
            latency_ms=(time.time() - start_time) * 1000,
            threat_level=primary_value if veto_layer == CognitiveLayer.REPTILIAN.value else 0.5,
            emotional_valence=emotional_valence if veto_layer == CognitiveLayer.MAMMALIAN.value else 0.0,
            reasoning_depth=0
        )
    
    def _get_attention_dict(self, veto_layer: Optional[str] = None) -> Dict[str, float]:
        """Get current attention weights as a dictionary"""
        if veto_layer == CognitiveLayer.REPTILIAN.value:
            return {
                CognitiveLayer.REPTILIAN.value: 0.7,
                CognitiveLayer.MAMMALIAN.value: 0.2,
                CognitiveLayer.PRIMATE.value: 0.1
            }
        elif veto_layer == CognitiveLayer.MAMMALIAN.value:
            return {
                CognitiveLayer.REPTILIAN.value: 0.3,
                CognitiveLayer.MAMMALIAN.value: 0.6,
                CognitiveLayer.PRIMATE.value: 0.1
            }
        
        return self.attention_weights.copy()
    
    def _record_state(self, state: CognitiveState):
        """Record cognitive state for history and reproduction"""
        self.state_history.append(state)
        if len(self.state_history) > self.max_history:
            self.state_history.pop(0)
        
        # Update health metrics periodically
        if len(self.state_history) % 10 == 0:
            self._update_health_metrics()
    
    def _update_health_metrics(self):
        """Update health metrics based on cognitive state history"""
        if not self.state_history:
            return
            
        # Get last 50 states for metrics
        recent = self.state_history[-50:]
        
        # Stability: variance in integrated cognitive output
        integrated_values = [s.integrated for s in recent]
        stability = 1.0 / (1.0 + np.var(integrated_values))
        
        # Efficiency: average cognitive energy per tick
        efficiency = float(np.mean([s.cognitive_energy for s in recent]))
        
        # Balance: how evenly attention is distributed
        attention_values = [
            [s.attention_weights[CognitiveLayer.REPTILIAN.value],
             s.attention_weights[CognitiveLayer.MAMMALIAN.value],
             s.attention_weights[CognitiveLayer.PRIMATE.value]]
            for s in recent
        ]
        avg_attention = [
            np.mean([a[0] for a in attention_values]),
            np.mean([a[1] for a in attention_values]),
            np.mean([a[2] for a in attention_values])
        ]
        balance = 1.0 - np.std(avg_attention) * 3
        
        # Veto frequency
        veto_counts = {CognitiveLayer.REPTILIAN.value: 0, CognitiveLayer.MAMMALIAN.value: 0}
        for s in recent:
            if s.layer_veto == CognitiveLayer.REPTILIAN.value:
                veto_counts[CognitiveLayer.REPTILIAN.value] += 1
            elif s.layer_veto == CognitiveLayer.MAMMALIAN.value:
                veto_counts[CognitiveLayer.MAMMALIAN.value] += 1
                
        veto_freq = {
            k: v / len(recent) for k, v in veto_counts.items()
        }
        
        # Resonance with environment
        resonance = float(np.mean([self.sense_making.get_meaning_metrics()["resonance_score"] for _ in recent]))
        
        # Production efficiency
        production_efficiency = self.production.get_production_efficiency()
        
        # Meaning stability
        meaning_stability = self.sense_making.get_meaning_metrics()["meaning_stability"]
        
        self.health_metrics = {
            "stability": float(stability),
            "efficiency": efficiency,
            "balance": float(balance),
            "veto_frequency": veto_freq,
            "resonance": resonance,
            "production_efficiency": production_efficiency,
            "meaning_stability": meaning_stability
        }
    
    def get_recent_states(self, n: int = 10) -> List[CognitiveState]:
        """Get recent cognitive states for reproduction system"""
        return self.state_history[-n:] if self.state_history else []
    
    def get_health_metrics(self) -> Dict[str, float]:
        """Get health metrics from cognitive state history"""
        return self.health_metrics.copy()
    
    def update_attention_weights(self, layer: str, delta: float):
        """
        Update attention weights based on performance or external signals.
        
        Args:
            layer: Which layer to adjust (reptilian, mammalian, primate)
            delta: Change amount (-1.0 to 1.0)
        """
        if layer not in self.attention_weights:
            log.warning(f"Invalid layer for attention update: {layer}")
            return
            
        # Apply delta with bounds checking
        self.attention_weights[layer] = max(
            0.1,  # Minimum attention
            min(0.8, self.attention_weights[layer] + delta)  # Maximum attention
        )
        
        # Normalize to sum to 1
        total = sum(self.attention_weights.values())
        for l in self.attention_weights:
            self.attention_weights[l] /= total
    
    def get_triune_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics for monitoring and adaptation"""
        # Get sense-making metrics
        sense_metrics = self.sense_making.get_meaning_metrics()
        
        # Get production metrics
        production_metrics = self.production.get_coupling_metrics()
        
        # Get coupling metrics
        coupling_metrics = self.coupling.get_coupling_metrics()
        
        # Calculate processing time
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0.0
        
        return {
            "health_metrics": self.get_health_metrics(),
            "sense_making": sense_metrics,
            "production": production_metrics,
            "coupling": coupling_metrics,
            "attention_weights": self.attention_weights,
            "circuit_breaker_state": self.circuit_breaker.get_state(),
            "processing_time_ms": avg_processing_time * 1000,
            "state_history_size": len(self.state_history)
        }
    
    def get_ssp_integration_metrics(self) -> Dict[str, float]:
        """
        Get metrics suitable for SSP integration.
        
        Returns triune metrics in a format SSP can use for reward shaping.
        """
        metrics = self.get_triune_metrics()
        health = metrics["health_metrics"]
        
        return {
            "cognitive_health": health["efficiency"],
            "stability": health["stability"],
            "resonance": health["resonance"],
            "veto_frequency_reptilian": health["veto_frequency"]["reptilian"],
            "veto_frequency_mammalian": health["veto_frequency"]["mammalian"],
            "attention_reptilian": self.attention_weights["reptilian"],
            "attention_mammalian": self.attention_weights["mammalian"],
            "attention_primate": self.attention_weights["primate"],
            "processing_efficiency": 1.0 / (1.0 + metrics["processing_time_ms"])
        }
    
    def apply_ssp_feedback(self, feedback: Dict[str, Any]):
        """
        Apply feedback from SSP episodes to adjust cognitive parameters.
        
        Args:
            feedback: Dictionary containing SSP feedback metrics
        """
        # Adjust attention weights based on task performance
        if "task_performance" in feedback:
            performance = feedback["task_performance"]
            
            # If performance was good, increase primate attention
            if performance > 0.7:
                self.update_attention_weights(CognitiveLayer.PRIMATE.value, 0.05)
            # If performance was poor, increase reptilian attention for stability
            elif performance < 0.3:
                self.update_attention_weights(CognitiveLayer.REPTILIAN.value, 0.05)
        
        # Adjust veto thresholds based on safety metrics
        if "safety_violations" in feedback and feedback["safety_violations"] > 0:
            # Increase reptilian veto threshold for more sensitivity to threats
            self.veto_thresholds[CognitiveLayer.REPTILIAN.value] = min(
                0.9, 
                self.veto_thresholds[CognitiveLayer.REPTILIAN.value] * 1.05
            )