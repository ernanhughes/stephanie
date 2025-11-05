# stephanie/components/jitter/production/closed_production.py
"""
closed_production.py
====================
Implementation of organizationally closed production networks with enhanced features.

This module implements Maturana & Varela's principle that an autopoietic system
must recursively produce the components that define the system itself, with:
- Configuration validation
- Circuit breaker resilience
- Quality control mechanisms
- SSP integration hooks
- Crisis management
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
from pydantic import BaseModel, Field, root_validator, validator

log = logging.getLogger("stephanie.jitter.production")

class ComponentType(str, Enum):
    MEMBRANE = "membrane"
    METABOLIC_PATHWAYS = "metabolic_pathways"
    COGNITIVE_STRUCTURES = "cognitive_structures"
    REGULATORY_MECHANISMS = "regulatory_mechanisms"

class ProductionConfig(BaseModel):
    """Validated configuration for ProductionNetwork"""
    max_history: int = Field(1000, ge=100, le=10000, description="Maximum history length")
    min_production_rate: float = Field(0.01, ge=0.0, le=0.1, description="Minimum production rate")
    max_production_rate: float = Field(0.5, ge=0.1, le=1.0, description="Maximum production rate")
    crisis_threshold: float = Field(0.8, ge=0.5, le=0.95, description="Crisis threshold for production")
    quality_threshold: float = Field(0.6, ge=0.3, le=0.9, description="Minimum quality threshold")
    heritage_preservation: bool = Field(True, description="Whether to preserve heritage")
    performance_history_size: int = Field(50, ge=10, le=100, description="Performance history size")
    
    @validator('min_production_rate')
    def validate_min_max_production_rate(cls, v, values):
        if 'max_production_rate' in values and v >= values['max_production_rate']:
            raise ValueError('min_production_rate must be less than max_production_rate')
        return v

class CircuitBreaker:
    """Circuit breaker pattern for critical service dependencies"""
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.last_failure_time = 0.0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.logger = logging.getLogger(f"{__name__}.circuit_breaker")
    
    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if self.state == "OPEN":
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.logger.info("Circuit breaker transitioning to HALF_OPEN state")
                    self.state = "HALF_OPEN"
                else:
                    log.warning("Circuit breaker is OPEN - skipping call")
                    return None
            
            try:
                result = func(*args, **kwargs)
                if self.state == "HALF_OPEN":
                    self.logger.info("Circuit breaker transitioning to CLOSED state")
                    self.state = "CLOSED"
                    self.failures = 0
                return result
            except Exception as e:
                self.failures += 1
                self.last_failure_time = time.time()
                self.logger.error(f"Service failure: {str(e)}, failures: {self.failures}")
                
                if self.failures >= self.failure_threshold:
                    log.warning("Circuit breaker transitioning to OPEN state")
                    self.state = "OPEN"
                raise

class CrisisLevel(str, Enum):
    NORMAL = "normal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class CrisisResponse:
    level: CrisisLevel
    actions: List[str]
    priority: str
    recommended_duration: float

class CrisisManager:
    """Enhanced crisis management system with multi-level response protocols"""
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.crisis_levels = {
            CrisisLevel.NORMAL: 0.0,
            CrisisLevel.LOW: 0.3,
            CrisisLevel.MEDIUM: 0.6,
            CrisisLevel.HIGH: 0.8,
            CrisisLevel.CRITICAL: 0.95
        }
        self.response_actions = {
            CrisisLevel.LOW: ["adjust_homeostasis", "monitor_closely"],
            CrisisLevel.MEDIUM: ["conserve_energy", "adjust_attention_weights"],
            CrisisLevel.HIGH: ["fortify_boundary", "reduce_cognitive_load", "alert_monitoring"],
            CrisisLevel.CRITICAL: ["initiate_apoptosis", "preserve_legacy", "alert_emergency"]
        }
        self.crisis_history = []
        self.max_crisis_history = 100
    
    def assess_crisis(self, production_metrics: Dict[str, Any]) -> CrisisResponse:
        """Comprehensive crisis assessment based on multiple metrics"""
        crisis_level = production_metrics.get("crisis_level", 0.0)
        health = production_metrics.get("health", 0.5)
        stability = production_metrics.get("stability", 0.7)
        
        # Determine crisis level
        level = CrisisLevel.NORMAL
        for crisis_level_enum, threshold in sorted(
            self.crisis_levels.items(), key=lambda x: x[1], reverse=True
        ):
            if crisis_level >= threshold:
                level = crisis_level_enum
                break
        
        # Determine priority
        priority = "low"
        if level == CrisisLevel.CRITICAL:
            priority = "highest"
        elif level == CrisisLevel.HIGH:
            priority = "high"
        elif level == CrisisLevel.MEDIUM:
            priority = "medium"
        
        # Calculate recommended duration based on severity
        recommended_duration = 0.0
        if level == CrisisLevel.CRITICAL:
            recommended_duration = 5.0  # Immediate action required
        elif level == CrisisLevel.HIGH:
            recommended_duration = 30.0
        elif level == CrisisLevel.MEDIUM:
            recommended_duration = 120.0
        
        return CrisisResponse(
            level=level,
            actions=self.response_actions.get(level, []),
            priority=priority,
            recommended_duration=recommended_duration
        )
    
    def execute_response(self, response: CrisisResponse, core_system: Any) -> Dict[str, Any]:
        """Execute crisis response actions with monitoring"""
        results = {"actions_executed": [], "status": "success"}
        
        for action in response.actions:
            try:
                if action == "conserve_energy":
                    core_system.energy.adjust_pathway_rates(0.5)
                    results["actions_executed"].append(f"{action}: success")
                elif action == "fortify_boundary":
                    core_system.membrane.thickness = min(1.0, core_system.membrane.thickness + 0.1)
                    results["actions_executed"].append(f"{action}: success")
                elif action == "reduce_cognitive_load":
                    if hasattr(core_system, 'triune'):
                        core_system.triune.update_attention_weights(-0.2)
                        results["actions_executed"].append(f"{action}: success")
                elif action == "initiate_apoptosis":
                    if hasattr(core_system, 'apoptosis_system'):
                        core_system.apoptosis_system.initiate()
                        results["actions_executed"].append(f"{action}: success")
            except Exception as e:
                results["status"] = "partial_failure"
                results["actions_executed"].append(f"{action}: failed - {str(e)}")
                log.error(f"Crisis response action failed: {action} - {str(e)}")
        
        # Record for learning
        self.crisis_history.append({
            "timestamp": time.time(),
            "response": response,
            "results": results,
            "core_state_snapshot": self._capture_core_snapshot(core_system)
        })
        
        # Keep history bounded
        if len(self.crisis_history) > self.max_crisis_history:
            self.crisis_history.pop(0)
        
        return results
    
    def _capture_core_snapshot(self, core_system: Any) -> Dict[str, Any]:
        """Capture a snapshot of the core system state"""
        snapshot = {}
        if hasattr(core_system, 'energy'):
            snapshot["energy"] = {
                "cognitive": core_system.energy.level("cognitive"),
                "metabolic": core_system.energy.level("metabolic"),
                "reserve": core_system.energy.level("reserve")
            }
        if hasattr(core_system, 'membrane'):
            snapshot["membrane"] = {
                "integrity": core_system.membrane.integrity,
                "thickness": core_system.membrane.thickness
            }
        return snapshot

class ProductionNetwork:
    """
    Network of processes that recursively produces the system's components with enhanced features.
    
    Key Features:
    - Organizationally closed production with validated configuration
    - Circuit breaker pattern for resilience
    - Multi-level crisis management
    - Quality-controlled component production
    - Performance-optimized processing
    - SSP integration hooks
    """
    
    def __init__(self, cfg: Dict[str, Any]):
        try:
            # Validate configuration
            self.config = ProductionConfig(**cfg)
            log.info("ProductionNetwork configuration validated successfully")
        except Exception as e:
            log.error(f"Configuration validation failed: {str(e)}")
            # Use safe defaults
            self.config = ProductionConfig()
        
        # Initialize components with validated rates
        self.components = {
            ComponentType.MEMBRANE: {
                "production_rate": self._clamp_production_rate(0.1),
                "dependencies": [ComponentType.METABOLIC_PATHWAYS],
                "min_rate": self.config.min_production_rate,
                "max_rate": self.config.max_production_rate
            },
            ComponentType.METABOLIC_PATHWAYS: {
                "production_rate": self._clamp_production_rate(0.2),
                "dependencies": [ComponentType.COGNITIVE_STRUCTURES],
                "min_rate": self.config.min_production_rate,
                "max_rate": self.config.max_production_rate
            },
            ComponentType.COGNITIVE_STRUCTURES: {
                "production_rate": self._clamp_production_rate(0.15),
                "dependencies": [ComponentType.MEMBRANE],
                "min_rate": self.config.min_production_rate,
                "max_rate": self.config.max_production_rate
            },
            ComponentType.REGULATORY_MECHANISMS: {
                "production_rate": self._clamp_production_rate(0.05),
                "dependencies": [ComponentType.COGNITIVE_STRUCTURES],
                "min_rate": self.config.min_production_rate,
                "max_rate": self.config.max_production_rate
            }
        }
        
        # Initialize history with validated size
        self.production_history = []
        self.max_history = self.config.max_history
        
        # Initialize circuit breaker
        self.circuit_breaker = CircuitBreaker()
        
        # Initialize crisis manager
        self.crisis_manager = CrisisManager(self.config)
        
        # Initialize performance tracking
        self.performance_history = []
        self.max_performance_history = self.config.performance_history_size
        
        log.info("ProductionNetwork initialized with enhanced features")
    
    def _clamp_production_rate(self, rate: float) -> float:
        """Clamp production rate within configured bounds"""
        return max(
            self.config.min_production_rate,
            min(self.config.max_production_rate, rate)
        )
    
    @CircuitBreaker()
    def update_production_rates(self, system_state: Dict[str, Any]) -> Dict[str, float]:
        """
        Update production rates based on current system state with circuit protection.
        
        This implements the recursive nature of autopoiesis - the system's state
        determines how it produces its own components.
        
        Returns:
            Dictionary of updated production rates
        """
        updated_rates = {}
        
        try:
            # Membrane production depends on metabolic energy
            metabolic_energy = system_state["energy"]["metabolic"]
            new_rate = 0.1 * (metabolic_energy / 100.0)
            self.components[ComponentType.MEMBRANE]["production_rate"] = self._clamp_production_rate(new_rate)
            updated_rates[ComponentType.MEMBRANE] = new_rate
            
            # Cognitive structures depend on membrane integrity
            membrane_integrity = system_state["membrane"]["integrity"]
            new_rate = 0.15 * membrane_integrity
            self.components[ComponentType.COGNITIVE_STRUCTURES]["production_rate"] = self._clamp_production_rate(new_rate)
            updated_rates[ComponentType.COGNITIVE_STRUCTURES] = new_rate
            
            # Regulatory mechanisms depend on cognitive coherence
            cognitive_coherence = system_state["cognitive"]["coherence"]
            new_rate = 0.05 * cognitive_coherence
            self.components[ComponentType.REGULATORY_MECHANISMS]["production_rate"] = self._clamp_production_rate(new_rate)
            updated_rates[ComponentType.REGULATORY_MECHANISMS] = new_rate
            
            # Metabolic pathways depend on cognitive structures
            cognitive_structures = system_state["cognitive"]["structures"]
            new_rate = 0.2 * cognitive_structures
            self.components[ComponentType.METABOLIC_PATHWAYS]["production_rate"] = self._clamp_production_rate(new_rate)
            updated_rates[ComponentType.METABOLIC_PATHWAYS] = new_rate
            
            return updated_rates
            
        except KeyError as e:
            log.error(f"Missing system state key in update_production_rates: {str(e)}")
            # Return current rates as fallback
            return {k: v["production_rate"] for k, v in self.components.items()}
        except Exception as e:
            log.error(f"Error updating production rates: {str(e)}")
            raise
    
    @CircuitBreaker()
    def produce_components(self, system_state: Dict[str, Any]) -> Dict[str, float]:
        """
        Execute production cycle to generate system components with circuit protection.
        
        Returns changes to system state from production processes.
        """
        try:
            # Update production rates first
            self.update_production_rates(system_state)
            
            changes = {
                ComponentType.MEMBRANE: 0.0,
                ComponentType.METABOLIC_PATHWAYS: 0.0,
                ComponentType.COGNITIVE_STRUCTURES: 0.0,
                ComponentType.REGULATORY_MECHANISMS: 0.0
            }
            
            # Membrane production (depends on metabolic pathways)
            metabolic_pathways = system_state["metabolic"]["pathways"]
            changes[ComponentType.MEMBRANE] = (
                self.components[ComponentType.MEMBRANE]["production_rate"] * metabolic_pathways
            )
            
            # Metabolic pathways production (depends on cognitive structures)
            cognitive_structures = system_state["cognitive"]["structures"]
            changes[ComponentType.METABOLIC_PATHWAYS] = (
                self.components[ComponentType.METABOLIC_PATHWAYS]["production_rate"] * cognitive_structures
            )
            
            # Cognitive structures production (depends on membrane integrity)
            membrane_integrity = system_state["membrane"]["integrity"]
            changes[ComponentType.COGNITIVE_STRUCTURES] = (
                self.components[ComponentType.COGNITIVE_STRUCTURES]["production_rate"] * membrane_integrity
            )
            
            # Regulatory mechanisms production (depends on cognitive structures)
            changes[ComponentType.REGULATORY_MECHANISMS] = (
                self.components[ComponentType.REGULATORY_MECHANISMS]["production_rate"] * cognitive_structures
            )
            
            # Record for history
            self.production_history.append({
                "time": system_state["time"],
                "changes": {k.value: v for k, v in changes.items()},
                "rates": {k.value: v["production_rate"] for k, v in self.components.items()},
                "system_state": self._extract_relevant_state(system_state)
            })
            
            # Keep history bounded
            if len(self.production_history) > self.max_history:
                self.production_history.pop(0)
                
            return changes
            
        except Exception as e:
            log.error(f"Error in produce_components: {str(e)}")
            # Return zero changes as fallback
            return {k: 0.0 for k in changes.keys()}
    
    def _extract_relevant_state(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract only relevant state for history to reduce memory usage"""
        return {
            "energy": {
                "metabolic": system_state["energy"]["metabolic"],
                "cognitive": system_state["energy"]["cognitive"]
            },
            "membrane": {
                "integrity": system_state["membrane"]["integrity"],
                "thickness": system_state["membrane"]["thickness"]
            },
            "cognitive": {
                "coherence": system_state["cognitive"]["coherence"],
                "structures": system_state["cognitive"]["structures"]
            }
        }
    
    def get_production_efficiency(self) -> float:
        """
        Calculate production network efficiency (0-1, higher=better).
        
        Measures how effectively the system produces its own components.
        """
        if not self.production_history:
            return 0.5
            
        # Calculate stability of production rates
        rate_history = [p["rates"] for p in self.production_history]
        stability = 1.0 / (1.0 + np.var([r[ComponentType.MEMBRANE.value] for r in rate_history]))
        
        # Calculate balance between components
        rates = [self.components[k]["production_rate"] for k in self.components]
        balance = 1.0 - np.std(rates)
        
        # Calculate recent performance trend
        recent_performance = self._calculate_performance_trend()
        
        return 0.5 * stability + 0.3 * balance + 0.2 * recent_performance
    
    def _calculate_performance_trend(self) -> float:
        """Calculate performance trend from recent history"""
        if len(self.performance_history) < 5:
            return 0.5  # Neutral if insufficient data
        
        # Simple moving average trend
        recent = self.performance_history[-5:]
        trend = (recent[-1] - recent[0]) / 5.0
        
        # Clamp to reasonable range
        return max(0.0, min(1.0, 0.5 + trend))
    
    def record_performance(self, performance: float):
        """Record performance metric for trend calculation"""
        self.performance_history.append(performance)
        
        # Keep history bounded
        if len(self.performance_history) > self.max_performance_history:
            self.performance_history.pop(0)
    
    def get_component_production_rate(self, component: ComponentType) -> float:
        """Get current production rate for a specific component"""
        return self.components.get(component, {}).get("production_rate", 0.0)
    
    def assess_component_quality(self, component: ComponentType, changes: Dict[ComponentType, float]) -> float:
        """
        Assess quality of component production.
        
        Returns a quality score between 0 and 1.
        """
        # Base quality on magnitude of change and production rate
        base_quality = min(1.0, abs(changes[component]) / 0.5)
        
        # Adjust based on component type
        if component == ComponentType.MEMBRANE:
            # Membrane quality depends on stability
            stability_factor = 0.7 + 0.3 * self.get_production_efficiency()
            return base_quality * stability_factor
        elif component == ComponentType.COGNITIVE_STRUCTURES:
            # Cognitive structures quality depends on coherence
            coherence = self.production_history[-1]["system_state"]["cognitive"]["coherence"] if self.production_history else 0.5
            return base_quality * (0.6 + 0.4 * coherence)
        
        return base_quality
    
    def get_quality_metrics(self) -> Dict[str, float]:
        """Get quality metrics for all components"""
        if not self.production_history:
            return {
                "overall_quality": 0.5,
                "quality_trend": 0.0,
                "component_qualities": {c.value: 0.5 for c in ComponentType}
            }
        
        # Get last production cycle
        last_cycle = self.production_history[-1]
        changes = {ComponentType(k): v for k, v in last_cycle["changes"].items()}
        
        # Calculate component qualities
        component_qualities = {}
        for component in ComponentType:
            component_qualities[component.value] = self.assess_component_quality(component, changes)
        
        # Calculate overall quality
        overall_quality = np.mean(list(component_qualities.values()))
        
        # Calculate quality trend
        quality_trend = 0.0
        if len(self.production_history) >= 5:
            recent_qualities = [
                np.mean([self.assess_component_quality(c, {ComponentType(k): v for k, v in cycle["changes"].items()})
                         for c in ComponentType])
                for cycle in self.production_history[-5:]
            ]
            quality_trend = recent_qualities[-1] - recent_qualities[0]
        
        return {
            "overall_quality": overall_quality,
            "quality_trend": quality_trend,
            "component_qualities": component_qualities
        }
    
    def get_crisis_assessment(self) -> Dict[str, Any]:
        """Get current crisis assessment"""
        if not self.production_history:
            return {
                "crisis_level": 0.0,
                "crisis_response": None,
                "recent_crisis_events": 0
            }
        
        # Calculate crisis level based on production efficiency
        production_efficiency = self.get_production_efficiency()
        crisis_level = max(0.0, 1.0 - production_efficiency)
        
        # Get crisis response
        crisis_response = self.crisis_manager.assess_crisis({
            "crisis_level": crisis_level,
            "health": production_efficiency,
            "stability": 1.0 / (1.0 + np.var([p["rates"][ComponentType.MEMBRANE.value] 
                                           for p in self.production_history[-10:]]))
        })
        
        # Count recent crisis events
        recent_crisis_events = sum(1 for p in self.production_history[-20:] 
                                 if p["rates"][ComponentType.MEMBRANE.value] < self.config.min_production_rate * 2)
        
        return {
            "crisis_level": crisis_level,
            "crisis_response": {
                "level": crisis_response.level.value,
                "actions": crisis_response.actions,
                "priority": crisis_response.priority,
                "duration": crisis_response.recommended_duration
            },
            "recent_crisis_events": recent_crisis_events
        }
    
    def execute_crisis_response(self, core_system: Any) -> Dict[str, Any]:
        """Execute appropriate crisis response based on current assessment"""
        crisis_assessment = self.get_crisis_assessment()
        crisis_level = CrisisLevel(crisis_assessment["crisis_response"]["level"])
        
        # Only execute response if above low threshold
        if crisis_level in [CrisisLevel.MEDIUM, CrisisLevel.HIGH, CrisisLevel.CRITICAL]:
            crisis_response = self.crisis_manager.assess_crisis({
                "crisis_level": crisis_assessment["crisis_level"]
            })
            return self.crisis_manager.execute_response(crisis_response, core_system)
        
        return {"status": "no_action_required"}
    
    def get_ssp_integration_metrics(self) -> Dict[str, float]:
        """
        Get metrics suitable for SSP integration.
        
        Returns quality metrics in a format SSP can use for reward shaping.
        """
        quality_metrics = self.get_quality_metrics()
        
        return {
            "production_efficiency": self.get_production_efficiency(),
            "overall_quality": quality_metrics["overall_quality"],
            "quality_trend": quality_metrics["quality_trend"],
            "crisis_level": self.get_crisis_assessment()["crisis_level"],
            "component_quality_membrane": quality_metrics["component_qualities"][ComponentType.MEMBRANE.value],
            "component_quality_cognitive": quality_metrics["component_qualities"][ComponentType.COGNITIVE_STRUCTURES.value]
        }