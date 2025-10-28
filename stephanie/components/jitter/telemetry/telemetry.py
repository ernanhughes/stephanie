# stephanie/components/jitter/telemetry/telemetry.py
"""
telemetry.py
============
Comprehensive telemetry system for monitoring the Jitter Autopoietic System.

Key Features:
- Multi-dimensional vital sign monitoring
- Crisis detection and alerting
- Historical data tracking for reproduction analysis
- Integration with Stephanie's monitoring infrastructure
- Custom serialization for efficient network transmission
- Configuration validation with Pydantic
- Circuit breaker pattern for resilience
- Comprehensive telemetry and monitoring
- SSP integration hooks
- Performance optimizations
"""
from __future__ import annotations


from typing import Dict, Any, List, Optional, Tuple
import json
import time
import logging
import numpy as np
from dataclasses import dataclass, asdict, field
from pydantic import BaseModel, Field, validator
from enum import Enum
from functools import wraps
import asyncio
import uuid

from stephanie.services.bus.nats_client import get_js
from stephanie.utils.serialization import compress_data
from stephanie.components.jitter.telemetry.jaf import JitterArtifactV0

log = logging.getLogger("stephanie.jitter.telemetry")

class VitalSignsConfig(BaseModel):
    """Validated configuration for JASTelemetry"""
    subject: str = Field("arena.jitter.telemetry", description="JetStream subject for telemetry")
    interval: float = Field(1.0, ge=0.1, le=10.0, description="Telemetry publish interval in seconds")
    max_history: int = Field(1000, ge=100, le=10000, description="Maximum history length")
    alert_threshold: float = Field(0.8, ge=0.5, le=0.95, description="Threshold for crisis alerts")
    
    @validator('alert_threshold')
    def validate_alert_threshold(cls, v):
        if v > 0.95:
            raise ValueError('alert_threshold must be less than or equal to 0.95')
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
    def publish(vital_signs):
        # Publishing logic here
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
class VitalSigns:
    """Core vital signs of the Jitter organism"""
    boundary_integrity: float
    energy_cognitive: float
    energy_metabolic: float
    energy_reserve: float
    cognitive_integrated: float
    cognitive_energy: float
    vpm_count: int
    vpm_diversity: float
    health_score: float
    crisis_level: float
    tick: int
    timestamp: float
    layer_attention: Dict[str, float]
    layer_veto: str
    reasoning_depth: int
    threat_level: float
    emotional_valence: float
    homeostatic_error: float
    regulatory_actions: Dict[str, float]
    alerts: List[str] = field(default_factory=list)

@dataclass
class TelemetryMetrics:
    """Metrics for telemetry system performance"""
    publish_rate: float = 0.0
    alert_frequency: float = 0.0
    data_quality: float = 0.0
    processing_time_ms: float = 0.0
    history_size: int = 0

class JASTelemetry:
    """
    The telemetry system for the Jitter Autopoietic System.
    
    This system:
    - Collects vital signs from all JAS components
    - Detects and reports critical events
    - Maintains historical data for analysis
    - Publishes telemetry to Stephanie's monitoring infrastructure
    - Supports efficient serialization for low-bandwidth transmission
    - Implements circuit breaker for resilience
    - Provides comprehensive metrics for monitoring
    """
    
    def __init__(
        self,
        cfg: Dict[str, Any],
        subject: str = "arena.jitter.telemetry",
        interval: float = 1.0
    ):
        try:
            # Validate configuration
            self.config = VitalSignsConfig(**cfg)
            log.info("JASTelemetry configuration validated successfully")
        except Exception as e:
            log.error(f"Configuration validation failed: {str(e)}")
            # Use safe defaults
            self.config = VitalSignsConfig()
        
        self.subject = subject
        self.interval = interval
        self.js = None
        self.last_publish = 0
        self.history: List[VitalSigns] = []
        self.max_history = self.config.max_history
        
        # Initialize metrics
        self.metrics = TelemetryMetrics()
        
        # Initialize performance tracking
        self.processing_times = []
        self.max_processing_history = 100
        
        # Initialize circuit breaker
        self.circuit_breaker = CircuitBreaker()
        
        log.info("JASTelemetry initialized with monitoring capabilities")
    
    async def init(self):
        """Initialize JetStream connection"""
        try:
            self.js = await get_js()
            log.info("JetStream connection established for telemetry")
        except Exception as e:
            log.error(f"Failed to establish JetStream connection: {str(e)}")
            self.js = None
    
    @CircuitBreaker()
    def collect(
        self,
        core,
        homeostasis,
        triune
    ) -> VitalSigns:
        """
        Collect vital signs from all JAS components.
        
        Args:
            core: AutopoieticCore instance
            homeostasis: Homeostasis system
            triune: TriuneCognition system
            
        Returns:
            VitalSigns object with collected data
        """
        start_time = time.time()
        
        try:
            # Collect data from core components
            boundary_integrity = core.membrane.integrity
            energy_cognitive = core.energy.level("cognitive")
            energy_metabolic = core.energy.level("metabolic")
            energy_reserve = core.energy.level("reserve")
            
            # Collect from triune cognition
            integrated = 0.0
            threat_level = 0.0
            emotional_valence = 0.0
            reasoning_depth = 0
            layer_attention = {}
            layer_veto = "none"
            
            if triune.state_history:
                latest_state = triune.state_history[-1]
                integrated = latest_state.integrated
                threat_level = latest_state.threat_level
                emotional_valence = latest_state.emotional_valence
                reasoning_depth = latest_state.reasoning_depth
                layer_attention = latest_state.attention_weights
                layer_veto = latest_state.layer_veto
            
            # Collect from homeostasis
            homeostatic_error = 0.0
            regulatory_actions = {}
            
            if hasattr(homeostasis, 'get_telemetry'):
                homeo_telemetry = homeostasis.get_telemetry()
                homeostatic_error = homeo_telemetry.get("homeostatic_error", 0.0)
                regulatory_actions = homeo_telemetry.get("regulatory_actions", {})
            
            # Calculate health score
            health_score = self._calculate_health_score(
                boundary_integrity,
                energy_cognitive,
                energy_metabolic,
                energy_reserve,
                integrated,
                threat_level
            )
            
            # Calculate crisis level
            crisis_level = self._calculate_crisis_level(
                boundary_integrity,
                energy_cognitive,
                energy_metabolic,
                energy_reserve,
                threat_level,
                emotional_valence
            )
            
            # Generate alerts if needed
            alerts = self._generate_alerts(
                boundary_integrity,
                energy_cognitive,
                energy_metabolic,
                energy_reserve,
                health_score,
                crisis_level
            )
            
            # Create vital signs object
            vital_signs = VitalSigns(
                boundary_integrity=boundary_integrity,
                energy_cognitive=energy_cognitive,
                energy_metabolic=energy_metabolic,
                energy_reserve=energy_reserve,
                cognitive_integrated=integrated,
                cognitive_energy=0.0,  # Would be calculated from triune
                vpm_count=getattr(core, 'vpm_count', 0),
                vpm_diversity=getattr(core, 'vpm_diversity', 0.0),
                health_score=health_score,
                crisis_level=crisis_level,
                tick=getattr(core, 'tick', 0),
                timestamp=time.time(),
                layer_attention=layer_attention,
                layer_veto=layer_veto,
                reasoning_depth=reasoning_depth,
                threat_level=threat_level,
                emotional_valence=emotional_valence,
                homeostatic_error=homeostatic_error,
                regulatory_actions=regulatory_actions,
                alerts=alerts
            )
            
            # Add to history
            self.history.append(vital_signs)
            if len(self.history) > self.max_history:
                self.history.pop(0)
            
            # Update metrics
            self._update_metrics()
            
            # Log processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > self.max_processing_history:
                self.processing_times.pop(0)
                
            log.debug(f"Collected vital signs (health={health_score:.3f}, crisis={crisis_level:.3f})")
            return vital_signs
            
        except Exception as e:
            log.error(f"Error collecting vital signs: {str(e)}", exc_info=True)
            # Return minimal vital signs as fallback
            return VitalSigns(
                boundary_integrity=0.5,
                energy_cognitive=50.0,
                energy_metabolic=50.0,
                energy_reserve=10.0,
                cognitive_integrated=0.5,
                cognitive_energy=0.0,
                vpm_count=0,
                vpm_diversity=0.0,
                health_score=0.5,
                crisis_level=0.0,
                tick=0,
                timestamp=time.time(),
                layer_attention={},
                layer_veto="none",
                reasoning_depth=0,
                threat_level=0.0,
                emotional_valence=0.0,
                homeostatic_error=0.0,
                regulatory_actions={},
                alerts=["COLLECT_ERROR"]
            )
    
    def _calculate_health_score(
        self,
        boundary_integrity: float,
        energy_cognitive: float,
        energy_metabolic: float,
        energy_reserve: float,
        integrated: float,
        threat_level: float
    ) -> float:
        """Calculate overall health score"""
        # Weighted combination of health indicators
        # Boundary integrity (20%)
        boundary_score = boundary_integrity * 0.2
        
        # Energy balance (30%)
        energy_balance = (energy_cognitive + energy_metabolic + energy_reserve) / 3.0
        energy_score = min(1.0, energy_balance / 100.0) * 0.3
        
        # Cognitive integration (30%)
        cognitive_score = integrated * 0.3
        
        # Threat mitigation (20%)
        threat_score = (1.0 - threat_level) * 0.2
        
        return boundary_score + energy_score + cognitive_score + threat_score
    
    def _calculate_crisis_level(
        self,
        boundary_integrity: float,
        energy_cognitive: float,
        energy_metabolic: float,
        energy_reserve: float,
        threat_level: float,
        emotional_valence: float
    ) -> float:
        """Calculate crisis level based on system stress"""
        # Calculate stress factors
        boundary_stress = 1.0 - boundary_integrity
        energy_stress = 1.0 - ((energy_cognitive + energy_metabolic + energy_reserve) / 300.0)
        threat_stress = threat_level
        emotional_stress = 1.0 - (emotional_valence + 1.0) / 2.0  # Normalize to 0-1
        
        # Combined crisis level
        crisis_level = (
            boundary_stress * 0.25 +
            energy_stress * 0.3 +
            threat_stress * 0.25 +
            emotional_stress * 0.2
        )
        
        return min(1.0, max(0.0, crisis_level))
    
    def _generate_alerts(
        self,
        boundary_integrity: float,
        energy_cognitive: float,
        energy_metabolic: float,
        energy_reserve: float,
        health_score: float,
        crisis_level: float
    ) -> List[str]:
        """Generate alerts based on system state"""
        alerts = []
        
        # Boundary integrity alerts
        if boundary_integrity < 0.2:
            alerts.append("CRITICAL_BOUNDARY_FAILURE")
        elif boundary_integrity < 0.4:
            alerts.append("HIGH_BOUNDARY_STRESS")
        elif boundary_integrity < 0.6:
            alerts.append("MODERATE_BOUNDARY_STRESS")
        
        # Energy alerts
        energy_total = energy_cognitive + energy_metabolic + energy_reserve
        if energy_total < 50.0:
            alerts.append("LOW_ENERGY_LEVEL")
        elif energy_total < 100.0:
            alerts.append("MODERATE_ENERGY_LEVEL")
        
        # Health alerts
        if health_score < 0.3:
            alerts.append("CRITICAL_HEALTH_DEGRADED")
        elif health_score < 0.5:
            alerts.append("HIGH_HEALTH_RISK")
        elif health_score < 0.7:
            alerts.append("MODERATE_HEALTH_RISK")
        
        # Crisis alerts
        if crisis_level > self.config.alert_threshold:
            alerts.append("HIGH_CRISIS_LEVEL")
        elif crisis_level > self.config.alert_threshold * 0.8:
            alerts.append("MODERATE_CRISIS_LEVEL")
        
        return alerts
    
    def _update_metrics(self):
        """Update telemetry metrics based on recent data"""
        # Update publish rate
        if len(self.history) > 1:
            time_diff = self.history[-1].timestamp - self.history[0].timestamp
            if time_diff > 0:
                self.metrics.publish_rate = len(self.history) / time_diff
        
        # Update alert frequency
        if len(self.history) > 0:
            alert_count = sum(len(v.alerts) for v in self.history[-10:])
            self.metrics.alert_frequency = alert_count / max(1, len(self.history[-10:]))
        
        # Update processing time
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0.0
        self.metrics.processing_time_ms = avg_processing_time * 1000
        self.metrics.history_size = len(self.history)
    
    @CircuitBreaker()
    async def publish(self, vital_signs: VitalSigns) -> bool:
        """
        Publish vital signs to the telemetry subject.
        
        Args:
            vital_signs: Vital signs to publish
            
        Returns:
            True if successful, False otherwise
        """
        start_time = time.time()
        
        try:
            # Skip if not connected to JetStream
            if not self.js:
                log.warning("JetStream not connected, skipping telemetry publish")
                return False
            
            # Convert to dictionary for JSON serialization
            payload = {
                "type": "jas_telemetry",
                "spec": "jaf/0",
                "data": asdict(vital_signs),
                "timestamp": time.time(),
                "seq": len(self.history)
            }
            
            # Compress payload for efficient transmission
            compressed_payload = compress_data(payload)
            
            # Publish to JetStream
            await self.js.publish(self.subject, compressed_payload)
            
            # Log processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > self.max_processing_history:
                self.processing_times.pop(0)
                
            log.debug(f"Published telemetry (seq={len(self.history)})")
            return True
            
        except Exception as e:
            log.error(f"Error publishing telemetry: {str(e)}", exc_info=True)
            return False
    
    async def emit_artifact(self, artifact: JitterArtifactV0, artifact_type: str) -> bool:
        """
        Emit a Jitter artifact (JAF) to the telemetry system.
        
        Args:
            artifact: Jitter artifact to emit
            artifact_type: Type of artifact (e.g., "reproduction", "apoptosis")
            
        Returns:
            True if successful, False otherwise
        """
        start_time = time.time()
        
        try:
            # Skip if not connected to JetStream
            if not self.js:
                log.warning("JetStream not connected, skipping artifact publish")
                return False
            
            # Convert artifact to dictionary
            artifact_dict = artifact.to_dict()
            
            # Create payload
            payload = {
                "type": "jas_artifact",
                "artifact_type": artifact_type,
                "data": artifact_dict,
                "timestamp": time.time()
            }
            
            # Compress payload
            compressed_payload = compress_data(payload)
            
            # Publish to JetStream
            await self.js.publish(self.subject, compressed_payload)
            
            # Log processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > self.max_processing_history:
                self.processing_times.pop(0)
                
            log.info(f"Published JAF artifact ({artifact_type})")
            return True
            
        except Exception as e:
            log.error(f"Error publishing artifact: {str(e)}", exc_info=True)
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current telemetry metrics for monitoring and adaptation"""
        return {
            "publish_rate": self.metrics.publish_rate,
            "alert_frequency": self.metrics.alert_frequency,
            "data_quality": self.metrics.data_quality,
            "processing_time_ms": self.metrics.processing_time_ms,
            "history_size": self.metrics.history_size,
            "circuit_breaker_state": self.circuit_breaker.get_state(),
            "subject": self.subject,
            "interval": self.interval
        }
    
    def get_telemetry_history(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get recent telemetry history for analysis or reporting"""
        return [
            {
                "timestamp": v.timestamp,
                "tick": v.tick,
                "health_score": v.health_score,
                "crisis_level": v.crisis_level,
                "boundary_integrity": v.boundary_integrity,
                "energy_cognitive": v.energy_cognitive,
                "energy_metabolic": v.energy_metabolic,
                "energy_reserve": v.energy_reserve,
                "alerts": v.alerts
            }
            for v in self.history[-n:]
        ]
    
    def get_health_trend(self) -> Dict[str, Any]:
        """
        Calculate health trend information for monitoring.
        
        Returns:
            Dictionary with health trend information
        """
        if not self.history:
            return {
                "current_health": 0.5,
                "trend": "stable",
                "stability": 0.5,
                "recent_crisis": 0
            }
        
        # Get health metrics for recent history
        recent_health = [v.health_score for v in self.history[-50:]]
        recent_crisis = [1 for v in self.history[-50:] if v.crisis_level > 0.5]
        
        # Calculate trend from last 10 measurements
        if len(recent_health) >= 10:
            recent = recent_health[-10:]
            slope = (recent[-1] - recent[0]) / 9
            
            if slope > 0.05:
                trend = "improving"
            elif slope < -0.05:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "stable"
        
        # Calculate stability (1 - variance)
        stability = 1.0 / (1.0 + np.var(recent_health)) if recent_health else 0.5
        
        return {
            "current_health": float(recent_health[-1]) if recent_health else 0.5,
            "trend": trend,
            "stability": float(stability),
            "recent_crisis": len(recent_crisis)
        }
    
    def get_ssp_integration_metrics(self) -> Dict[str, float]:
        """
        Get metrics suitable for SSP integration.
        
        Returns telemetry metrics in a format SSP can use for reward shaping.
        """
        metrics = self.get_metrics()
        health_trend = self.get_health_trend()
        
        return {
            "health_score": health_trend["current_health"],
            "health_trend": 1.0 if health_trend["trend"] == "improving" else 
                          0.5 if health_trend["trend"] == "stable" else 0.0,
            "stability": health_trend["stability"],
            "crisis_level": self.history[-1].crisis_level if self.history else 0.0,
            "alert_frequency": metrics["alert_frequency"],
            "publish_rate": metrics["publish_rate"],
            "processing_efficiency": 1.0 / (1.0 + metrics["processing_time_ms"])
        }
    
    def reset(self):
        """Reset telemetry history and metrics"""
        self.history.clear()
        self.metrics = TelemetryMetrics()
        self.processing_times.clear()
        log.info("Telemetry system reset")