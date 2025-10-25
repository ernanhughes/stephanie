"""
JASTelemetry
============
Comprehensive telemetry system for monitoring the Jitter Autopoietic System.

Key Features:
- Multi-dimensional vital sign monitoring
- Crisis detection and alerting
- Historical data tracking for reproduction analysis
- Integration with Stephanie's monitoring infrastructure
- Custom serialization for efficient network transmission
"""

import json
import time
import logging
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
from stephanie.services.bus.nats_client import get_js
from stephanie.utils.serialization import compress_data

log = logging.getLogger("stephanie.jas.telemetry")

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
    alerts: List[str]

class JASTelemetry:
    """
    The telemetry system for the Jitter Autopoietic System.
    
    This system:
    - Collects vital signs from all JAS components
    - Detects and reports critical events
    - Maintains historical data for analysis
    - Publishes telemetry to Stephanie's monitoring infrastructure
    - Supports efficient serialization for low-bandwidth transmission
    """
    
    def __init__(
        self,
        cfg: Dict[str, Any],
        subject: str = "arena.jitter.telemetry",
        interval: float = 1.0
    ):
        self.cfg = cfg
        self.subject = subject
        self.interval = interval
        self.js = None
        self.last_publish = 0
        self.history: List[VitalSigns] = []
        self.max_history = cfg.get("telemetry_max_history", 1000)
        self.alerts: List[Dict[str, Any]] = []
        self.max_alerts = cfg.get("telemetry_max_alerts", 100)
        self.crisis_threshold = cfg.get("crisis_threshold", 0.5)
        self.health_history = []
        self.max_health_history = 100
        
        log.info(f"JASTelemetry initialized with publish interval={interval}s")

    async def init(self):
        """Initialize telemetry system (connect to NATS)"""
        try:
            self.js = await get_js()
            log.info("JASTelemetry connected to NATS")
        except Exception as e:
            log.error(f"Failed to connect to NATS: {str(e)}")
            # Continue without NATS - telemetry will still collect locally

    def collect(self, core, homeo, triune) -> VitalSigns:
        """
        Collect vital signs from all JAS components.
        
        Args:
            core: The AutopoieticCore instance
            homeo: The EnhancedHomeostasis instance
            triune: The TriuneCognition instance
            
        Returns:
            VitalSigns: Current vital signs
        """
        # Get homeostasis telemetry
        homeo_telem = homeo.get_telemetry()
        
        # Get most recent cognitive state
        cognitive_state = triune.state_history[-1] if triune.state_history else None
        
        # Get VPM metrics
        vpm_count = 0
        vpm_diversity = 0.5
        if hasattr(core, "vpm_manager"):
            vpm_count = core.vpm_manager.count()
            vpm_diversity = core.vpm_manager.diversity_score()
        
        # Collect alerts
        alerts = self._detect_alerts(core, homeo, triune, homeo_telem)
        
        # Create vital signs object
        vital_signs = VitalSigns(
            boundary_integrity=core.membrane.integrity,
            energy_cognitive=core.energy.level("cognitive"),
            energy_metabolic=core.energy.level("metabolic"),
            energy_reserve=core.energy.level("reserve"),
            cognitive_integrated=cognitive_state.integrated if cognitive_state else 0.5,
            cognitive_energy=cognitive_state.cognitive_energy if cognitive_state else 0.0,
            vpm_count=vpm_count,
            vpm_diversity=vpm_diversity,
            health_score=homeo_telem["health"],
            crisis_level=homeo_telem["crisis_level"],
            tick=getattr(core, "tick", 0),
            timestamp=time.time(),
            layer_attention=cognitive_state.attention_weights if cognitive_state else {"reptilian": 0.3, "mammalian": 0.3, "primate": 0.4},
            layer_veto=cognitive_state.layer_veto if cognitive_state else "none",
            reasoning_depth=cognitive_state.reasoning_depth if cognitive_state else 0,
            threat_level=cognitive_state.threat_level if cognitive_state else 0.5,
            emotional_valence=cognitive_state.emotional_valence if cognitive_state else 0.0,
            homeostatic_error=homeo_telem["homeostatic_error"] if "homeostatic_error" in homeo_telem else 0.0,
            regulatory_actions=homeo_telem["regulatory_actions"],
            alerts=alerts
        )
        
        # Record for history
        self.history.append(vital_signs)
        if len(self.history) > self.max_history:
            self.history.pop(0)
            
        # Record health for trend analysis
        self.health_history.append(homeo_telem["health"])
        if len(self.health_history) > self.max_health_history:
            self.health_history.pop(0)
            
        return vital_signs

    def _detect_alerts(
        self, 
        core, 
        homeo, 
        triune, 
        homeo_telem: Dict[str, Any]
    ) -> List[str]:
        """Detect and record critical alerts"""
        alerts = []
        
        # Energy alerts
        if core.energy.level("metabolic") < 5:
            alerts.append("metabolic_low")
        if core.energy.level("cognitive") < 5:
            alerts.append("cognitive_low")
        if core.energy.level("reserve") > core.energy.max_reserve * 0.8:
            alerts.append("reproduction_ready")
            
        # Boundary alerts
        if core.membrane.integrity < 0.3:
            alerts.append("boundary_critical")
        if core.membrane.thickness < 0.2:
            alerts.append("boundary_thin")
            
        # Cognitive alerts
        if triune.state_history and triune.state_history[-1].layer_veto != "none":
            alerts.append(f"veto_active:{triune.state_history[-1].layer_veto}")
            
        # Crisis alerts
        if homeo_telem["crisis_level"] > 0.8:
            alerts.append("crisis_high")
        if homeo_telem["crisis_level"] > 0.95:
            alerts.append("crisis_critical")
            
        # Record alerts with timestamps
        for alert in alerts:
            self.alerts.append({
                "alert": alert,
                "timestamp": time.time(),
                "tick": getattr(core, "tick", 0),
                "health": homeo_telem["health"]
            })
            log.warning(f"JAS Alert: {alert}")
            
        # Prune old alerts
        if len(self.alerts) > self.max_alerts:
            self.alerts = self.alerts[-self.max_alerts:]
            
        return alerts

    async def publish(self, vital_signs: VitalSigns):
        """Publish vital signs to NATS"""
        if not self.js:
            return
            
        # Respect publish interval
        now = time.time()
        if now - self.last_publish < self.interval:
            return
            
        try:
            # Prepare payload
            payload = {
                "type": "vital_signs",
                "data": asdict(vital_signs),
                "timestamp": vital_signs.timestamp
            }
            
            # Compress for efficient transmission
            compressed = compress_data(payload)
            
            # Publish
            await self.js.publish(self.subject, compressed)
            
            self.last_publish = now
            log.debug(f"Published telemetry (tick={vital_signs.tick}, health={vital_signs.health_score:.3f})")
            
        except Exception as e:
            log.error(f"Failed to publish telemetry: {str(e)}")

    def get_health_trend(self) -> Dict[str, Any]:
        """Get health trend information for monitoring"""
        if not self.health_history:
            return {
                "current_health": 0.5,
                "trend": "stable",
                "stability": 0.5
            }
        
        current_health = self.health_history[-1]
        trend = "stable"
        
        # Calculate trend from last 10 measurements
        if len(self.health_history) >= 10:
            recent = self.health_history[-10:]
            slope = (recent[-1] - recent[0]) / 9
            
            if slope > 0.05:
                trend = "improving"
            elif slope < -0.05:
                trend = "declining"
        
        # Calculate stability (1 - variance)
        stability = 1.0 / (1.0 + np.var(self.health_history))
        
        return {
            "current_health": current_health,
            "trend": trend,
            "stability": float(stability)
        }

    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of recent alerts"""
        if not self.alerts:
            return {
                "total": 0,
                "critical": 0,
                "by_type": {},
                "last_24h": 0
            }
        
        now = time.time()
        last_24h = [a for a in self.alerts if now - a["timestamp"] < 86400]
        
        # Count by type
        by_type = {}
        for alert in self.alerts:
            alert_type = alert["alert"].split(":")[0]
            by_type[alert_type] = by_type.get(alert_type, 0) + 1
        
        # Count critical alerts (crisis_critical and boundary_critical)
        critical_types = ["crisis_critical", "boundary_critical"]
        critical = sum(1 for a in self.alerts if any(ct in a["alert"] for ct in critical_types))
        
        return {
            "total": len(self.alerts),
            "critical": critical,
            "by_type": by_type,
            "last_24h": len(last_24h)
        }

    def get_vital_signs_history(self, n: int = 100) -> List[Dict[str, Any]]:
        """Get historical vital signs for analysis"""
        return [asdict(v) for v in self.history[-n:]]

    def get_crisis_events(self) -> List[Dict[str, Any]]:
        """Get all crisis events from history"""
        return [
            asdict(v) for v in self.history 
            if v.crisis_level > self.crisis_threshold
        ]

    def get_reproduction_opportunities(self) -> List[Dict[str, Any]]:
        """Get times when reproduction was possible"""
        return [
            asdict(v) for v in self.history 
            if "reproduction_ready" in v.alerts
        ]
