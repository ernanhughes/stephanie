# stephanie/components/jitter/core/crisis_manager.py
from __future__ import annotations

import time
import logging
from typing import Dict, Any, List
from enum import Enum

log = logging.getLogger("stephanie.jitter.crisis")

class CrisisLevel(str, Enum):
    NORMAL = "normal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class CrisisResponse:
    """Response protocol for different crisis levels"""
    def __init__(
        self,
        level: CrisisLevel,
        actions: List[str],
        priority: str,
        duration: float,
        success_criteria: Dict[str, Any]
    ):
        self.level = level
        self.actions = actions
        self.priority = priority
        self.duration = duration
        self.success_criteria = success_criteria

class CrisisManager:
    """
    Manages crisis detection, assessment, and response.
    
    Features:
    - Multi-level crisis detection
    - Adaptive response protocols
    - Crisis history tracking
    - Success metrics for response evaluation
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.crisis_levels = {
            CrisisLevel.NORMAL: 0.0,
            CrisisLevel.LOW: 0.3,
            CrisisLevel.MEDIUM: 0.6,
            CrisisLevel.HIGH: 0.8,
            CrisisLevel.CRITICAL: 0.95
        }
        
        # Response protocols for different crisis levels
        self.response_protocols = {
            CrisisLevel.LOW: CrisisResponse(
                level=CrisisLevel.LOW,
                actions=["adjust_homeostasis", "monitor_closely"],
                priority="low",
                duration=120.0,
                success_criteria={"crisis_level": 0.2}
            ),
            CrisisLevel.MEDIUM: CrisisResponse(
                level=CrisisLevel.MEDIUM,
                actions=["conserve_energy", "adjust_attention_weights"],
                priority="medium",
                duration=60.0,
                success_criteria={"crisis_level": 0.4}
            ),
            CrisisLevel.HIGH: CrisisResponse(
                level=CrisisLevel.HIGH,
                actions=["fortify_boundary", "reduce_cognitive_load", "alert_monitoring"],
                priority="high",
                duration=30.0,
                success_criteria={"crisis_level": 0.6}
            ),
            CrisisLevel.CRITICAL: CrisisResponse(
                level=CrisisLevel.CRITICAL,
                actions=["initiate_apoptosis", "preserve_legacy", "alert_emergency"],
                priority="highest",
                duration=5.0,
                success_criteria={"apoptosis_initiated": True}
            )
        }
        
        self.crisis_history = []
        self.max_history = config.get("max_crisis_history", 100)
        self.logger = logging.getLogger("stephanie.jitter.crisis.manager")
    
    def assess_crisis(self, telemetry: Dict[str, Any]) -> CrisisResponse:
        """Assess current crisis level based on telemetry data"""
        crisis_level = telemetry.get("crisis_level", 0.0)
        health = telemetry.get("health", 0.5)
        stability = telemetry.get("stability", 0.7)
        
        # Determine crisis level
        level = CrisisLevel.NORMAL
        for crisis_level_enum, threshold in sorted(
            self.crisis_levels.items(), key=lambda x: x[1], reverse=True
        ):
            if crisis_level >= threshold:
                level = crisis_level_enum
                break
        
        return self.response_protocols[level]
    
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
                self.logger.error(f"Crisis response action failed: {action} - {str(e)}")
        
        # Record for learning
        self.crisis_history.append({
            "timestamp": time.time(),
            "response": {
                "level": response.level.value,
                "actions": response.actions,
                "priority": response.priority,
                "duration": response.duration
            },
            "results": results,
            "core_state_snapshot": self._capture_core_snapshot(core_system)
        })
        
        # Keep history bounded
        if len(self.crisis_history) > self.max_history:
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
    
    def evaluate_response_success(self, response: CrisisResponse, core_system: Any) -> bool:
        """Evaluate if crisis response was successful"""
        # Check against success criteria
        success = True
        
        for criterion, value in response.success_criteria.items():
            if criterion == "crisis_level":
                current_level = core_system.homeostasis.get_telemetry()["crisis_level"]
                success = success and (current_level <= value)
            elif criterion == "apoptosis_initiated":
                success = success and value == core_system.apoptosis_system.initiated
        
        return success
    
    def get_crisis_trends(self) -> Dict[str, Any]:
        """Analyze trends in crisis events for predictive capabilities"""
        if not self.crisis_history:
            return {
                "crisis_frequency": 0.0,
                "average_duration": 0.0,
                "success_rate": 0.0,
                "trend": "stable"
            }
        
        # Calculate metrics
        total_events = len(self.crisis_history)
        high_events = sum(1 for c in self.crisis_history 
                         if c["response"]["level"] in [CrisisLevel.HIGH.value, CrisisLevel.CRITICAL.value])
        success_count = 0
        total_duration = 0.0
        
        for event in self.crisis_history:
            # Simplified success check (would be more complex in production)
            success = "success" in str(event["results"]).lower()
            if success:
                success_count += 1
            
            # Could calculate actual duration from timestamps
            total_duration += event["response"]["duration"]
        
        # Calculate trend
        recent_events = self.crisis_history[-5:]
        recent_levels = [self.crisis_levels[CrisisLevel(event["response"]["level"])] 
                        for event in recent_events]
        trend = "improving" if all(recent_levels[i] >= recent_levels[i+1] 
                                  for i in range(len(recent_levels)-1)) else "worsening"
        
        return {
            "crisis_frequency": high_events / total_events if total_events > 0 else 0.0,
            "average_duration": total_duration / total_events if total_events > 0 else 0.0,
            "success_rate": success_count / total_events if total_events > 0 else 0.0,
            "trend": trend
        }