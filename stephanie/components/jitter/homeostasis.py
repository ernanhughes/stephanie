# stephanie/components/jitter/homeostasis.py
"""
EnhancedHomeostasis
===================
The self-regulation system that maintains Jitter's internal stability across
multiple dimensions using biological feedback loops.

Key Features:
- Multi-dimensional PID controllers for different physiological parameters
- Adaptive setpoints that shift with experience
- Crisis detection and response protocols
- Energy-based homeostatic efficiency measurement
- Integration with cognitive state for anticipatory regulation
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

log = logging.getLogger(__file__)

@dataclass
class HomeostaticState:
    """Complete homeostatic state snapshot"""
    energy_balance: float       # Cognitive/metabolic energy ratio
    boundary_integrity: float   # Membrane integrity (0-1)
    cognitive_flow: float       # Cognitive processing efficiency
    vpm_diversity: float        # Diversity of VPM store (0-1)
    homeostatic_error: float    # Overall error from setpoints
    crisis_level: float         # Current crisis level (0-1)
    regulatory_actions: Dict[str, float]  # Recent regulatory actions
    setpoints: Dict[str, float] # Current setpoints
    stability: float            # Recent stability metric

class PIDController:
    """Enhanced PID controller with adaptive tuning and anti-windup"""
    
    def __init__(
        self, 
        kp: float, 
        ki: float, 
        kd: float, 
        setpoint: float,
        output_limits: Tuple[float, float] = (-1.0, 1.0),
        anti_windup: bool = True
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.output_limits = output_limits
        self.anti_windup = anti_windup
        
        # State variables
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None
        
        # Tracking for adaptation
        self.error_history = []
        self.max_history = 100
        
    def update(self, measurement: float, dt: Optional[float] = None) -> float:
        """Update PID controller with new measurement"""
        current_time = time.time()
        if dt is None and self.prev_time is not None:
            dt = current_time - self.prev_time
        elif dt is None:
            dt = 0.1  # Default time step
            
        # Calculate error
        error = self.setpoint - measurement
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term
        self.integral += self.ki * error * dt
        if self.anti_windup:
            # Anti-windup: limit integral to output range
            min_out, max_out = self.output_limits
            self.integral = np.clip(
                self.integral, 
                min_out / (self.ki * dt + 1e-8), 
                max_out / (self.ki * dt + 1e-8)
            )
        
        # Derivative term
        d_term = 0.0
        if self.prev_time is not None:
            d_term = self.kd * (error - self.prev_error) / dt
            
        # Calculate output
        output = p_term + self.integral + d_term
        
        # Apply output limits
        output = np.clip(output, *self.output_limits)
        
        # Update state
        self.prev_error = error
        self.prev_time = current_time
        
        # Track for adaptation
        self.error_history.append(abs(error))
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)
            
        return output
    
    def adapt(self, performance_metric: float):
        """Adapt PID parameters based on performance"""
        if len(self.error_history) < 10:
            return
            
        # Calculate recent error trend
        recent_errors = self.error_history[-10:]
        error_trend = np.mean(recent_errors[5:]) - np.mean(recent_errors[:5])
        
        # Adjust parameters based on performance
        if performance_metric > 0.7:  # Good performance
            # Slightly reduce gains to avoid overcorrection
            self.kp *= 0.95
            self.ki *= 0.9
        elif performance_metric < 0.3:  # Poor performance
            # Increase gains to respond more strongly
            self.kp *= 1.1
            self.ki *= 1.05
            # If error is increasing, increase derivative term
            if error_trend > 0:
                self.kd *= 1.2
                
        # Constrain to reasonable ranges
        self.kp = np.clip(self.kp, 0.1, 5.0)
        self.ki = np.clip(self.ki, 0.01, 2.0)
        self.kd = np.clip(self.kd, 0.0, 2.0)
    
    def get_performance(self) -> float:
        """Get current controller performance (0-1, higher=better)"""
        if not self.error_history:
            return 0.5
            
        # Performance is 1 - normalized average error
        avg_error = np.mean(self.error_history)
        # Normalize assuming typical error range of 0-0.5
        return max(0.0, min(1.0, 1.0 - (avg_error / 0.5)))

class EnhancedHomeostasis:
    """
    The complete homeostasis system with multi-dimensional regulation.
    
    This system:
    - Monitors key physiological parameters
    - Applies regulatory actions to maintain stability
    - Detects and responds to crises
    - Learns optimal setpoints over time
    - Integrates with cognitive system for anticipatory regulation
    """
    
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.crisis_threshold = cfg.get("crisis_threshold", 0.5)
        self.crisis_counter = 0
        self.max_crisis = cfg.get("max_crisis", 10)
        self.regulatory_history = []
        
        # Initialize PID controllers for different dimensions
        self.controllers = {
            "energy_balance": PIDController(
                kp=cfg.get("energy_kp", 1.0),
                ki=cfg.get("energy_ki", 0.1),
                kd=cfg.get("energy_kd", 0.05),
                setpoint=cfg.get("energy_setpoint", 1.0),
                output_limits=(-0.5, 0.5)
            ),
            "boundary_integrity": PIDController(
                kp=cfg.get("boundary_kp", 2.0),
                ki=cfg.get("boundary_ki", 0.2),
                kd=cfg.get("boundary_kd", 0.1),
                setpoint=cfg.get("boundary_setpoint", 0.8),
                output_limits=(-0.3, 0.3)
            ),
            "cognitive_flow": PIDController(
                kp=cfg.get("cognitive_kp", 0.8),
                ki=cfg.get("cognitive_ki", 0.05),
                kd=cfg.get("cognitive_kd", 0.02),
                setpoint=cfg.get("cognitive_setpoint", 0.7),
                output_limits=(-0.2, 0.2)
            ),
            "vpm_diversity": PIDController(
                kp=cfg.get("diversity_kp", 0.5),
                ki=cfg.get("diversity_ki", 0.03),
                kd=cfg.get("diversity_kd", 0.01),
                setpoint=cfg.get("diversity_setpoint", 0.6),
                output_limits=(-0.1, 0.1)
            )
        }
        
        # Adaptive setpoints
        self.adaptive_setpoints = {
            "energy_balance": cfg.get("energy_setpoint", 1.0),
            "boundary_integrity": cfg.get("boundary_setpoint", 0.8),
            "cognitive_flow": cfg.get("cognitive_setpoint", 0.7),
            "vpm_diversity": cfg.get("diversity_setpoint", 0.6)
        }
        
        # Performance tracking for adaptation
        self.performance_history = []
        self.max_performance_history = 100
        
        log.info("EnhancedHomeostasis initialized with adaptive PID control")

    def regulate(self, core) -> HomeostaticState:
        """
        Apply homeostatic regulation to maintain internal stability.
        
        Args:
            core: The AutopoieticCore instance to regulate
            
        Returns:
            HomeostaticState: Current homeostatic state
        """
        # Get current measurements
        measurements = self._get_measurements(core)
        
        # Calculate regulatory actions
        regulatory_actions = {}
        total_error = 0.0
        
        for dim, controller in self.controllers.items():
            # Update setpoint if adaptive
            controller.setpoint = self.adaptive_setpoints[dim]
            
            # Get measurement for this dimension
            measurement = measurements[dim]
            
            # Calculate regulatory action
            action = controller.update(measurement)
            regulatory_actions[dim] = action
            
            # Track error
            error = abs(controller.setpoint - measurement)
            total_error += error
        
        # Calculate overall homeostatic error
        homeostatic_error = total_error / len(self.controllers)
        
        # Apply regulatory actions
        self._apply_regulatory_actions(core, regulatory_actions)
        
        # Detect crisis
        crisis_level = self._detect_crisis(homeostatic_error)
        
        # Update adaptive setpoints based on performance
        self._adapt_setpoints(core, measurements, homeostatic_error)
        
        # Create homeostatic state
        state = HomeostaticState(
            energy_balance=measurements["energy_balance"],
            boundary_integrity=measurements["boundary_integrity"],
            cognitive_flow=measurements["cognitive_flow"],
            vpm_diversity=measurements["vpm_diversity"],
            homeostatic_error=homeostatic_error,
            crisis_level=crisis_level,
            regulatory_actions=regulatory_actions,
            setpoints=self.adaptive_setpoints.copy(),
            stability=self._calculate_stability()
        )
        
        # Record for history
        self.regulatory_history.append({
            "state": state,
            "timestamp": time.time()
        })
        if len(self.regulatory_history) > 1000:
            self.regulatory_history.pop(0)
            
        return state

    def set_setpoint(self, dim: str, value: float) -> bool:
        if dim not in self.controllers:
            return False
        v = float(value)
        # keep energy_balance wide, others 0..1 (match your clamps)
        if dim == "energy_balance":
            v = float(np.clip(v, 0.5, 2.0))
        else:
            v = float(np.clip(v, 0.0, 1.0))
        self.adaptive_setpoints[dim] = v
        self.controllers[dim].setpoint = v
        return True

    def _get_measurements(self, core) -> Dict[str, float]:
        """Get current measurements for all regulated dimensions"""
        # Energy balance (cognitive/metabolic)
        cognitive = core.energy.level("cognitive")
        metabolic = core.energy.level("metabolic")
        energy_balance = cognitive / (metabolic + 1e-8) if metabolic > 0 else float('inf')
        
        # Boundary integrity
        boundary_integrity = core.membrane.integrity
        
        # Cognitive flow (from triune system)
        cognitive_flow = 0.5
        if hasattr(core, "triune") and core.triune:
            health = core.triune.get_health_metrics()
            cognitive_flow = health["efficiency"]
        
        # VPM diversity
        vpm_diversity = 0.5
        if hasattr(core, "vpm_manager"):
            vpm_diversity = core.vpm_manager.diversity_score()
        
        return {
            "energy_balance": energy_balance,
            "boundary_integrity": boundary_integrity,
            "cognitive_flow": cognitive_flow,
            "vpm_diversity": vpm_diversity
        }

    def _apply_regulatory_actions(
        self, 
        core, 
        actions: Dict[str, float]
    ):
        """Apply regulatory actions to the core system"""
        # Energy balance regulation
        if abs(actions["energy_balance"]) > 0.05:
            # Adjust metabolic pathways
            factor = 1.0 + actions["energy_balance"] * 0.1
            if hasattr(core.energy, "adjust_pathway_rates"):
                core.energy.adjust_pathway_rates(factor)          # EnergyMetabolism path
            elif hasattr(core.energy, "adjust_c2m_bias"):
                core.energy.adjust_c2m_bias((factor - 1.0) * 0.1) # AutopoieticCore path

        
        # Boundary integrity regulation
        if actions["boundary_integrity"] > 0.1:
            # Increase boundary thickness
            core.membrane.thickness = min(1.0, core.membrane.thickness + actions["boundary_integrity"] * 0.2)
        elif actions["boundary_integrity"] < -0.1:
            # Decrease boundary thickness (carefully)
            core.membrane.thickness = max(0.1, core.membrane.thickness + actions["boundary_integrity"] * 0.1)
        
        # Cognitive flow regulation
        if hasattr(core, "triune") and core.triune:
            # Adjust attention weights based on cognitive flow
            if actions["cognitive_flow"] > 0.1:
                # Increase primate attention for better reasoning
                core.triune.update_attention(actions["cognitive_flow"])
            elif actions["cognitive_flow"] < -0.1:
                # Increase reptilian attention for stability
                core.triune.update_attention(-actions["cognitive_flow"])
        
        # VPM diversity regulation
        if hasattr(core, "vpm_manager"):
            if actions["vpm_diversity"] > 0.1:
                # Increase mutation rate to boost diversity
                core.vpm_manager.mutation_rate = min(0.5, core.vpm_manager.mutation_rate * 1.2)
            elif actions["vpm_diversity"] < -0.1:
                # Decrease mutation rate to stabilize
                core.vpm_manager.mutation_rate = max(0.05, core.vpm_manager.mutation_rate * 0.8)

    def _detect_crisis(self, homeostatic_error: float) -> float:
        """Detect and track crisis levels based on homeostatic error"""
        # Crisis level is normalized homeostatic error
        crisis_level = min(1.0, homeostatic_error / self.crisis_threshold)
        
        # Update crisis counter
        if crisis_level > 0.8:
            self.crisis_counter += 1
        else:
            self.crisis_counter = max(0, self.crisis_counter - 1)
        
        # Log critical crises
        if crisis_level > 0.9 and self.crisis_counter % 5 == 0:
            log.warning(f"Homeostasis crisis detected: level={crisis_level:.2f}, counter={self.crisis_counter}")
        
        return crisis_level

    def _adapt_setpoints(
        self, 
        core, 
        measurements: Dict[str, float], 
        homeostatic_error: float
    ):
        """Adapt setpoints based on performance and experience"""
        # Calculate performance metric (1 - normalized error)
        performance = max(0.0, min(1.0, 1.0 - (homeostatic_error / self.crisis_threshold)))
        
        # Track performance for adaptation
        self.performance_history.append(performance)
        if len(self.performance_history) > self.max_performance_history:
            self.performance_history.pop(0)
        
        # Only adapt if we have enough data
        if len(self.performance_history) < 10:
            return
            
        # Calculate recent performance trend
        recent_perf = self.performance_history[-10:]
        perf_trend = np.mean(recent_perf[5:]) - np.mean(recent_perf[:5])
        
        # Adapt setpoints based on performance
        for dim in self.adaptive_setpoints:
            # Get controller for this dimension
            controller = self.controllers[dim]
            
            # Current measurement
            measurement = measurements[dim]
            
            # If performance is good and stable, lock in current setpoint
            if performance > 0.7 and abs(perf_trend) < 0.05:
                # Move setpoint toward current measurement (stabilize around what works)
                self.adaptive_setpoints[dim] = 0.9 * self.adaptive_setpoints[dim] + 0.1 * measurement
            # If performance is poor, explore new setpoints
            elif performance < 0.3:
                # Move setpoint away from problematic areas
                if measurement < controller.setpoint:
                    self.adaptive_setpoints[dim] = min(1.0, controller.setpoint + 0.1)
                else:
                    self.adaptive_setpoints[dim] = max(0.0, controller.setpoint - 0.1)
            
            # Constrain setpoints to reasonable ranges
            if dim == "energy_balance":
                self.adaptive_setpoints[dim] = np.clip(self.adaptive_setpoints[dim], 0.5, 2.0)
            else:  # All other dimensions are 0-1
                self.adaptive_setpoints[dim] = np.clip(self.adaptive_setpoints[dim], 0.2, 0.95)
            
            # Update controller setpoint
            controller.setpoint = self.adaptive_setpoints[dim]
            
            # Adapt controller parameters
            controller.adapt(performance)

    def _calculate_stability(self) -> float:
        """Calculate recent stability metric from regulatory history"""
        if not self.regulatory_history or len(self.regulatory_history) < 10:
            return 0.5
            
        # Get last 10 regulatory actions
        recent = self.regulatory_history[-10:]
        
        # Calculate variance in homeostatic error
        errors = [r["state"].homeostatic_error for r in recent]
        stability = 1.0 / (1.0 + np.var(errors))
        
        return float(stability)

    def get_telemetry(self) -> Dict[str, Any]:
        """Get telemetry data for visualization and monitoring"""
        if not self.regulatory_history:
            return {
                "health": 0.5,
                "crisis_level": 0.0,
                "stability": 0.5,
                "setpoints": self.adaptive_setpoints.copy()
            }
        
        # Get most recent state
        latest = self.regulatory_history[-1]["state"]
        
        # Calculate health score (1 - normalized homeostatic error)
        health = max(0.0, min(1.0, 1.0 - (latest.homeostatic_error / self.crisis_threshold)))
        
        # Determine trend
        trend = "stable"
        if len(self.regulatory_history) >= 5:
            recent_health = [1.0 - (r["state"].homeostatic_error / self.crisis_threshold) 
                            for r in self.regulatory_history[-5:]]
            if recent_health[-1] > recent_health[0] + 0.1:
                trend = "improving"
            elif recent_health[-1] < recent_health[0] - 0.1:
                trend = "declining"
        
        return {
            "health": health,
            "crisis_level": latest.crisis_level,
            "stability": latest.stability,
            "trend": trend,
            "setpoints": self.adaptive_setpoints.copy(),
            "regulatory_actions": latest.regulatory_actions,
            "measurements": {
                "energy_balance": latest.energy_balance,
                "boundary_integrity": latest.boundary_integrity,
                "cognitive_flow": latest.cognitive_flow,
                "vpm_diversity": latest.vpm_diversity
            }
        }
