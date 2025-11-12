# stephanie/components/jitter/lifecycle/lifecycle_agent.py
"""
lifecycle_agent.py
==================
The main execution agent for the Jitter Autopoietic System.

This agent orchestrates the complete lifecycle of a Jitter organism,
managing all components and ensuring proper autopoietic operation.

Key Features:
- Complete autopoietic cycle execution
- Resource management and allocation
- Crisis detection and response
- Reproduction and legacy preservation
- Telemetry and monitoring
- Configuration-driven operation
- Integration with Stephanie's ecosystem
- Graceful shutdown with legacy preservation
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

import torch
from pydantic import BaseModel, Field, validator

from stephanie.agents.base_agent import BaseAgent
from stephanie.components.jitter.boundary import BoundaryMaintenance
from stephanie.components.jitter.cognition.triune import TriuneCognition
from stephanie.components.jitter.core.energy import EnergyPools
from stephanie.components.jitter.telemetry.jaf import JitterArtifactV0
from stephanie.components.jitter.telemetry.telemetry import (JASTelemetry,
                                                             VitalSigns)

log = logging.getLogger("stephanie.jitter.lifecycle.agent")

class LifecycleConfig(BaseModel):
    """Validated configuration for JASLifecycleAgent"""
    tick_interval: float = Field(1.0, ge=0.1, le=10.0, description="Time between ticks in seconds")
    enable_reproduction: bool = Field(True, description="Whether reproduction is enabled")
    reproduction_energy_threshold: float = Field(80.0, ge=0.0, le=100.0, description="Energy threshold for reproduction")
    max_runtime: Optional[int] = Field(None, description="Maximum runtime in seconds")
    health_check_interval: float = Field(5.0, ge=1.0, le=30.0, description="Health check interval in seconds")
    telemetry_interval: float = Field(1.0, ge=0.1, le=10.0, description="Telemetry publish interval in seconds")
    
    @validator('reproduction_energy_threshold')
    def validate_energy_threshold(cls, v):
        if v < 0 or v > 100:
            raise ValueError('reproduction_energy_threshold must be between 0 and 100')
        return v

class JASLifecycleAgent(BaseAgent):
    """
    The main lifecycle agent for the Jitter Autopoietic System.
    
    This agent orchestrates the complete lifecycle of a Jitter organism:
    - Runs the autopoietic core cycle
    - Processes sensory input through triune cognition
    - Applies homeostatic regulation
    - Manages crisis detection and response
    - Handles reproduction and apoptosis
    - Publishes telemetry and maintains legacy
    
    Key Features:
    - Complete autopoietic cycle execution
    - Resource management and allocation
    - Crisis detection and response
    - Reproduction and legacy preservation
    - Telemetry and monitoring
    - Configuration-driven operation
    - Integration with Stephanie's ecosystem
    - Graceful shutdown with legacy preservation
    """
    
    def __init__(self, cfg: Dict[str, Any], container, memory=None, logger=None):
        self.cfg = cfg
        self.container = container
        self.memory = memory
        self.logger = logger or logging.getLogger("stephanie.jitter.lifecycle.agent")
        
        # Validate configuration
        try:
            self.config = LifecycleConfig(**cfg)
            self.logger.info("JASLifecycleAgent configuration validated successfully")
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {str(e)}")
            # Use safe defaults
            self.config = LifecycleConfig()
        
        # Initialize components
        self.core = None
        self.triune = None
        self.homeostasis = None
        self.telemetry = None
        self.apoptosis = None
        self.reproduction = None
        self.running = False
        self.tick = 0
        self.start_time = None
        
        # Initialize telemetry history
        self.telemetry_history = []
        self.max_telemetry_history = 1000
        
        # Initialize health tracking
        self.health_history = []
        self.max_health_history = 1000
        
        log.info("JASLifecycleAgent initialized")
    
    async def initialize(self) -> bool:
        """Initialize all JAS components with proper integration"""
        try:
            # Get required services from container
            self.core = self.container.get("jas_membrane")
            if not self.core:
                self.core = AutopoieticCore(
                    cfg=self.cfg.get("core", {}),
                    ebt_model=self.memory.get_model("ebt", "document", "clarity"),
                    vpm_manager=self.memory.vpm_manager if hasattr(self.memory, 'vpm_manager') else None,
                    logger=self.logger
                )
            
            # Initialize triune cognition with proper dependencies
            self.triune = self.container.get("jas_triune")
            if not self.triune:
                self.triune = TriuneCognition(
                    cfg=self.cfg.get("triune", {}),
                    sense_making=self.container.get("jas_sense_making"),
                    production=self.container.get("jas_production"),
                    coupling=self.container.get("jas_coupling", None) or None
                )
            
            # Initialize homeostasis system
            self.homeostasis = self.container.get("jas_pid_controller")
            if not self.homeostasis:
                self.homeostasis = EnhancedHomeostasis(
                    cfg=self.cfg.get("homeostasis", {})
                )
            
            # Initialize telemetry system
            self.telemetry = self.container.get("jas_telemetry")
            if not self.telemetry:
                self.telemetry = JASTelemetry(
                    cfg=self.cfg.get("telemetry", {}),
                    subject=self.cfg.get("telemetry_subject", "arena.jitter.telemetry"),
                    interval=self.cfg.get("telemetry_interval", 1.0)
                )
                await self.telemetry.init()
            
            # Initialize apoptosis system
            self.apoptosis = self.container.get("jas_apoptosis")
            if not self.apoptosis:
                self.apoptosis = ApoptosisSystem(
                    cfg=self.cfg.get("apoptosis", {})
                )
            
            # Initialize reproduction system
            self.reproduction = self.container.get("jas_reproduction")
            if not self.reproduction:
                self.reproduction = ReproductionSystem(
                    cfg=self.cfg.get("reproduction", {})
                )
            
            # Initialize energy system
            self.energy = self.container.get("jas_energy")
            if not self.energy:
                self.energy = EnergyPools(cfg=self.cfg.get("energy", {}))
            
            # Initialize boundary maintenance
            self.boundary_maintenance = self.container.get("jas_boundary_maintenance")
            if not self.boundary_maintenance:
                self.boundary_maintenance = BoundaryMaintenance(
                    cfg=self.cfg.get("boundary_maintenance", {}),
                    membrane=self.core.membrane if hasattr(self.core, 'membrane') else None
                )
            
            self.logger.info("JAS components initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"JAS initialization failed: {str(e)}", exc_info=True)
            return False
    
    async def run(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run the JAS lifecycle agent with proper autopoietic integration.
        
        This method implements the fundamental autopoietic cycle:
        1. Receive environmental perturbation
        2. Process through triune cognition (create meaning)
        3. Update production network (self-production)
        4. Apply structural changes (adaptation)
        5. Regulate homeostasis
        6. Check for reproduction or apoptosis
        7. Publish telemetry
        8. Monitor and log progress
        
        Args:
            context: Context dictionary for pipeline integration
            
        Returns:
            Dictionary with execution results and metrics
        """
        if not await self.initialize():
            return {"status": "error", "message": "Initialization failed"}
            
        self.running = True
        self.start_time = time.time()
        self.logger.info("🌱 JAS lifecycle started")
        
        try:
            while self.running:
                await self._tick()
                # Respect tick interval
                await asyncio.sleep(max(0.0, self.config.tick_interval))
                
        except asyncio.CancelledError:
            self.logger.info("JAS lifecycle cancelled")
        except Exception as e:
            self.logger.error(f"JAS lifecycle error: {str(e)}", exc_info=True)
        finally:
            await self._shutdown()
            
        status = "reproduced" if self.reproduction and self.reproduction.get_offspring_count() > 0 else "apoptotic"
        self.logger.info(f"JAS lifecycle completed ({status})")
        return {"status": status}
    
    async def _tick(self):
        """Execute a single autopoietic cycle tick"""
        self.tick += 1
        
        try:
            # 1. Get sensory input from environment
            sensory_input = await self._get_sensory_input()
            
            # 2. Process through triune cognition (create meaning)
            cognitive_state = self.triune.process(
                sensory_input, 
                {"type": "default_action"}, 
                {"quality": 0.5}
            )
            
            # 3. Run core autopoietic cycle
            core_state = self.core.cycle(sensory_input)
            
            # 4. Apply boundary maintenance
            boundary_state = self.boundary_maintenance.maintain_boundary(
                core_state.get("energy", {}).get("metabolic", 0.0),
                {"cognitive": core_state.get("energy", {}).get("cognitive", 0.0)}
            )
            
            # 5. Apply homeostatic regulation
            homeostasis_state = self.homeostasis.regulate(self)
            
            # 6. Collect telemetry
            vital_signs = self.telemetry.collect(
                self.core,
                self.homeostasis,
                self.triune
            )
            
            # 7. Publish telemetry
            await self.telemetry.publish(vital_signs)
            
            # 8. Check for reproduction opportunity
            self._check_reproduction(vital_signs)
            
            # 9. Check for apoptosis
            self._check_apoptosis(vital_signs)
            
            # 10. Log progress
            self._log_progress(vital_signs, homeostasis_state)
            
        except Exception as e:
            self.logger.error(f"JAS tick error: {str(e)}", exc_info=True)
    
    async def _get_sensory_input(self) -> Dict[str, Any]:
        """Get sensory input from environment with structural coupling"""
        try:
            # Use ZeroModelService to generate a VPM
            zm = self.container.get("zeromodel-service-v2")
            if zm and hasattr(zm, "_pipeline"):
                # Generate VPM from random embedding
                random_emb = torch.randn(1024)
                vpm, _ = zm._pipeline.run(random_emb, {"enable_gif": False})
                return {
                    "type": "vpm",
                    "vpm_embedding": vpm.detach().cpu().numpy(),
                    "timestamp": time.time()
                }
            
            # Fallback to memory VPMs
            if hasattr(self.memory, "vpm_manager") and hasattr(self.memory.vpm_manager, "get_random_embedding"):
                embedding = self.memory.vpm_manager.get_random_embedding()
                return {
                    "type": "vpm",
                    "vpm_embedding": embedding,
                    "timestamp": time.time()
                }
                
        except Exception as e:
            log.warning(f"VPM generation failed: {str(e)}")
        
        # Final fallback
        return {
            "type": "random",
            "vpm_embedding": torch.randn(1024).numpy(),
            "timestamp": time.time()
        }
    
    def _check_reproduction(self, vital_signs: VitalSigns):
        """Check if reproduction conditions are met"""
        # Check if reproduction is enabled
        if not self.config.enable_reproduction:
            return
        
        # Check energy levels
        energy_balance = (vital_signs.energy_cognitive + 
                         vital_signs.energy_metabolic + 
                         vital_signs.energy_reserve) / 3.0
        
        # Check if energy threshold is met
        if energy_balance > self.config.reproduction_energy_threshold:
            self.logger.info(f"Reproduction opportunity at tick {self.tick}")
            # Attempt reproduction
            if self.reproduction and self.reproduction.can_reproduce(self.core):
                offspring = self.reproduction.reproduce(self.core)
                if offspring:
                    self.logger.info("Reproduction successful, offspring created")
    
    def _check_apoptosis(self, vital_signs: VitalSigns):
        """Check if apoptosis should be initiated"""
        # Check if apoptosis system is available
        if not self.apoptosis:
            return
        
        # Check if apoptosis conditions are met
        if self.apoptosis.should_initiate(self.core, self.homeostasis):
            log.warning("Apoptosis initiated - preserving legacy")
            asyncio.create_task(self._initiate_apoptosis())
    
    async def _initiate_apoptosis(self):
        """Initiate apoptosis process with legacy preservation"""
        self.logger.info("Initiating apoptosis process")
        
        try:
            # Create legacy artifact
            artifact = self._create_legacy_artifact()
            
            # Preserve legacy
            await self._preserve_legacy(artifact)
            
            # Stop running
            self.running = False
            
            self.logger.info("Apoptosis process completed")
            
        except Exception as e:
            self.logger.error(f"Apoptosis failed: {str(e)}", exc_info=True)
    
    def _create_legacy_artifact(self) -> JitterArtifactV0:
        """Create legacy artifact for preservation"""
        # This would typically be implemented with actual system state
        # For now, returning a minimal artifact
        return JitterArtifactV0(
            organism_id=f"jas_{int(time.time())}",
            generation=getattr(self.core, 'generation', 0),
            tick=self.tick,
            cause="apoptosis",
            timestamp=time.time()
        )
    
    async def _preserve_legacy(self, artifact: JitterArtifactV0):
        """Preserve legacy data before shutdown"""
        try:
            # Save legacy data
            legacy_id = f"legacy_{int(time.time())}"
            if self.memory:
                self.memory.store_legacy_data(legacy_id, artifact.to_dict())
            
            self.logger.info(f"Legacy preserved (id={legacy_id})")
            
        except Exception as e:
            self.logger.error(f"Legacy preservation failed: {str(e)}", exc_info=True)
    
    def _log_progress(self, vital_signs: VitalSigns, homeostasis_state: Dict[str, Any]):
        """Log progress for monitoring"""
        # Log at decreasing frequency as tick count increases
        log_interval = max(1, min(100, self.tick // 100))
        
        if self.tick % log_interval == 0:
            self.logger.info(
                f"JAS Tick {self.tick} | "
                f"Health: {vital_signs.health_score:.2f} | "
                f"Energy: C={vital_signs.energy_cognitive:.1f}/"
                f"M={vital_signs.energy_metabolic:.1f}/"
                f"R={vital_signs.energy_reserve:.1f} | "
                f"Boundary: {vital_signs.boundary_integrity:.2f} | "
                f"VPMs: {vital_signs.vpm_count} (diversity={vital_signs.vpm_diversity:.2f})"
            )
    
    async def _shutdown(self):
        """Perform safe shutdown with legacy preservation"""
        self.running = False
        self.logger.info("Initiating JAS shutdown procedure")
        
        try:
            # Check if apoptosis is needed
            if self.apoptosis and self.apoptosis.should_initiate(self.core, self.homeostasis):
                self.logger.info("Apoptosis initiated - preserving legacy")
                await self._initiate_apoptosis()
            
            # Clean up resources
            if self.telemetry and hasattr(self.telemetry, 'js'):
                await self.telemetry.js.close()
                
            # Log final state
            duration = time.time() - self.start_time
            self.logger.info(f"JAS shutdown complete (tick={self.tick}, duration={duration:.1f}s)")
            
        except Exception as e:
            self.logger.error(f"JAS shutdown error: {str(e)}", exc_info=True)
    
    def get_health_metrics(self) -> Dict[str, Any]:
        """Get health metrics for monitoring and analysis"""
        return {
            "current_tick": self.tick,
            "uptime_seconds": time.time() - self.start_time if self.start_time else 0,
            "reproduction_count": self.reproduction.get_offspring_count() if self.reproduction else 0,
            "apoptosis_count": 0,  # Would be tracked separately
            "system_state": "running" if self.running else "stopped"
        }
    
    def get_telemetry_history(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get recent telemetry history for analysis"""
        return [
            {
                "tick": v.tick,
                "health_score": v.health_score,
                "crisis_level": v.crisis_level,
                "boundary_integrity": v.boundary_integrity,
                "energy_cognitive": v.energy_cognitive,
                "energy_metabolic": v.energy_metabolic,
                "energy_reserve": v.energy_reserve
            }
            for v in self.telemetry.history[-n:] if hasattr(self, 'telemetry') and self.telemetry.history
        ]

# Integration modules
class SSPIntegration:
    """Integration with the Stephanine Planning System (SSP)"""
    
    def __init__(self, jitter_agent, ssp_client=None):
        self.jitter = jitter_agent
        self.ssp_client = ssp_client
        self.integration_enabled = ssp_client is not None
    
    async def get_context_snapshot(self) -> Dict[str, Any]:
        """Get lightweight context snapshot for SSP"""
        if not self.jitter.core:
            return {}
        
        latest_state = self.jitter.triune.state_history[-1] if self.jitter.triune.state_history else None
        
        return {
            "jitter_health": self.jitter.homeostasis.get_telemetry()["health"],
            "cognitive_state": {
                "integrated": latest_state.integrated if latest_state else 0.5,
                "veto_layer": latest_state.layer_veto if latest_state else "none",
                "threat_level": latest_state.threat_level if latest_state else 0.5
            },
            "energy_status": {
                "metabolic": self.jitter.core.energy.level("metabolic"),
                "cognitive": self.jitter.core.energy.level("cognitive")
            },
            "crisis_level": self.jitter.homeostasis.get_telemetry()["crisis_level"],
            "timestamp": time.time()
        }
    
    def apply_ssp_feedback(self, feedback: Dict[str, Any]):
        """Apply feedback from SSP episodes"""
        if not self.integration_enabled:
            return
        
        # Adjust attention weights based on SSP performance
        if "episode_quality" in feedback:
            quality = feedback["episode_quality"]
            # This would be implemented based on the actual feedback mechanism
            pass
        
        # Adjust homeostasis based on task complexity
        if "task_complexity" in feedback:
            complexity = feedback["task_complexity"]
            # This would be implemented based on the actual feedback mechanism
            pass

class RewardShaper:
    """Handles reward shaping for learning and adaptation"""
    
    def __init__(self, jitter_agent):
        self.jitter = jitter_agent
        self.reward_history = []
        self.max_history = 1000
    
    def calculate_reward(self, feedback: Dict[str, Any]) -> float:
        """
        Calculate reward based on system performance and feedback.
        
        Args:
            feedback: Feedback from SSP or other systems
            
        Returns:
            Reward value (0-1)
        """
        # Base reward calculation
        reward = 0.0
        
        # Health-based reward
        health = self.jitter.homeostasis.get_telemetry().get("health", 0.5)
        reward += health * 0.4
        
        # Energy-based reward
        energy_balance = self.jitter.core.energy.level("metabolic") + self.jitter.core.energy.level("cognitive")
        energy_score = min(1.0, energy_balance / 100.0)
        reward += energy_score * 0.3
        
        # Boundary integrity reward
        boundary = self.jitter.core.membrane.integrity
        reward += boundary * 0.3
        
        # Normalize to 0-1
        reward = max(0.0, min(1.0, reward))
        
        # Store in history
        self.reward_history.append({
            "timestamp": time.time(),
            "reward": reward,
            "feedback": feedback
        })
        
        if len(self.reward_history) > self.max_history:
            self.reward_history.pop(0)
        
        return reward
    
    def get_reward_history(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get recent reward history"""
        return self.reward_history[-n:]