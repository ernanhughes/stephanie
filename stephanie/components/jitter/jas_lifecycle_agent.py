"""
JASLifecycleAgent
=================
The central orchestrator that brings all JAS components together in a continuous
tick-based loop, implementing the complete autopoietic cycle.

Key Features:
- Integrated metabolic, cognitive, and homeostatic processing
- Crisis detection and response protocols
- Reproduction system integration
- Comprehensive telemetry collection
- Safe shutdown procedures with legacy preservation
"""

import asyncio
import torch
import time
import random
import logging
from typing import Dict, Any, Optional

from stephanie.components.jitter.jas_core import AutopoieticCore
from stephanie.components.jitter.jas_triune import TriuneCognition
from stephanie.components.jitter.jas_homeostasis import EnhancedHomeostasis
from stephanie.components.jitter.telemetry.jas_telemetry import JASTelemetry
from stephanie.agents.base_agent import BaseAgent
from stephanie.components.jitter.telemetry.jas_telemetry import VitalSigns
from stephanie.scoring.vpm_scorable import VPMScorable  
from stephanie.data.score_bundle import ScoreBundle
from stephanie.components.jitter.apoptosis import ApoptosisSystem


import logging
log = logging.getLogger("stephanie.jas.lifecycle")

class JASLifecycleAgent(BaseAgent):
    """
    The lifecycle agent that orchestrates the complete Jitter Autopoietic System.
    
    This agent:
    - Manages the continuous autopoietic cycle
    - Coordinates all JAS components
    - Handles reproduction and legacy preservation
    - Implements safe shutdown procedures
    - Provides comprehensive telemetry
    """
    
    def __init__(self, cfg: Dict[str, Any], memory, container, logger=None):
        super().__init__(cfg, memory, container, logger)
        self.cfg = cfg
        self.memory = memory
        self.tick = 0
        self.running = False
        
        # Initialize JAS components
        self.core = None
        self.triune = None
        self.homeostasis = None
        self.telemetry = None
        
        # Reproduction system
        self.reproduction_ready = False
        self.last_reproduction = 0
        self.reproduction_interval = cfg.get("reproduction_interval", 1000)
        
        # Apoptosis (programmed cell death) system
        self.apoptosis_system = None

                # VPM scoring config
        self.vpm_cfg = (cfg or {}).get("vpm_scoring", {})
        self.vpm_enabled = self.vpm_cfg.get("enabled", True)
        self.vpm_dimensions = self.vpm_cfg.get("dimensions", [
            "clarity","novelty","confidence","contradiction","coherence","complexity","alignment","vpm_overall"
        ])
        self.vpm_dimension_weights = self.vpm_cfg.get("dimension_weights", {})      # per-dimension weights
        self.vpm_dimension_order   = self.vpm_cfg.get("dimension_order", [])        # most→least important
        self.vpm_force_rescore     = self.vpm_cfg.get("force_rescore", False)
        self.vpm_save_results      = self.vpm_cfg.get("save_results", True)
        self.scoring = self.container.get("scoring")
        self.zm = self.container.get("zeromodel")
        
        log.info("JASLifecycleAgent initialized")

    async def initialize(self) -> bool:
        """Initialize all JAS components"""
        try:
            # Load required models from memory
                
            # Initialize core metabolic system
            self.core = AutopoieticCore(
                cfg=self.cfg.get("core", {}),
                container=self.container,
                memory=self.memory
            )
            
            # Initialize triune cognition
            self.triune = TriuneCognition(
                cfg=self.cfg.get("triune", {}),
                memory=self.memory,
                container=self.container,
                logger=self.logger,
            )
            
            # Initialize homeostasis system
            self.homeostasis = EnhancedHomeostasis(
                cfg=self.cfg.get("homeostasis", {})
            )
            
            # Initialize telemetry system
            self.telemetry = JASTelemetry(
                cfg=self.cfg.get("telemetry", {}),
                subject=self.cfg.get("telemetry_subject", "arena.jitter.telemetry"),
                interval=self.cfg.get("telemetry_interval", 1.0)
            )
            await self.telemetry.init()
            
            # Initialize apoptosis system
            self.apoptosis_system = ApoptosisSystem(
                cfg=self.cfg.get("apoptosis", {})
            )
            
            log.info("JAS components initialized successfully")
            return True
            
        except Exception as e:
            log.error(f"JAS initialization failed: {str(e)}", exc_info=True)
            return False

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the JAS lifecycle agent.
        
        This method:
        - Starts the continuous autopoietic cycle
        - Handles reproduction and apoptosis
        - Publishes telemetry
        - Ensures safe shutdown
        """
        if not await self.initialize():
            return {"status": "error", "message": "Initialization failed"}
            
        self.running = True
        log.info("JAS lifecycle started")
        
        try:
            while self.running and self.core.energy.alive():
                await self._tick()
                await asyncio.sleep(self.cfg.get("tick_interval", 1.0))
                
        except asyncio.CancelledError:
            log.info("JAS lifecycle cancelled")
        except Exception as e:
            log.error(f"JAS lifecycle error: {str(e)}", exc_info=True)
        finally:
            await self._shutdown()
            
        status = "reproduced" if self.reproduction_ready else "apoptotic"
        log.info(f"JAS lifecycle completed ({status})")
        return {"status": status}

    async def _tick(self):
        """Execute a single autopoietic cycle tick"""
        self.tick += 1
        
        try:
            # 1. Get sensory input (from memory or external sources)
            sensory_input = await self._get_sensory_input()
            
            # 2. Process through triune cognition
            cognitive_state = self.triune(sensory_input)
            
            # 2.5. Score current VPM (non-blocking wrapper around CPU-bound scoring)
            scorable = VPMScorable.from_tensor(
                sensory_input,
                run_id=self.run_id,
                step=self.tick,
                meta={"source": "jas_tick"}
            )

            vpm_eval = self.scoring.score("vpm", scorable, {}, dimensions=["clarity"])
            if vpm_eval is not None:
                self.logger.log("VPMScored", vpm_eval)  # structured JSON event

            # 3. Update energy based on cognitive processing
            self.core.energy.replenish("cognitive", cognitive_state.cognitive_energy)
            
            # 4. Run the core autopoietic cycle
            vitals = self.core.cycle(sensory_input)
            
            # 5. Apply homeostatic regulation
            homeo_state = self.homeostasis.regulate(self.core)
            
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
            
            # 9. Log progress
            self._log_progress(vital_signs, homeo_state)
            
        except Exception as e:
            log.error(f"JAS tick error: {str(e)}", exc_info=True)

    async def _get_sensory_input(self) -> torch.Tensor:
        """Get sensory input for the next tick"""
        # In production, this would pull from real data sources
        if random.random() < 0.1:
            # Occasionally introduce external input
            return await self.memory.vpms.get_external_input()
        else:
            # Normal operation: pull from VPM store
            return self.memory.vpms.get_random_embedding()

    def _check_reproduction(self, vital_signs: VitalSigns):
        """Check if reproduction conditions are met"""
        # Reproduction requires sufficient reserve energy and stability
        if (self.core.energy.level("reserve") > self.cfg.get("reproduction_energy_threshold", 80) and
            vital_signs.health_score > self.cfg.get("reproduction_health_threshold", 0.7) and
            self.tick - self.last_reproduction > self.reproduction_interval):
            
            self.reproduction_ready = True
            log.info(f"Reproduction ready (tick={self.tick}, reserve={self.core.energy.level('reserve'):.1f})")
            
            # Trigger reproduction if enabled
            if self.cfg.get("enable_reproduction", True):
                asyncio.create_task(self._reproduce())

    async def _reproduce(self):
        """Create a new JAS instance (offspring)"""
        log.info("Initiating reproduction process")
        
        try:
            # Create reproduction package
            reproduction_data = {
                "core_state": self.core.snapshot(),
                "triune_state": self.triune.get_recent_states(),
                "homeostasis_state": self.homeostasis.get_telemetry(),
                "telemetry_history": self.telemetry.get_vital_signs_history(100),
                "generation": self.core.generation + 1,
                "parent_id": self.core.id,
                "timestamp": time.time(),
                "tick": self.tick
            }
            
            # Apply controlled variation (biological mutation)
            mutated_data = self._apply_variation(reproduction_data)
            
            # Save reproduction data for offspring initialization
            reproduction_id = f"reproduction_{int(time.time())}"
            self.memory.store_reproduction_data(reproduction_id, mutated_data)
            
            # Record reproduction event
            self.memory.log_event(
                "jas_reproduction",
                {
                    "reproduction_id": reproduction_id,
                    "generation": mutated_data["generation"],
                    "parent_id": mutated_data["parent_id"]
                }
            )
            
            # Reset reproduction counter
            self.last_reproduction = self.tick
            self.reproduction_ready = False
            
            log.info(f"Reproduction successful (generation={mutated_data['generation']})")
            
        except Exception as e:
            log.error(f"Reproduction failed: {str(e)}", exc_info=True)

    def _apply_variation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply controlled variation to reproduction data"""
        # Copy data to avoid modifying original
        mutated = {k: v for k, v in data.items()}
        
        # Apply variation to core state
        if "core_state" in mutated:
            core_state = mutated["core_state"]
            # Slightly vary membrane properties
            core_state["membrane"]["thickness"] *= random.uniform(0.95, 1.05)
            core_state["membrane"]["integrity"] = min(1.0, max(0.0, core_state["membrane"]["integrity"] + random.uniform(-0.05, 0.05)))
            
            # Slightly vary energy pools
            for pool in ["cognitive", "metabolic", "reserve"]:
                if pool in core_state["energy"]:
                    # Apply proportional variation (more variation for lower energy values)
                    variation_factor = 0.1 + (1.0 - core_state["energy"][pool]/100.0) * 0.2
                    core_state["energy"][pool] *= random.uniform(
                        1.0 - variation_factor, 
                        1.0 + variation_factor
                    )
                    # Ensure energy values stay within valid range
                    core_state["energy"][pool] = max(0.0, min(100.0, core_state["energy"][pool]))

        # Apply variation to triune state
        if "triune_state" in mutated and mutated["triune_state"]:
            # Slightly vary attention weights
            last_state = mutated["triune_state"][-1]
            attention = last_state.attention_weights
            for layer in attention:
                attention[layer] *= random.uniform(0.95, 1.05)
            # Normalize to sum to 1
            total = sum(attention.values())
            for layer in attention:
                attention[layer] /= total
        
        # Apply variation to homeostasis state
        if "homeostasis_state" in mutated:
            # Slightly vary setpoints
            setpoints = mutated["homeostasis_state"]["setpoints"]
            for dim in setpoints:
                setpoints[dim] *= random.uniform(0.98, 1.02)
                # Constrain to reasonable ranges
                if dim == "energy_balance":
                    setpoints[dim] = max(0.5, min(2.0, setpoints[dim]))
                else:  # All other dimensions are 0-1
                    setpoints[dim] = max(0.2, min(0.95, setpoints[dim]))
        
        return mutated

    def _log_progress(self, vital_signs: VitalSigns, homeo_state):
        """Log progress for monitoring"""
        # Log at decreasing frequency as tick count increases
        log_interval = max(1, min(100, self.tick // 100))
        
        if self.tick % log_interval == 0:
            log.info(
                f"JAS Tick {self.tick} | "
                f"Health: {vital_signs.health_score:.2f} | "
                f"Energy: C={vital_signs.energy_cognitive:.1f}/M={vital_signs.energy_metabolic:.1f}/R={vital_signs.energy_reserve:.1f} | "
                f"Boundary: {vital_signs.boundary_integrity:.2f} | "
                f"VPMs: {vital_signs.vpm_count} (diversity={vital_signs.vpm_diversity:.2f})"
            )

    async def _shutdown(self):
        """Perform safe shutdown with legacy preservation"""
        self.running = False
        log.info("Initiating JAS shutdown procedure")
        
        try:
            # 1. Check if apoptosis is needed
            if self.apoptosis_system.should_initiate(self.core, self.homeostasis):
                log.info("Apoptosis initiated - preserving legacy")
                await self._preserve_legacy()
                
            # 2. Clean up resources
            if self.telemetry.js:
                await self.telemetry.js.close()
                
            # 3. Log final state
            log.info(f"JAS shutdown complete (tick={self.tick})")
            
        except Exception as e:
            log.error(f"JAS shutdown error: {str(e)}", exc_info=True)

    async def _preserve_legacy(self):
        """Preserve valuable knowledge before shutdown"""
        try:
            # Identify high-value VPMs
            high_value_vpms = self.memory.vpm_manager.get_high_value_vpms(
                min_value=self.cfg.get("legacy_min_value", 0.7),
                max_count=self.cfg.get("legacy_max_count", 50)
            )
            
            # Create legacy package
            legacy_data = {
                "generation": self.core.generation,
                "high_value_vpms": [v.to_dict() for v in high_value_vpms],
                "core_snapshot": self.core.snapshot(),
                "telemetry_summary": self.telemetry.get_health_trend(),
                "reproduction_history": self.memory.get_reproduction_events(self.core.id),
                "shutdown_reason": self.apoptosis_system.get_reason(self.core, self.homeostasis),
                "timestamp": time.time(),
                "tick": self.tick
            }
            
            # Save legacy data
            legacy_id = f"legacy_{int(time.time())}"
            self.memory.store_legacy_data(legacy_id, legacy_data)
            
            # Log preservation
            log.info(f"Legacy preserved ({len(high_value_vpms)} VPMs, id={legacy_id})")
            
        except Exception as e:
            log.error(f"Legacy preservation failed: {str(e)}", exc_info=True)

    async def stop(self):
        """Stop the lifecycle agent gracefully"""
        self.running = False
        log.info("JAS lifecycle stop requested")
        
        # Wait for current tick to complete
        await asyncio.sleep(self.cfg.get("tick_interval", 1.0))

