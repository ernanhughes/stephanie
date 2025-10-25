"""
Enhanced Lifecycle with Evolution
--------------------------------
Jitter lifecycle that includes reproduction and evolutionary management.
"""

import asyncio
import torch
import logging
from typing import Dict, Any

from .jas_core import AutopoieticCore
from .jas_triune import TriuneCognition
from .jas_homeostasis import EnhancedHomeostasis
from .telemetry.jas_telemetry import TelemetryBridge
from .jas_reproduction import ReproductionSystem, GeneticMaterial
from .jas_evolution import EvolutionManager

logger = logging.getLogger("stephanie.jas.evolution_lifecycle")

class EvolutionaryJitterLifecycle:
    """
    Jitter lifecycle with full evolutionary capabilities
    """
    
    def __init__(self, config: Dict[str, Any], models, managers):
        self.config = config
        self.models = models
        self.managers = managers
        
        # Initialize core systems
        self.core = self._initialize_core()
        self.telemetry = TelemetryBridge()
        
        # Evolutionary systems
        self.reproduction = ReproductionSystem(config.get("reproduction", {}))
        self.evolution_manager = EvolutionManager(config.get("evolution", {}))
        
        # Track organism identity
        self.genetic_id = getattr(self.core, 'genetic_id', 'root')
        self.evolution_manager.add_organism(self)
        
        logger.info(f"🧬 EvolutionaryJitterLifecycle initialized: {self.genetic_id}")
    
    def _initialize_core(self) -> AutopoieticCore:
        """Initialize core with genetic configuration"""
        core = AutopoieticCore(
            self.config, 
            self.models.ebt, 
            self.managers.vpm
        )
        
        # Add triune cognition
        core.triune = TriuneCognition(
            self.models.mrq, 
            self.models.ebt, 
            self.models.svm,
            self.config.get("cognition", {})
        )
        
        # Add homeostasis
        core.homeo = EnhancedHomeostasis(self.config.get("homeostasis", {}))
        
        # Add reproduction system to core
        core.reproduction_system = self.reproduction
        
        # Initialize genetic identity
        core.genetic_id = str(id(core))[-8:]
        core.genetic_material = GeneticMaterial(id=core.genetic_id)
        
        return core
    
    async def start(self):
        """Start the evolutionary lifecycle"""
        await self.telemetry.init()
        
        # Start evolution management in background
        asyncio.create_task(self.evolution_manager.manage_population())
        
        logger.info("🌱 Evolutionary lifecycle started")
        
        # Main lifecycle loop
        tick_count = 0
        while await self._run_tick(tick_count):
            tick_count += 1
            await asyncio.sleep(self.config.get("tick_interval", 1.0))
        
        # Handle death
        await self._handle_death()
    
    async def _run_tick(self, tick_count: int) -> bool:
        """Run one lifecycle tick"""
        try:
            # 1. Gather sensory input
            sensory_input = await self._gather_sensory_input()
            
            # 2. Cognitive processing
            cognitive_state = self.core.triune.evaluate(sensory_input)
            self.core.metabolism.energy_pools.replenish("cognitive", cognitive_state.cognitive_energy)
            
            # 3. Core autopoietic cycle
            cycle_result = self.core.cycle(sensory_input)
            if not cycle_result.success:
                logger.warning(f"Core cycle failed at tick {tick_count}")
                return False
            
            # 4. Homeostatic regulation
            self.core.homeo.regulate(self.core)
            
            # 5. Check for reproduction
            if self.reproduction.can_reproduce(self.core):
                offspring_genetics = self.reproduction.reproduce(self.core)
                if offspring_genetics:
                    await self._handle_offspring(offspring_genetics)
            
            # 6. Emit telemetry
            await self.telemetry.emit(self.core, self.core.homeo, cognitive_state)
            
            # 7. Check viability
            if not self.core.is_alive():
                logger.info(f"🧬 Organism {self.genetic_id} reached end of lifecycle")
                return False
            
            # Periodic logging
            if tick_count % 100 == 0:
                health = self.core.homeo.get_health_telemetry()
                evolution_stats = self.evolution_manager.get_evolution_stats()
                logger.info(f"🧬 Tick {tick_count}: health={health['health_score']:.3f}, "
                           f"population={evolution_stats['population_size']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Lifecycle tick failed: {e}")
            return False
    
    async def _gather_sensory_input(self) -> torch.Tensor:
        """Gather sensory input from environment"""
        # This would integrate with your actual sensory systems
        # For now, return mock embedding
        return torch.randn(1, 1024)
    
    async def _handle_offspring(self, offspring_genetics: GeneticMaterial):
        """Handle creation of new offspring organism"""
        try:
            # Create new Jitter instance with offspring genetics
            offspring_config = offspring_genetics.to_config()
            offspring_config.update(self.config)  # Inherit parent configuration
            
            # Create new lifecycle instance
            offspring_lifecycle = EvolutionaryJitterLifecycle(
                offspring_config, 
                self.models, 
                self.managers
            )
            
            # Set genetic material
            offspring_lifecycle.core.genetic_material = offspring_genetics
            offspring_lifecycle.genetic_id = offspring_genetics.id
            
            # Add to evolution manager
            self.evolution_manager.add_organism(offspring_lifecycle)
            
            # Start offspring lifecycle in background
            asyncio.create_task(offspring_lifecycle.start())
            
            logger.info(f"🧬 Offspring created: {offspring_genetics.id} "
                       f"(parent: {self.genetic_id}, generation: {offspring_genetics.generation})")
            
        except Exception as e:
            logger.error(f"Failed to create offspring: {e}")
    
    async def _handle_death(self):
        """Handle organism death"""
        logger.info(f"🧬 Organism {self.genetic_id} dying")
        
        # Remove from evolution manager
        self.evolution_manager.remove_organism(self)
        
        # Emit final telemetry
        try:
            final_telemetry = {
                "event": "death",
                "genetic_id": self.genetic_id,
                "lifetime_ticks": getattr(self, 'tick_count', 0),
                "final_health": self.core.homeo.get_health_telemetry()["health_score"],
                "offspring_produced": self.reproduction.offspring_count,
                "lineage_stats": self.reproduction.get_lineage_stats()
            }
            
            if self.telemetry.js:
                await self.telemetry.js.publish(
                    "arena.jitter.events",
                    str(final_telemetry).encode()
                )
        except Exception as e:
            logger.error(f"Final telemetry failed: {e}")
    
    def get_organism_stats(self) -> Dict[str, Any]:
        """Get organism statistics for monitoring"""
        health = self.core.homeo.get_health_telemetry()
        evolution_stats = self.evolution_manager.get_evolution_stats()
        lineage_stats = self.reproduction.get_lineage_stats()
        
        return {
            "genetic_id": self.genetic_id,
            "health_score": health["health_score"],
            "energy_reserve": self.core.metabolism.energy_pools.get_level("reserve"),
            "boundary_integrity": self.core.membrane.integrity,
            "offspring_produced": self.reproduction.offspring_count,
            "generation": getattr(self.core.genetic_material, 'generation', 0),
            "population_size": evolution_stats["population_size"],
            "lineage_diversity": lineage_stats["diversity"]
        }
