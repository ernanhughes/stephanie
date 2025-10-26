# stephanie/components/jitter/evolution.py
"""
JAS Evolution Manager
--------------------
Manages population-level evolution across multiple Jitter instances.
Implements selection, competition, and environmental adaptation.
"""
from __future__ import annotations

import asyncio
import numpy as np
import logging
from typing import Dict, Any, List
from dataclasses import dataclass

logger = logging.getLogger(__file__)

@dataclass
class PopulationMetrics:
    total_organisms: int
    average_fitness: float
    fitness_variance: float
    generation_diversity: float
    extinction_risk: float
    adaptation_rate: float

class EvolutionManager:
    """
    Manages evolutionary processes across Jitter population
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.population: List[Any] = []  # List of Jitter instances
        self.environment = config.get("environment", {})
        self.selection_pressure = config.get("selection_pressure", 0.7)
        self.carrying_capacity = config.get("carrying_capacity", 10)
        self.competition_interval = config.get("competition_interval", 1000)  # ticks
        
        # Evolutionary tracking
        self.generation = 0
        self.extinction_events = 0
        self.adaptation_history: List[float] = []
        
        logger.info("🌍 EvolutionManager initialized")
    
    async def manage_population(self):
        """Main evolution management loop"""
        while True:
            await asyncio.sleep(self.competition_interval)
            
            try:
                # 1. Assess population health
                metrics = self._calculate_population_metrics()
                
                # 2. Apply environmental pressures
                self._apply_environmental_pressure(metrics)
                
                # 3. Natural selection
                await self._natural_selection(metrics)
                
                # 4. Track adaptation
                self._track_adaptation(metrics)
                
                logger.info(f"🌍 Evolution cycle: {metrics.total_organisms} organisms, "
                           f"avg_fitness={metrics.average_fitness:.3f}, "
                           f"extinction_risk={metrics.extinction_risk:.3f}")
                
                self.generation += 1
                
            except Exception as e:
                logger.error(f"Evolution management failed: {e}")
    
    def add_organism(self, jitter_instance: Any):
        """Add a new organism to the population"""
        self.population.append(jitter_instance)
        logger.debug(f"🌍 Added organism to population: {getattr(jitter_instance, 'genetic_id', 'unknown')}")
    
    def remove_organism(self, jitter_instance: Any):
        """Remove organism from population (death)"""
        if jitter_instance in self.population:
            self.population.remove(jitter_instance)
            logger.debug(f"🌍 Removed organism from population: {getattr(jitter_instance, 'genetic_id', 'unknown')}")
    
    def _calculate_population_metrics(self) -> PopulationMetrics:
        """Calculate current population health metrics"""
        if not self.population:
            return PopulationMetrics(0, 0.0, 0.0, 0.0, 1.0, 0.0)
        
        # Calculate individual fitness scores
        fitness_scores = []
        generations = []
        
        for organism in self.population:
            # Simple fitness proxy from energy and boundary
            if hasattr(organism, 'get_current_state'):
                state = organism.get_current_state()
                fitness = (state.energy_levels["reserve"] / 100.0 + state.boundary_integrity) / 2
                fitness_scores.append(fitness)
            
            # Track generations
            if hasattr(organism, 'genetic_material'):
                generations.append(organism.genetic_material.generation)
        
        if not fitness_scores:
            return PopulationMetrics(len(self.population), 0.5, 0.1, 0.5, 0.5, 0.0)
        
        avg_fitness = np.mean(fitness_scores)
        fitness_variance = np.var(fitness_scores)
        
        # Generation diversity
        if generations:
            gen_diversity = len(set(generations)) / len(generations)
        else:
            gen_diversity = 0.0
        
        # Extinction risk (inverse of population health)
        extinction_risk = max(0.0, 1.0 - (avg_fitness * len(self.population) / self.carrying_capacity))
        
        # Adaptation rate (from history)
        if self.adaptation_history:
            adaptation_rate = np.mean(self.adaptation_history[-10:])
        else:
            adaptation_rate = 0.0
        
        return PopulationMetrics(
            total_organisms=len(self.population),
            average_fitness=avg_fitness,
            fitness_variance=fitness_variance,
            generation_diversity=gen_diversity,
            extinction_risk=extinction_risk,
            adaptation_rate=adaptation_rate
        )
    
    def _apply_environmental_pressure(self, metrics: PopulationMetrics):
        """Apply environmental selection pressure"""
        # Overpopulation pressure
        if metrics.total_organisms > self.carrying_capacity:
            excess = metrics.total_organisms - self.carrying_capacity
            survival_probability = self.carrying_capacity / metrics.total_organisms
            
            # Randomly cull excess population
            organisms_to_remove = []
            for organism in self.population:
                if np.random.random() > survival_probability:
                    organisms_to_remove.append(organism)
                    if len(organisms_to_remove) >= excess:
                        break
            
            for organism in organisms_to_remove:
                self.remove_organism(organism)
                logger.info(f"🌍 Environmental pressure: removed organism {getattr(organism, 'genetic_id', 'unknown')}")
    
    async def _natural_selection(self, metrics: PopulationMetrics):
        """Apply natural selection based on fitness"""
        if len(self.population) < 2:
            return
        
        # Tournament selection for reproduction opportunities
        tournament_size = min(3, len(self.population))
        selected_for_reproduction = []
        
        for _ in range(min(5, len(self.population) // 2)):  # Number of reproduction slots
            # Random tournament
            tournament = np.random.choice(self.population, tournament_size, replace=False)
            
            # Select fittest from tournament
            fittest = max(tournament, key=lambda org: self._calculate_individual_fitness(org))
            selected_for_reproduction.append(fittest)
        
        # Allow selected organisms to reproduce
        for organism in selected_for_reproduction:
            if hasattr(organism, 'reproduction_system'):
                offspring_genetics = organism.reproduction_system.reproduce(organism)
                if offspring_genetics:
                    # In a full implementation, would create new Jitter instance here
                    logger.info(f"🌍 Natural selection: {getattr(organism, 'genetic_id', 'unknown')} "
                               f"selected for reproduction")
    
    def _calculate_individual_fitness(self, organism: Any) -> float:
        """Calculate fitness for individual organism"""
        if not hasattr(organism, 'get_current_state'):
            return 0.5
        
        state = organism.get_current_state()
        
        # Base fitness from energy and boundary
        base_fitness = (state.energy_levels["reserve"] / 100.0 + state.boundary_integrity) / 2
        
        # Bonus for cognitive health
        if hasattr(organism, 'triune'):
            cog_health = organism.triune.get_cognitive_health()
            cognitive_bonus = (cog_health.get("stability", 0.5) + cog_health.get("efficiency", 0.5)) / 4
            base_fitness += cognitive_bonus
        
        return min(1.0, base_fitness)
    
    def _track_adaptation(self, metrics: PopulationMetrics):
        """Track population adaptation over time"""
        # Adaptation is improvement in average fitness
        if len(self.adaptation_history) > 0:
            last_avg_fitness = self.adaptation_history[-1]
            adaptation = metrics.average_fitness - last_avg_fitness
        else:
            adaptation = 0.0
        
        self.adaptation_history.append(metrics.average_fitness)
        
        # Keep history manageable
        if len(self.adaptation_history) > 100:
            self.adaptation_history = self.adaptation_history[-100:]
    
    def get_evolution_stats(self) -> Dict[str, Any]:
        """Get evolution statistics for monitoring"""
        metrics = self._calculate_population_metrics()
        
        return {
            "population_size": metrics.total_organisms,
            "average_fitness": metrics.average_fitness,
            "generation": self.generation,
            "extinction_events": self.extinction_events,
            "carrying_capacity": self.carrying_capacity,
            "adaptation_rate": metrics.adaptation_rate,
            "diversity": metrics.generation_diversity
        }
