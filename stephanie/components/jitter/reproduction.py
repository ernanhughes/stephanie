# stephanie/components/jitter/reproduction.py
"""
JAS Reproduction System
-----------------------
Biologically plausible reproduction with genetic inheritance and mutation.
Creates new Jitter instances when conditions are favorable.
"""
from __future__ import annotations

import logging
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("stephanie.jas.reproduction")

@dataclass
class GeneticMaterial:
    """Genetic blueprint for a Jitter organism"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    parent_id: Optional[str] = None
    generation: int = 0
    
    # Core physiological parameters
    membrane_thickness: float = 1.0
    metabolic_efficiency: float = 1.0
    cognitive_gain: float = 0.5
    boundary_resilience: float = 1.0
    
    # Cognitive architecture parameters  
    attention_weights: List[float] = field(default_factory=lambda: [0.3, 0.3, 0.4])
    energy_conversion_rate: float = 0.8
    stress_tolerance: float = 0.5
    
    # Mutation history
    mutation_history: List[str] = field(default_factory=list)
    
    def mutate(self, mutation_rate: float = 0.1) -> 'GeneticMaterial':
        """Create mutated offspring"""
        child = deepcopy(self)
        child.parent_id = self.id
        child.generation = self.generation + 1
        child.id = str(uuid.uuid4())[:8]
        
        # Apply mutations
        mutations = []
        
        # Membrane mutations
        if np.random.random() < mutation_rate:
            child.membrane_thickness *= np.random.uniform(0.9, 1.1)
            mutations.append(f"membrane_thickness→{child.membrane_thickness:.3f}")
        
        # Metabolic mutations
        if np.random.random() < mutation_rate:
            child.metabolic_efficiency *= np.random.uniform(0.95, 1.05)
            mutations.append(f"metabolic_efficiency→{child.metabolic_efficiency:.3f}")
        
        # Cognitive mutations
        if np.random.random() < mutation_rate:
            child.cognitive_gain *= np.random.uniform(0.9, 1.1)
            mutations.append(f"cognitive_gain→{child.cognitive_gain:.3f}")
        
        # Attention weight mutations
        if np.random.random() < mutation_rate:
            # Slightly shift attention distribution
            shift = np.random.normal(0, 0.05, 3)
            new_weights = np.array(child.attention_weights) + shift
            new_weights = np.clip(new_weights, 0.1, 0.8)  # Keep reasonable bounds
            new_weights = new_weights / new_weights.sum()  # Renormalize
            child.attention_weights = new_weights.tolist()
            mutations.append(f"attention_weights→[{', '.join(f'{w:.3f}' for w in new_weights)}]")
        
        # Stress tolerance mutations
        if np.random.random() < mutation_rate:
            child.stress_tolerance = np.clip(child.stress_tolerance + np.random.normal(0, 0.1), 0.1, 1.0)
            mutations.append(f"stress_tolerance→{child.stress_tolerance:.3f}")
        
        child.mutation_history = mutations
        return child
    
    def to_config(self) -> Dict[str, Any]:
        """Convert genetic material to Jitter configuration"""
        return {
            "membrane": {
                "thickness": self.membrane_thickness,
                "resilience": self.boundary_resilience
            },
            "energy_metabolism": {
                "efficiency": self.metabolic_efficiency,
                "conversion_rate": self.energy_conversion_rate
            },
            "cognition": {
                "energy_gain": self.cognitive_gain,
                "attention_weights": self.attention_weights
            },
            "stress_tolerance": self.stress_tolerance,
            "genetic_id": self.id,
            "generation": self.generation,
            "parent_id": self.parent_id
        }

class ReproductionSystem:
    """
    Manages Jitter reproduction based on fitness and environmental conditions
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.reproduction_threshold = config.get("reproduction_threshold", 0.7)
        self.mutation_rate = config.get("mutation_rate", 0.1)
        self.max_offspring = config.get("max_offspring", 3)
        self.offspring_energy_cost = config.get("offspring_energy_cost", 0.3)
        
        # Lineage tracking
        self.lineage: List[GeneticMaterial] = []
        self.offspring_count = 0
        self.total_generations = 0
        
        logger.info("🧬 ReproductionSystem initialized")
    
    def can_reproduce(self, core) -> bool:
        """Check if organism can reproduce based on fitness and energy"""
        if self.offspring_count >= self.max_offspring:
            return False
        
        # Check energy reserves
        energy_reserve = core.metabolism.energy_pools.get_level("reserve")
        if energy_reserve < self.reproduction_threshold * 100:  # Scale to energy units
            return False
        
        # Check overall health and stability
        health_metrics = self._calculate_fitness(core)
        fitness_score = health_metrics["fitness"]
        
        return fitness_score > self.reproduction_threshold
    
    def reproduce(self, core) -> Optional[GeneticMaterial]:
        """Create offspring genetic material"""
        if not self.can_reproduce(core):
            return None
        
        try:
            # Extract current genetic state
            current_genetics = self._extract_genetics(core)
            
            # Create mutated offspring
            offspring_genetics = current_genetics.mutate(self.mutation_rate)
            
            # Deduct reproduction energy cost
            reproduction_cost = core.metabolism.energy_pools.total_energy() * self.offspring_energy_cost
            core.metabolism.energy_pools.consume("reserve", reproduction_cost)
            
            # Track lineage
            self.lineage.append(offspring_genetics)
            self.offspring_count += 1
            self.total_generations = max(self.total_generations, offspring_genetics.generation)
            
            logger.info(f"🧬 Reproduction: {core.genetic_id} → {offspring_genetics.id} "
                       f"(gen {offspring_genetics.generation})")
            logger.info(f"🧬 Mutations: {', '.join(offspring_genetics.mutation_history)}")
            
            return offspring_genetics
            
        except Exception as e:
            logger.error(f"Reproduction failed: {e}")
            return None
    
    def _extract_genetics(self, core) -> GeneticMaterial:
        """Extract genetic material from current organism state"""
        # If core already has genetics, use them
        if hasattr(core, 'genetic_material'):
            return core.genetic_material
        
        # Otherwise create from current state
        return GeneticMaterial(
            membrane_thickness=core.membrane.thickness,
            metabolic_efficiency=getattr(core.metabolism, 'efficiency', 1.0),
            cognitive_gain=getattr(core.triune, 'energy_gain', 0.5),
            boundary_resilience=core.membrane.integrity,
            attention_weights=getattr(core.triune.energy_modulator.attention_weights, 'detach', lambda: [0.3,0.3,0.4])().tolist(),
            stress_tolerance=getattr(core, 'stress_tolerance', 0.5)
        )
    
    def _calculate_fitness(self, core) -> Dict[str, float]:
        """Calculate fitness score for reproduction decision"""
        state = core.get_current_state()
        
        # Energy fitness (higher reserve = better)
        energy_fitness = min(1.0, state.energy_levels["reserve"] / 100.0)
        
        # Boundary fitness (stable boundary = better)
        boundary_fitness = state.boundary_integrity
        
        # Cognitive fitness (stable, efficient cognition = better)
        cognitive_health = core.triune.get_cognitive_health() if hasattr(core, 'triune') else {"stability": 0.5, "efficiency": 0.5}
        cognitive_fitness = (cognitive_health.get("stability", 0.5) + cognitive_health.get("efficiency", 0.5)) / 2
        
        # Activity fitness (moderate activity = better)
        activity = min(1.0, state.vpm_count / 200.0)  # Normalize VPM count
        
        # Overall fitness (weighted combination)
        fitness = (
            0.3 * energy_fitness +
            0.3 * boundary_fitness + 
            0.3 * cognitive_fitness +
            0.1 * activity
        )
        
        return {
            "fitness": fitness,
            "energy_fitness": energy_fitness,
            "boundary_fitness": boundary_fitness,
            "cognitive_fitness": cognitive_fitness,
            "activity_fitness": activity
        }
    
    def get_lineage_stats(self) -> Dict[str, Any]:
        """Get statistics about the reproduction lineage"""
        if not self.lineage:
            return {"total_offspring": 0, "max_generation": 0, "diversity": 0.0}
        
        generations = [g.generation for g in self.lineage]
        unique_mutations = set()
        for genetics in self.lineage:
            unique_mutations.update(genetics.mutation_history)
        
        return {
            "total_offspring": len(self.lineage),
            "max_generation": max(generations),
            "diversity": len(unique_mutations) / max(1, len(self.lineage)),
            "mutation_rate_actual": len(unique_mutations) / max(1, sum(len(g.mutation_history) for g in self.lineage))
        }
