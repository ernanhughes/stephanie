# stephanie/components/jitter/reproduction/quality_control.py
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

log = logging.getLogger("stephanie.jitter.reproduction.quality")

@dataclass
class GeneticMaterial:
    """Genetic blueprint for a Jitter organism with quality metrics"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    parent_id: Optional[str] = None
    generation: int = 0
    mutation_rate: float = 0.1
    quality_score: float = 0.0
    heritage: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=lambda: time.time())

class QualityAssessor:
    """Assesses quality of potential offspring"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.quality_threshold = config.get("quality_threshold", 0.6)
        self.heritage_preservation = config.get("heritage_preservation", True)
        self.lineage_diversity_weight = config.get("lineage_diversity_weight", 0.3)
        self.parent_fitness_weight = config.get("parent_fitness_weight", 0.4)
        self.mutation_quality_weight = config.get("mutation_quality_weight", 0.3)
        self.max_heritage_length = config.get("max_heritage_length", 5)
    
    def assess_offspring_quality(
        self,
        parent_core: Any,
        parent_genetics: GeneticMaterial,
        offspring_genetics: GeneticMaterial
    ) -> float:
        """
        Assess quality of potential offspring based on multiple factors.
        
        Returns a quality score between 0 and 1.
        """
        # 1. Lineage diversity score
        lineage_diversity = self._calculate_lineage_diversity(parent_genetics)
        
        # 2. Parent fitness inheritance
        parent_fitness = self._calculate_fitness(parent_core)["fitness"]
        
        # 3. Mutation quality (beneficial vs detrimental)
        mutation_quality = self._assess_mutation_quality(
            parent_genetics, 
            offspring_genetics
        )
        
        # 4. Heritage preservation (if enabled)
        heritage_score = 1.0
        if self.heritage_preservation:
            heritage_score = self._calculate_heritage_score(
                parent_genetics,
                offspring_genetics
            )
        
        # Weighted combination
        quality_score = (
            self.lineage_diversity_weight * lineage_diversity +
            self.parent_fitness_weight * parent_fitness +
            self.mutation_quality_weight * mutation_quality +
            0.1 * heritage_score  # Smaller weight for heritage
        )
        
        # Ensure score stays within bounds
        return max(0.0, min(1.0, quality_score))
    
    def _calculate_lineage_diversity(self, parent_genetics: GeneticMaterial) -> float:
        """Calculate diversity score based on lineage"""
        if not parent_genetics.heritage:
            return 0.5  # Neutral score for first generation
        
        # Simple diversity measure based on heritage length
        diversity = min(1.0, len(parent_genetics.heritage) / self.max_heritage_length)
        return diversity
    
    def _calculate_fitness(self, core: Any) -> Dict[str, float]:
        """Calculate fitness metrics from core system state"""
        health_metrics = core.homeostasis.get_health_metrics()
        
        return {
            "fitness": health_metrics["efficiency"],
            "stability": health_metrics["stability"],
            "resonance": health_metrics["resonance"]
        }
    
    def _assess_mutation_quality(
        self,
        parent_genetics: GeneticMaterial,
        offspring_genetics: GeneticMaterial
    ) -> float:
        """Assess whether mutations are beneficial or detrimental"""
        # Simple implementation - would be more sophisticated in production
        # For now, assume random mutation quality based on mutation rate
        base_quality = 1.0 - offspring_genetics.mutation_rate
        
        # Penalize if quality score decreased significantly
        if (parent_genetics.quality_score > 0.5 and 
            offspring_genetics.quality_score < parent_genetics.quality_score * 0.7):
            base_quality *= 0.5
        
        return base_quality
    
    def _calculate_heritage_score(
        self,
        parent_genetics: GeneticMaterial,
        offspring_genetics: GeneticMaterial
    ) -> float:
        """Calculate score based on heritage preservation"""
        # Check if heritage was properly inherited
        heritage_match = 0
        for i, heritage_id in enumerate(parent_genetics.heritage):
            if i < len(offspring_genetics.heritage) and offspring_genetics.heritage[i] == heritage_id:
                heritage_match += 1
        
        # Calculate match ratio
        match_ratio = heritage_match / max(1, len(parent_genetics.heritage))
        
        # Add current parent to heritage if not already present
        if parent_genetics.id not in offspring_genetics.heritage:
            match_ratio *= 0.8  # Small penalty for not inheriting parent
        
        return match_ratio
    
    def is_quality_acceptable(self, quality_score: float) -> bool:
        """Check if quality score meets threshold"""
        return quality_score >= self.quality_threshold

class QualityControlledReproduction:
    """Reproduction system with quality control mechanisms"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.quality_assessor = QualityAssessor(config)
        self.max_candidates = config.get("max_candidates", 3)
        self.logger = logging.getLogger("stephanie.jitter.reproduction.quality_control")
    
    def reproduce_with_quality_control(
        self,
        core: Any,
        parent_genetics: GeneticMaterial
    ) -> Optional[GeneticMaterial]:
        """
        Reproduction with quality control - generates multiple candidates and selects the best.
        
        Returns the best candidate that meets quality threshold, or None if none qualify.
        """
        # Check basic reproduction conditions first
        if not self._can_reproduce(core):
            return None
        
        # Generate multiple candidates
        candidates = []
        for _ in range(self.max_candidates):
            # Create mutated offspring
            offspring = self._create_offspring(parent_genetics)
            
            # Assess quality
            quality = self.quality_assessor.assess_offspring_quality(
                core,
                parent_genetics,
                offspring
            )
            
            offspring.quality_score = quality
            candidates.append((offspring, quality))
        
        # Select best candidate
        best_candidate, best_quality = max(candidates, key=lambda x: x[1])
        
        # Check quality threshold
        if self.quality_assessor.is_quality_acceptable(best_quality):
            self.logger.info(f"Reproduction successful - quality: {best_quality:.3f}")
            return best_candidate
        else:
            self.logger.info(f"Reproduction quality too low: {best_quality:.3f} < {self.quality_assessor.quality_threshold}")
            return None
    
    def _can_reproduce(self, core: Any) -> bool:
        """Check basic reproduction conditions"""
        # Check energy levels
        metabolic_energy = core.energy.level("metabolic")
        cognitive_energy = core.energy.level("cognitive")
        
        energy_threshold = self.config.get("reproduction_energy_threshold", 80.0)
        return (metabolic_energy > energy_threshold and 
                cognitive_energy > energy_threshold * 0.7)
    
    def _create_offspring(self, parent_genetics: GeneticMaterial) -> GeneticMaterial:
        """Create offspring with mutation"""
        mutation_rate = parent_genetics.mutation_rate * (0.8 + 0.4 * np.random.random())
        
        # Create new genetics
        offspring = GeneticMaterial(
            parent_id=parent_genetics.id,
            generation=parent_genetics.generation + 1,
            mutation_rate=mutation_rate
        )
        
        # Inherit heritage
        offspring.heritage = parent_genetics.heritage.copy()
        if len(offspring.heritage) < self.quality_assessor.max_heritage_length:
            offspring.heritage.append(parent_genetics.id)
        
        return offspring