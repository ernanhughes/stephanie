from typing import Dict, List
from enum import Enum
from stephanie.data.score_bundle import ScoreBundle
from stephanie.components.elm.core.reward_vector import RewardAxis
from stephanie.data.score_result import ScoreResult

class AxisDirection(str, Enum):
    HIGHER_IS_BETTER = "higher"
    LOWER_IS_BETTER = "lower"

# Map ScoreBundle dimensions to governance axes
DIMENSION_TO_AXIS: Dict[str, RewardAxis] = {
    # HRM dimensions
    "alignment": RewardAxis.HRM_ALIGNMENT,
    "reasoning_quality": RewardAxis.HRM_ALIGNMENT,
    
    # EBT energy dimensions (lower is better)
    "energy": RewardAxis.HALLUCINATION_ENERGY,
    "speculation": RewardAxis.HALLUCINATION_ENERGY,
    
    # Embedding dimensions
    "margin": RewardAxis.EMBEDDING_MARGIN,
    "similarity": RewardAxis.EMBEDDING_MARGIN,
    
    # Policy dimensions
    "advantage": RewardAxis.POLICY_ADVANTAGE,
    "context_grounding": RewardAxis.CONTEXT_FIDELITY,
}

AXIS_SEMANTICS: Dict[RewardAxis, AxisDirection] = {
    RewardAxis.HRM_ALIGNMENT: AxisDirection.HIGHER_IS_BETTER,
    RewardAxis.HALLUCINATION_ENERGY: AxisDirection.LOWER_IS_BETTER,
    RewardAxis.EMBEDDING_MARGIN: AxisDirection.HIGHER_IS_BETTER,
    RewardAxis.POLICY_ADVANTAGE: AxisDirection.HIGHER_IS_BETTER,
    RewardAxis.METRIC_ALIGNMENT: AxisDirection.HIGHER_IS_BETTER,
    RewardAxis.COHERENCE: AxisDirection.HIGHER_IS_BETTER,
    RewardAxis.CONTEXT_FIDELITY: AxisDirection.HIGHER_IS_BETTER,
}

class GovernanceSignalExtractor:
    """
    Extract governance signals from ScoreBundle.
    
    Converts your dynamic ScoreBundle into structured governance metrics.
    """
    
    def __init__(self, critical_dimensions: List[str] = None):
        self.critical_dimensions = critical_dimensions or [
            "alignment", "energy", "margin"
        ]
    
    def extract_from_bundle(self, bundle: "ScoreBundle") -> Dict[str, float]:
        """
        Extract governance metrics from ScoreBundle.
        
        Returns:
            Dict with keys: energy, hrm_alignment, embedding_margin, etc.
        """
        metrics = {}
        
        for dim_name, result in bundle.results.items():
            # Map dimension to governance axis
            axis = DIMENSION_TO_AXIS.get(dim_name)
            if not axis:
                continue  # Skip non-governance dimensions
            
            # Extract score (normalized to 0-1)
            score = self._normalize_score(result.score, dim_name)
            
            # Store by axis name
            axis_key = axis.value
            metrics[axis_key] = score
            
            # Extract energy specifically (from attributes)
            if axis == RewardAxis.HALLUCINATION_ENERGY:
                energy = self._extract_energy(result)
                metrics["energy_raw"] = energy
        
        return metrics
    
    def _normalize_score(self, score: float, dimension: str) -> float:
        """
        Normalize score to [0, 1] based on dimension semantics.
        
        Your scores are 0-100, so divide by 100.
        For energy (lower=better), invert.
        """
        # Your scores are 0-100
        normalized = score / 100.0
        
        # Clamp to [0, 1]
        normalized = max(0.0, min(1.0, normalized))
        
        # Invert if lower-is-better dimension
        if dimension in ["energy", "speculation"]:
            normalized = 1.0 - normalized
        
        return normalized
    
    def _extract_energy(self, result: "ScoreResult") -> float:
        """
        Extract raw energy from attributes.
        
        Your EBT stores raw_energy in attributes.
        """
        if hasattr(result, "attributes") and result.attributes:
            raw_energy = result.attributes.get("raw_energy")
            if raw_energy is not None:
                return float(raw_energy)
        
        # Fallback: use score as proxy
        return 100.0 - result.score  # Higher score = lower energy
    
    def compute_dominance(
        self,
        bundle_before: "ScoreBundle",
        bundle_after: "ScoreBundle"
    ) -> bool:
        """
        Check if bundle_after dominates bundle_before on critical dimensions.
        
        Uses ScoreBundle.diff() for precise comparison.
        """
        diff = bundle_after.diff(bundle_before)
        
        for dim in self.critical_dimensions:
            if dim not in diff.get("dimensions", {}):
                continue
            
            dim_diff = diff["dimensions"][dim]
            delta = dim_diff.get("score_delta", 0)
            
            # Check direction semantics
            axis = DIMENSION_TO_AXIS.get(dim)
            if not axis:
                continue
            
            direction = AXIS_SEMANTICS[axis]
            
            if direction == AxisDirection.HIGHER_IS_BETTER:
                if delta <= 0:
                    return False  # Must improve
            else:  # LOWER_IS_BETTER
                if delta >= 0:
                    return False  # Must decrease
        
        return True  # All critical dimensions improved
    
    def compute_delta_vector(
        self,
        bundle_before: "ScoreBundle",
        bundle_after: "ScoreBundle"
    ) -> Dict[str, float]:
        """
        Compute direction-normalized delta vector.
        
        Positive = improvement, regardless of axis direction.
        """
        diff = bundle_after.diff(bundle_before)
        delta_vector = {}
        
        for dim, dim_diff in diff.get("dimensions", {}).items():
            axis = DIMENSION_TO_AXIS.get(dim)
            if not axis:
                continue
            
            delta = dim_diff.get("score_delta", 0)
            direction = AXIS_SEMANTICS[axis]
            
            # Normalize delta to [-1, 1]
            normalized_delta = delta / 100.0
            
            # Direction-aware: positive always = improvement
            if direction == AxisDirection.LOWER_IS_BETTER:
                normalized_delta = -normalized_delta
            
            delta_vector[axis.value] = normalized_delta
        
        return delta_vector