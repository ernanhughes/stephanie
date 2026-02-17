from dataclasses import dataclass, field
from typing import Dict, List
from enum import Enum
import time
import uuid

class RewardAxis(str, Enum):
    """Explicit semantic axes - no ambiguous scalars"""
    HRM_ALIGNMENT = "hrm_alignment"          # Higher = better (truthfulness)
    HALLUCINATION_ENERGY = "hallucination_energy"  # Lower = better (Certum)
    EMBEDDING_MARGIN = "embedding_margin"    # Higher = better (Embed-RL)
    POLICY_ADVANTAGE = "policy_advantage"    # Higher = better (task progress)
    METRIC_ALIGNMENT = "metric_alignment"    # Higher = better (goal fidelity)
    COHERENCE = "coherence"                  # Higher = better (narrative flow)
    CONTEXT_FIDELITY = "context_fidelity"    # Higher = better (VPM stability)

@dataclass(frozen=True)
class RewardVector:
    """
    Multi-dimensional reward primitive for trace-native self-improvement.
    Immutable. Always tied to a trace. Always interpretable.
    """
    # Core axes (normalized to [-1.0, 1.0] where semantics allow)
    values: Dict[RewardAxis, float] = field(default_factory=dict)
    
    # Trace provenance (critical for MemCube + forensic auditing)
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    source_model: str = "unknown"  # e.g., "HRM-v3", "Certum-Energy", "Embed-RL"
    
    # Diagnostic metadata (for reflection engine)
    confidence: float = 1.0  # Est. reliability of this vector (0.0-1.0)
    failure_signatures: List[str] = field(default_factory=list)  # e.g., ["energy_spike", "margin_collapse"]
    
    def __post_init__(self):
        # Enforce normalization boundaries where defined
        for axis, val in self.values.items():
            if axis == RewardAxis.HALLUCINATION_ENERGY:
                assert -1.0 <= val <= 1.0, f"Energy must be normalized: {val}"
            elif axis in [RewardAxis.HRM_ALIGNMENT, RewardAxis.EMBEDDING_MARGIN]:
                assert 0.0 <= val <= 1.0, f"{axis} must be [0,1]: {val}"
    
    def delta(self, other: "RewardVector") -> "RewardVector":
        """Compute improvement vector (self - other). Preserves provenance of *this* vector."""
        delta_vals = {
            axis: self.values.get(axis, 0.0) - other.values.get(axis, 0.0)
            for axis in set(self.values) | set(other.values)
        }
        return RewardVector(
            values=delta_vals,
            trace_id=f"delta_{self.trace_id}_{other.trace_id}",
            source_model="reward_delta",
            confidence=min(self.confidence, other.confidence),
            failure_signatures=[]  # Delta has no failures
        )
    
    def dominates(self, other: "RewardVector", critical_axes: List[RewardAxis]) -> bool:
        """
        True iff self improves on *all* critical_axes vs other.
        Prevents reward hacking (e.g., boosting HRM while collapsing embedding margin).
        """
        for axis in critical_axes:
            self_val = self.values.get(axis, -float('inf'))
            other_val = other.values.get(axis, -float('inf'))
            
            # Special handling: lower energy = better
            if axis == RewardAxis.HALLUCINATION_ENERGY:
                if self_val > other_val:  # Higher energy = worse
                    return False
            else:
                if self_val <= other_val:  # Must strictly improve
                    return False
        return True
    
    def to_memcube_payload(self) -> Dict:
        """Structured export for MemCube storage"""
        return {
            "reward_vector": {k.value: v for k, v in self.values.items()},
            "trace_id": self.trace_id,
            "failure_signatures": self.failure_signatures,
            "critical_axes": [ax.value for ax in self._infer_critical_axes()],
            "timestamp": self.timestamp
        }
    
    def _infer_critical_axes(self) -> List[RewardAxis]:
        """Heuristic: axes where value is below threshold OR has failure signature"""
        critical = []
        thresholds = {
            RewardAxis.HRM_ALIGNMENT: 0.6,
            RewardAxis.HALLUCINATION_ENERGY: 0.3,  # Lower = better, so high value = bad
            RewardAxis.EMBEDDING_MARGIN: 0.4
        }
        for axis, thresh in thresholds.items():
            val = self.values.get(axis, 0.0)
            if (axis == RewardAxis.HALLUCINATION_ENERGY and val > thresh) or \
               (axis != RewardAxis.HALLUCINATION_ENERGY and val < thresh):
                critical.append(axis)
        return critical + [RewardAxis(sig) for sig in self.failure_signatures if sig in RewardAxis.__members__]