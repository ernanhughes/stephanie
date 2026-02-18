from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class GovernanceSignal:
    """
    Policy-level representation of AI output quality.

    Independent of Stephanie or Certum.
    """

    energy: float
    energy_p90: float
    high_energy_count: int
    entailment_mean: Optional[float] = None
    entailment_min: Optional[float] = None
    sim_margin: Optional[float] = None
    coverage: Optional[float] = None
    sentence_count: int
    embedding_margin: Optional[float] = None
    alignment: Optional[float] = None
    metadata: Optional[Dict] = None

    @property
    def instability(self) -> float:
        """Return raw instability proxy (1 - energy)."""
        return 1.0 - self.energy


def from_support_diagnostics(sd) -> GovernanceSignal:
    return GovernanceSignal(
        energy=sd.mean_energy,
        energy_p90=sd.p90_energy,
        high_energy_count=sd.high_energy_count,
        entailment_mean=sd.mean_entailment,
        entailment_min=sd.min_entailment,
        sim_margin=sd.mean_sim_margin,
        coverage=sd.mean_coverage,
        sentence_count=sd.sentence_count,
    )
