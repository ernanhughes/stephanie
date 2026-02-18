from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class SupportDiagnostics:
    """
    Sentence-level support analysis for summarization-style tasks.
    Aggregated over all sentences in a summary.
    """

    sentence_count: int
    paragraph_count: int

    # Entailment-style support scores
    max_entailment: float
    mean_entailment: float
    min_entailment: float

    # Similarity-based fallback metrics
    mean_sim_top1: float
    min_sim_top1: float
    mean_sim_margin: float
    min_sim_margin: float

    # Coverage signals
    mean_coverage: float
    min_coverage: float

    # Energy aggregates
    max_energy: float
    mean_energy: float
    min_energy: float
    high_energy_count: int

    p90_energy: float
    frac_above_threshold: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "counts": {
                "sentence_count": self.sentence_count,
                "paragraph_count": self.paragraph_count,
            },
            "entailment": {
                "max": self.max_entailment,
                "mean": self.mean_entailment,
                "min": self.min_entailment,
            },
            "similarity": {
                "mean_sim_top1": self.mean_sim_top1,
                "min_sim_top1": self.min_sim_top1,
                "mean_sim_margin": self.mean_sim_margin,
                "min_sim_margin": self.min_sim_margin,
            },
            "coverage": {
                "mean": self.mean_coverage,
                "min": self.min_coverage,
            },
            "energy": {
                "max": self.max_energy,
                "min": self.min_energy,
                "high_count": self.high_energy_count,
                "mean": self.mean_energy,
                "p90": self.p90_energy,
                "frac_above_threshold": self.frac_above_threshold,
            }
        }
