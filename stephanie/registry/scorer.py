# stephanie/registry/scorer.py
from __future__ import annotations

from stephanie.scoring.scorer.ebt_scorer import EBTScorer
from stephanie.scoring.scorer.llm_scorer import LLMScorer
from stephanie.scoring.scorer.mrq_scorer import MRQScorer
from stephanie.scoring.scorer.proximity_scorer import ProximityScorer
from stephanie.scoring.scorer.sicql_scorer import SICQLScorer
from stephanie.scoring.scorer.svm_scorer import SVMScorer

SCORER_REGISTRY = {
    "mrq": MRQScorer,
    "llm": LLMScorer,
    "svm": SVMScorer,
    "sicql": SICQLScorer,
    "ebt": EBTScorer,
    "proximity": ProximityScorer
}


def get_scorer(scorer_type: str, cfg: dict, memory, container, logger):
    """
    Factory function to get a scorer instance by type.

    :param scorer_type: Type of the scorer (e.g., 'mrq', 'llm', 'svm').
    :param cfg: Configuration dictionary for the scorer.
    :param memory: Optional memory object for the scorer.
    :param logger: Optional logger object for the scorer.
    :return: An instance of the requested scorer type.
    """
    if scorer_type not in SCORER_REGISTRY:
        raise ValueError(f"Unknown scorer type: {scorer_type}")

    return SCORER_REGISTRY[scorer_type](cfg, memory, container, logger)
