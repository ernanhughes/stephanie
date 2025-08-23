# stephanie/registry/scorer.py
from stephanie.scoring.ebt_scorer import EBTScorer
from stephanie.scoring.llm_scorer import LLMScorer
from stephanie.scoring.mrq_scorer import MRQScorer
from stephanie.scoring.proximity_scorer import ProximityScorer
from stephanie.scoring.sicql_scorer import SICQLScorer
from stephanie.scoring.svm_scorer import SVMScorer

SCORER_REGISTRY = {
    "mrq": MRQScorer,
    "llm": LLMScorer,
    "svm": SVMScorer,
    "sicql": SICQLScorer,
    "ebt": EBTScorer,
    "proximity": ProximityScorer
}


def get_scorer(scorer_type: str, cfg: dict, memory=None, logger=None):
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

    return SCORER_REGISTRY[scorer_type](cfg, memory, logger)
