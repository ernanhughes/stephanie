# stephanie/scoring/base_scorer.py
from abc import ABC, abstractmethod

import numpy as np

from stephanie.scoring.scorable import Scorable


class BaseScorer(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique name or tag for the scorer (e.g. 'svm', 'mrq', 'llm-feedback')"""
        pass

    """ 
    Base interface for any scorer that evaluates a hypothesis given a goal and dimensions.

    Returns:
        A dictionary with dimension names as keys, and for each:
            - score (float)
            - rationale (str)
            - weight (float, optional)
    """

    @abstractmethod
    def score(self, goal: dict, scorable: Scorable, dimensions: list[str]) -> dict:
        raise NotImplementedError("Subclasses must implement the score method.")

    def _build_feature_vector(self, goal: dict, scorable: Scorable):
        emb_goal = self.memory.embedding.get_or_create(goal["goal_text"])
        emb_hyp = self.memory.embedding.get_or_create(scorable.text)

        # Optional: make sure they're both numpy arrays
        emb_goal = np.array(emb_goal)
        emb_hyp = np.array(emb_hyp)

        vec = np.concatenate([emb_goal, emb_hyp])

        return vec
