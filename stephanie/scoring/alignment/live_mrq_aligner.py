# stephanie/scoring/alignment/live_mrq_aligner.py

import logging
from typing import List, Optional

from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)

class LiveMRQAligner:
    def __init__(self):
        self.mrq_scores: List[float] = []
        self.llm_scores: List[float] = []
        self.model: Optional[LinearRegression] = None
        self.min_points = 10  # Minimum data before tuning

    def add_score_pair(self, mrq_score: float, llm_score: float):
        self.mrq_scores.append(mrq_score)
        self.llm_scores.append(llm_score)
        if len(self.mrq_scores) >= self.min_points:
            self._fit()

    def _fit(self):
        X = [[s] for s in self.mrq_scores]
        y = self.llm_scores
        self.model = LinearRegression().fit(X, y)
        logger.info(
            f"[LiveMRQAligner] Updated transformation model "
            f"with {len(self.mrq_scores)} examples. "
            f"Slope: {self.model.coef_[0]:.3f}, Intercept: {self.model.intercept_:.3f}"
        )

    def transform(self, mrq_score: float) -> float:
        if self.model:
            return float(self.model.predict([[mrq_score]])[0])
        return mrq_score  # Return raw score if model not ready

    def clear(self):
        self.mrq_scores.clear()
        self.llm_scores.clear()
        self.model = None
