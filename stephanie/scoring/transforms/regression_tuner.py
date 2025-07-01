# stephanie/scoring/transforms/regression_tuner.py

import json

import numpy as np
from sklearn.linear_model import LinearRegression


class RegressionTuner:
    """
    Learns to transform MR.Q scores to align with LLM scores dynamically.
    Does not save any state to disk—purely in-memory and real-time.
    """

    def __init__(self, dimension: str, logger=None, min_samples: int = 10):
        self.dimension = dimension
        self.logger = logger
        self.min_samples = min_samples
        self.x = []  # MRQ scores
        self.y = []  # LLM scores
        self.model = None

    def train_single(self, mrq_score: float, llm_score: float):
        """Adds a new training pair and refits if threshold reached."""
        self.x.append(mrq_score)
        self.y.append(llm_score)

        if len(self.x) >= self.min_samples:
            self._fit()

        if self.logger:
            self.logger.log("RegressionTunerTrainSingle", {
                "dimension": self.dimension,
                "mrq_score": mrq_score,
                "llm_score": llm_score,
                "total_samples": len(self.x)
            })

    def _fit(self):
        """Fits a linear regression model to current examples."""
        x_arr = np.array(self.x).reshape(-1, 1)
        y_arr = np.array(self.y)

        self.model = LinearRegression().fit(x_arr, y_arr)

        if self.logger:
            self.logger.log("RegressionTunerFitted", {
                "dimension": self.dimension,
                "count": len(self.x),
                "coef": float(self.model.coef_[0]),
                "intercept": float(self.model.intercept_),
            })

    def transform(self, score: float) -> float:
        """Transforms a score using the fitted regression model if available."""
        if self.model:
            return float(self.model.predict(np.array([[score]]))[0])
        return score

    def save(self, path):
        if not self.model:
            raise ValueError("Model has not been trained yet — nothing to save.")
        
        data = {
            "dimension": self.dimension,
            "samples": list(zip(self.x, self.y)),
            "coef": float(self.model.coef_[0]),
            "intercept": float(self.model.intercept_),
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    
    def load(self, path):
        with open(path, "r") as f:
            data = json.load(f)

        self.dimension = data["dimension"]
        self.x, self.y = zip(*data["samples"]) if data["samples"] else ([], [])
        self.x = list(self.x)
        self.y = list(self.y)

        if self.x and len(self.x) >= self.min_samples:
            self._fit()
