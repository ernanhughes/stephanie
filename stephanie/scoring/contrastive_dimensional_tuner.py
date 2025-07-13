# stephanie/scoring/contrastive_dimensional_tuner.py
import numpy as np
from sklearn.linear_model import LogisticRegression


class ContrastiveDimensionalTuner:
    """
    Learns weights for each scoring dimension using contrastive learning.
    Given pairs of scored examples (A vs B) and a preference, it learns which dimensions matter most.
    """

    def __init__(self, dimensions, logger=None):
        """
        Args:
            dimensions (list of str): List of dimension names (e.g., ["correctness", "clarity"]).
            logger (optional): Optional logger to record training events.
        """
        self.dimensions = dimensions
        self.logger = logger
        self.X = []  # Feature differences (vector of deltas across dimensions)
        self.y = []  # Labels: 1 if A preferred over B, 0 otherwise
        self.model = None

    def add_training_pair(self, scores_a: dict, scores_b: dict, preferred: str):
        """
        Adds a training example.

        Args:
            scores_a (dict): Scores for option A, keyed by dimension.
            scores_b (dict): Scores for option B, keyed by dimension.
            preferred (str): "A" or "B", indicating which output was preferred.
        """
        delta = np.array([scores_a[dim] - scores_b[dim] for dim in self.dimensions])

        # If B is preferred, invert the delta
        if preferred.upper() == "B":
            delta = -delta
            label = 1  # B preferred (inverted delta)
        else:
            label = 1  # A preferred (original delta)

        self.X.append(delta)
        self.y.append(label)

        if self.logger:
            self.logger.log(
                "ContrastiveTrainingPairAdded",
                {"delta": delta.tolist(), "preferred": preferred},
            )

    def train(self):
        """
        Trains a logistic regression model using the current contrastive data.
        """
        if len(self.X) < 3:
            if self.logger:
                self.logger.log(
                    "ContrastiveTrainingSkipped",
                    {"reason": "Not enough data", "num_examples": len(self.X)},
                )
            return

        X_array = np.array(self.X)
        y_array = np.array(self.y)

        self.model = LogisticRegression()
        self.model.fit(X_array, y_array)

        if self.logger:
            self.logger.log(
                "ContrastiveModelTrained", {"coefficients": self.get_weights()}
            )

    def get_weights(self) -> dict:
        """
        Returns the learned dimension weights (if trained).

        Returns:
            dict: Mapping from dimension to learned weight.
        """
        if self.model is None:
            return {dim: 1.0 for dim in self.dimensions}  # fallback: equal weights

        weights = self.model.coef_[0]
        return {dim: round(float(w), 4) for dim, w in zip(self.dimensions, weights)}

    def score(self, dimension_scores: dict) -> float:
        """
        Calculates a single weighted score from per-dimension scores.

        Args:
            dimension_scores (dict): Scores keyed by dimension.

        Returns:
            float: Weighted total score.
        """
        weights = self.get_weights()
        total = sum(
            dimension_scores[dim] * weights.get(dim, 1.0) for dim in self.dimensions
        )
        return round(total, 4)
