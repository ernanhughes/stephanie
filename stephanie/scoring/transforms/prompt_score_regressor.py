# stephanie/scoring/transforms/prompt_score_regressor.py
import os

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class PromptScoreRegressor:
    def __init__(self, model_path="prompt_score_regressor.pkl"):
        self.model_path = model_path
        self.models = {}  # dimension -> trained model

    def train(self, embeddings, score_dicts):
        """
        embeddings: List[List[float]]
        score_dicts: List[Dict[str, float]] (one per embedding)
        """
        if not score_dicts:
            raise ValueError("No score dictionaries provided for training.")

        # Infer all dimensions from the first sample
        dimensions = list(score_dicts[0].keys())
        X = np.array(embeddings)

        for dim in dimensions:
            y = np.array([score[dim] for score in score_dicts])
            model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            self.models[dim] = model.fit(X, y)

        if self.model_path:
            joblib.dump(self.models, self.model_path)

    def load(self):
        if os.path.exists(self.model_path):
            self.models = joblib.load(self.model_path)
        else:
            raise FileNotFoundError(f"No model found at {self.model_path}")

    def predict(self, embedding):
        if not self.models:
            raise ValueError("Models not trained or loaded.")

        scores = {}
        for dim, model in self.models.items():
            scores[dim] = float(model.predict([embedding])[0])
        return scores

    def predict_batch(self, embeddings):
        """
        Returns: List[Dict[dimension -> score]]
        """
        if not self.models:
            raise ValueError("Models not trained or loaded.")

        results = []
        for emb in embeddings:
            results.append(self.predict(emb))
        return results
