# stephanie/agents/pipeline_preference_trainer.py

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from stephanie.models.comparison_preference import ComparisonPreferenceORM
from stephanie.models.goal import GoalORM


class PipelinePreferenceTrainerAgent:
    def __init__(self, session, logger=None):
        self.session = session
        self.logger = logger
        self.model = None
        self.scaler = None

    def train(self):
        prefs = self.session.query(ComparisonPreferenceORM).all()
        if not prefs:
            print("No preference data found.")
            return

        X, y = [], []

        for pref in prefs:
            goal = self.session.query(GoalORM).get(pref.goal_id)
            if not goal:
                continue

            # Extract features from preferred and rejected runs
            feat_pref = self._extract_features(
                goal, pref.preferred_tag, pref.dimension_scores.get(pref.preferred_tag)
            )
            feat_rej = self._extract_features(
                goal, pref.rejected_tag, pref.dimension_scores.get(pref.rejected_tag)
            )
            if feat_pref is None or feat_rej is None:
                continue

            # Create difference vector (preferred - rejected)
            X.append(np.array(feat_pref) - np.array(feat_rej))
            y.append(1)  # preferred > rejected

        if not X:
            print("No valid feature pairs.")
            return

        X = np.array(X)
        self.scaler = StandardScaler().fit(X)
        X_scaled = self.scaler.transform(X)

        self.model = LogisticRegression()
        self.model.fit(X_scaled, y)

        joblib.dump((self.model, self.scaler), "models/pipeline_preference_model.pkl")
        print("✅ Trained and saved preference model.")

    def _extract_features(self, goal, tag, scores):
        if not scores:
            return None

        features = []

        # Add basic features — e.g. per-dimension score
        for dim in sorted(scores.keys()):
            if dim == "overall":
                continue
            features.append(scores[dim])

        # Add tag as one-hot (TODO: use embedding or smarter encoding)
        tag_feature = hash(tag) % 1000 / 1000.0
        features.append(tag_feature)

        # Add goal length (example feature)
        features.append(len(goal.description or ""))

        return features
