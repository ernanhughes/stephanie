# stephanie/agents/knowledge/document_svm_trainer.py
import os
import torch
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.transforms.regression_tuner import RegressionTuner
from stephanie.utils.file_utils import save_json
from stephanie.utils.model_utils import get_model_path
from stephanie.scoring.document_pair_builder import DocumentPreferencePairBuilder


class DocumentSVMTrainerAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.model_path = cfg.get("model_path", "models")
        self.model_type = cfg.get("model_type", "svm")
        self.target_type = cfg.get("target_type", "document")
        self.model_version = cfg.get("model_version", "v1")

        self.models = {}  # dim -> (scaler, model)
        self.regression_tuners = {}

        self.dimensions = cfg.get("dimensions", [])
        # Initialize tuners and models
        for dim in self.dimensions:
            self._initialize_dimension(dim)

    def _initialize_dimension(self, dim):
        """Initialize SVM model, scaler, and tuner for each dimension"""
        self.models[dim] = (StandardScaler(), SVR(kernel='linear'))
        self.regression_tuners[dim] = RegressionTuner(dimension=dim, logger=self.logger)

    async def run(self, context: dict) -> dict:
        goal_text = context.get("goal", {}).get("goal_text")

        builder = DocumentPreferencePairBuilder(
            db=self.memory.session,
            logger=self.logger
        )
        training_pairs = builder.get_training_pairs_by_dimension(goal=goal_text)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for dim, pairs in training_pairs.items():
            self._initialize_dimension(dim)
            if not pairs:
                continue

            self.logger.log("SVMTrainingStart", {"dimension": dim, "num_pairs": len(pairs)})

            X, y = [], []

            # Build dataset
            for pair in pairs:
                title = pair["title"]
                for side in ["a", "b"]:
                    output = pair[f"output_{side}"]
                    score = pair[f"value_{side}"]

                    ctx_emb = self.memory.embedding.get_or_create(title)
                    doc_emb = self.memory.embedding.get_or_create(output)

                    # Combine embeddings as feature vector
                    feature = np.array(ctx_emb + doc_emb)
                    X.append(feature)
                    y.append(score)

            if len(X) < 2:
                self.logger.log("SVMNotEnoughData", {"dimension": dim})
                continue

            X = np.array(X)
            y = np.array(y)

            # Normalize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Train model
            model = SVR(kernel="linear")
            model.fit(X_scaled, y)

            # Save model
            self.models[dim] = (scaler, model)

            # Save model files
            model_path = get_model_path(
                self.model_path,
                self.model_type,
                self.target_type,
                dim,
                self.model_version,
            )
            os.makedirs(model_path, exist_ok=True)

            # Save model state (we'll serialize separately)
            predicttor_path = f"{model_path}/{dim}.pt"
            meta_path = f"{model_path}/{dim}.meta.json"
            tuner_path = f"{model_path}/{dim}.tuner.json"

            # Since we're using scikit-learn, we'll use joblib or custom serialization
            from joblib import dump
            scaler_path = f"{model_path}/{dim}_scaler.joblib"
            model_path_joblib = f"{model_path}/{dim}.joblib"
            dump(scaler, scaler_path)
            dump(model, model_path_joblib)

            # Save normalization meta
            meta = {
                "min_score": float(np.min(y)),
                "max_score": float(np.max(y))
            }
            save_json(meta, meta_path)

            # Train regression tuner using same data
            tuner = self.regression_tuners[dim]
            for i in range(len(X)):
                tuner.train_single(model.predict(X_scaled[i].reshape(1, -1))[0], y[i])

            tuner.save(tuner_path)

            self.logger.log("SVMModelSaved", {"dimension": dim, "path": model_path})

        context[self.output_key] = training_pairs
        return context