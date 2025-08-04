import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from stephanie.scoring.training.base_trainer import BaseTrainer
from stephanie.scoring.transforms.regression_tuner import RegressionTuner


class SVMTrainer(BaseTrainer):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.kernel = cfg.get("kernel", "rbf")
        self.C = cfg.get("C", 1.0)
        self.epsilon = cfg.get("epsilon", 0.1)


    def train(self, samples, dimension):
        samples = [
            {
                "title": pair["title"],
                "output": pair["output_a"],
                "score": pair["value_a"],
            }
            for pair in samples
        ]

        dataloader = self._create_dataloader(samples)
        if not dataloader:
            return {"error": "insufficient_data", "dimension": dimension}

        # Convert DataLoader to numpy arrays
        X, y = [], []
        for ctx_emb, doc_emb, scores in dataloader:
            ctx_emb = ctx_emb.cpu().numpy()
            doc_emb = doc_emb.cpu().numpy()
            x = np.concatenate([ctx_emb, doc_emb], axis=1)
            X.append(x)
            y.append(scores.cpu().numpy())

        X = np.concatenate(X, axis=0)
        y = np.concatenate(y, axis=0)

        # Fit scaler and scale features 
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train SVM
        model = SVR(kernel=self.kernel, C=self.C, epsilon=self.epsilon)
        model.fit(X_scaled, y)

        # Save both model and scaler
        locator = self.get_locator(dimension)
        model_path = locator.model_file(suffix=".joblib")
        scaler_path = locator.scaler_file()

        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)

        meta = {
            "dimension": dimension,
            "model_type": "svm",
            "target_type": self.target_type,
            "embedding_type": self.embedding_type,
            "version": self.version,
            "kernel": self.kernel,
            "C": self.C,
            "epsilon": self.epsilon,
            "dim": self.dim,
            "hdim": self.hdim,
            "min_value": self.cfg.get("min_value", 0),
            "max_value": self.cfg.get("max_value", 100),
        }
        self._save_meta_file(meta, dimension)

        # Add before return meta
        tuner = RegressionTuner(dimension=dimension, logger=self.logger)
        for i in range(len(X)):
            prediction = float(model.predict(X_scaled[i].reshape(1, -1))[0])
            actual = float(y[i])
            tuner.train_single(prediction, actual)
        tuner.save(locator.tuner_file())

        self.log_event("SVMTrainingComplete", {"dimension": dimension})

        return meta
