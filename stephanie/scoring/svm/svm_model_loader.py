import os
from pathlib import Path
from typing import Dict, Tuple

from joblib import load
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from stephanie.scoring.transforms.regression_tuner import RegressionTuner
from stephanie.utils.file_utils import load_json
from stephanie.utils.model_utils import (discover_saved_dimensions,
                                         get_svm_file_paths)


class SVMModelLoader:
    def __init__(self, cfg, logger=None):
        self.cfg = cfg
        self.logger = logger
        self.model_path = cfg.get("model_path", "models")
        self.model_type = cfg.get("model_type", "svm")
        self.target_type = cfg.get("target_type", "document")
        self.model_version = cfg.get("model_version", "v1")
        self.default_model_name = cfg.get("default_model", "default")
        self.auto_train = cfg.get("auto_train_if_missing", False)
        self.fallback_to_default = cfg.get("fallback_to_default", True)


    def load_all(self, dimensions: list[str] = None) -> Dict[str, Tuple[StandardScaler, SVR]]:
        """
        Load all SVM models and scalers for the given dimensions.
        """
        if not dimensions:
            dimensions = discover_saved_dimensions(
                model_type=self.model_type,
                target_type=self.target_type,
            )

        models = {}
        for dim in dimensions:
            scaler, model = self.load_dimension(dim)
            models[dim] = (scaler, model)

        return models

    def load_dimension(self, dim: str) -> Tuple[StandardScaler, SVR]:
        paths = get_svm_file_paths(
            self.model_path,
            self.model_type,
            self.target_type,
            dim,
            self.model_version,
        )

        if self.logger:
            self.logger.log("LoadingSVMModel", {"dimension": dim, "model": paths["model"]})

        scaler = load(paths["scaler"])
        model = load(paths["model"])
        return scaler, model

    def load_tuner(self, dim: str) -> RegressionTuner:
        paths = get_svm_file_paths(
            self.model_path,
            self.model_type,
            self.target_type,
            dim,
            self.model_version,
        )
        tuner = RegressionTuner(dimension=dim, logger=self.logger)
        tuner.load(paths["tuner"])
        return tuner

    def load_meta(self, dim: str) -> dict:
        paths = get_svm_file_paths(
            self.model_path,
            self.model_type,
            self.target_type,
            dim,
            self.model_version,
        )
        return (
            load_json(paths["meta"])
            if os.path.exists(paths["meta"])
            else {"min_score": 0, "max_score": 100}
        )


    def load_dimension(self, dimension: str):
        """
        Try to load a model for the given dimension.
        Falls back to default or auto-trains if enabled.
        """
        model_path = self._get_model_paths(dimension)

        if self._file_exists(model_path["scaler"]) and self._file_exists(model_path["model"]):
            self.logger.log("ModelLoaded", {"dimension": dimension, "path": str(model_path["scaler"])})
            return load(model_path["scaler"]), load(model_path["model"])

        self.logger.log("ModelNotFound", {"dimension": dimension, "path": str(model_path["scaler"])})

        if self.fallback_to_default:
            default_path = self._get_model_paths(self.default_model_name)
            if self._file_exists(default_path["scaler"]) and self._file_exists(default_path["model"]):
                self.logger.log("FallbackToDefaultModel", {"dimension": dimension})
                return load(default_path["scaler"]), load(default_path["model"])

        if self.auto_train:
            self.logger.log("AutoTrainingModel", {"dimension": dimension})
            return self._train_and_save(dimension)

        self.logger.log("NoModelAvailable", {"dimension": dimension})
        return None, None

    def _get_model_paths(self, dimension: str):
        version = self.cfg.get("model_version", "v1")
        dim_path = self.model_base_path / dimension / version
        return {
            "scaler": dim_path / f"{dimension}_scaler.joblib",
            "model": dim_path / f"{dimension}_model.joblib"
        }

    def _file_exists(self, path: Path):
        return path.exists()

    def _train_and_save(self, dimension: str):
        """
        Placeholder for auto-training logic.
        This would load training data, train an SVM model, and save it.
        """
        self.logger.log("TrainingModelForDimension", {"dimension": dimension})
        # Here you'd implement or call the actual training logic
        # scaler, model = train_model(dimension)
        # save(scaler, model_path)
        # For now, return dummy objects
        return "fallback_scaler", "fallback_model"