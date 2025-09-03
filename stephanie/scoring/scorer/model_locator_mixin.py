# stephanie/scoring/model_locator_mixin.py

import os


class ModelLocatorMixin:
    class Locator:
        def __init__(
            self,
            root_dir: str,
            model_type: str,
            target_type: str,
            dimension: str,
            version: str,
            embedding_type: str,
        ):
            self.root_dir = root_dir
            self.model_type = model_type
            self.target_type = target_type
            self.dimension = dimension
            self.version = version
            self.embedding_type = embedding_type

        @property
        def base_path(self) -> str:
            path = os.path.join(
                self.root_dir,
                self.embedding_type,
                self.model_type,
                self.target_type,
                self.dimension,
                self.version,
            )
            os.makedirs(path, exist_ok=True)
            return path

        # Model-specific paths
        def model_file(self, suffix: str = ".pt") -> str:
            return os.path.join(self.base_path, f"{self.dimension}{suffix}")

        def encoder_file(self) -> str:
            return os.path.join(self.base_path, f"{self.dimension}_encoder.pt")

        def get_q_head_path(self) -> str:
            return os.path.join(self.base_path, f"{self.dimension}_q.pt")

        def get_v_head_path(self) -> str:
            return os.path.join(self.base_path, f"{self.dimension}_v.pt")

        def get_pi_head_path(self) -> str:
            return os.path.join(self.base_path, f"{self.dimension}_pi.pt")

        def meta_file(self) -> str:
            return os.path.join(self.base_path, f"{self.dimension}.meta.json")

        def tuner_file(self) -> str:
            return os.path.join(self.base_path, f"{self.dimension}.tuner.json")

        def scaler_file(self) -> str:
            return os.path.join(self.base_path, f"{self.dimension}_scaler.joblib")

    def get_model_name(self) -> str:
        return f"{self.target_type}_{self.model_type}_{self.model_version}"

    def get_locator(self, dimension: str):
        return self.Locator(
            root_dir=self.model_path,  # Path to the root directory for models
            model_type=self.model_type,
            target_type=self.target_type,
            dimension=dimension,
            version=self.version,
            embedding_type=self.embedding_type,
        )
