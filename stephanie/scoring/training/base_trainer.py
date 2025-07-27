import json
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class BaseTrainer:
    class Locator:
        def __init__(self, root_dir, model_type, target_type, dimension, version, embedding_type):
            self.root_dir = root_dir
            self.model_type = model_type
            self.target_type = target_type
            self.dimension = dimension
            self.version = version
            self.embedding_type = embedding_type

        @property
        def base_path(self):
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

        def encoder_file(self) -> str:
            return os.path.join(self.base_path, f"{self.dimension}_encoder.pt")

        def q_head_file(self) -> str:
            return os.path.join(self.base_path, f"{self.dimension}_q.pt")

        def v_head_file(self) -> str:
            return os.path.join(self.base_path, f"{self.dimension}_v.pt")

        def pi_head_file(self) -> str:
            return os.path.join(self.base_path, f"{self.dimension}_pi.pt")

        def meta_file(self) -> str:
            return os.path.join(self.base_path, f"{self.dimension}.meta.json")

        def tuner_file(self) -> str:
            return os.path.join(self.base_path, f"{self.dimension}.tuner.json")

        def scaler_file(self) -> str:
            return os.path.join(self.base_path, f"{self.dimension}_scaler.joblib")

        def joblib_file(self) -> str:
            return os.path.join(self.base_path, f"{self.dimension}_model.joblib")

        def model_file(self, suffix: str = ".pt") -> str:
            return os.path.join(self.base_path, f"{self.dimension}{suffix}")


        def model_exists(self) -> bool:
            return False

    def __init__(self, cfg, memory=None, logger=None):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger
        self.embedding_type = memory.embedding.type
        self.dim = memory.embedding.dim
        self.hdim = memory.embedding.hdim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.root_dir = cfg.get("model_path", "models")
        self.version = cfg.get("model_version", "v1")
        self.target_type = cfg.get("target_type", "document")
        self.model_type = cfg.get("model_type", "base")
        self.dimensions = cfg.get("dimensions", [])
        self.min_samples = cfg.get("min_samples", 5)

    def get_locator(self, dimension):
        return self.Locator(
            root_dir=self.root_dir,
            model_type=self.model_type,
            target_type=self.target_type,
            dimension=dimension,
            version=self.version,
            embedding_type=self.embedding_type,
        )

    def _create_dataloader(self, samples):
        valid = []
        for s in samples:
            ctx_text = s.get("title", "")
            doc_text = s.get("output", "")
            score = s.get("score", 0.5)

            if not ctx_text or not doc_text or not isinstance(score, (float, int)):
                continue

            ctx_emb = torch.tensor(self.memory.embedding.get_or_create(ctx_text)).to(self.device)
            doc_emb = torch.tensor(self.memory.embedding.get_or_create(doc_text)).to(self.device)

            valid.append({"context": ctx_emb, "document": doc_emb, "score": score})

        if len(valid) < self.min_samples:
            return None

        return DataLoader(
            TensorDataset(
                torch.stack([s["context"] for s in valid]),
                torch.stack([s["document"] for s in valid]),
                torch.tensor([s["score"] for s in valid])
            ),
            batch_size=self.cfg.get("batch_size", 32),
            shuffle=True
        )

    def _save_meta_file(self, meta: dict, dimension: str):
        locator = self.get_locator(dimension)
        with open(locator.meta_file(), "w") as f:
            json.dump(meta, f)

    def _calculate_policy_metrics(self, logits):
        probs = F.softmax(torch.tensor(logits), dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
        stability = probs.max().item()
        return entropy, stability

    def log_event(self, name: str, payload: dict):
        if self.logger:
            self.logger.log(name, payload)

