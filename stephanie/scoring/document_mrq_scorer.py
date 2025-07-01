# stephanie/scoring/document_mrq_scorer.py

import torch
from stephanie.evaluator.text_encoder import TextEncoder
from stephanie.scoring.document_value_predictor import DocumentValuePredictor
from stephanie.scoring.transforms.regression_tuner import RegressionTuner
import os


class DocumentMRQScorer:
    def __init__(self, memory, logger, device="cpu", dimensions=None, cfg=None, model_dir=None, model_prefix=None):
        self.memory = memory
        self.logger = logger
        self.device = device
        self.dimensions = dimensions or []
        self.cfg = cfg or {}

        # ✅ Accept model_dir/model_prefix from args or fallback to cfg or defaults
        self.model_dir = model_dir or self.cfg.get("model_dir", "models/document")
        self.model_prefix = model_prefix or self.cfg.get("model_prefix", "document_rm_")

        self.models = {}
        self.regression_tuners = {}
        self.min_score_by_dim = {}
        self.max_score_by_dim = {}

        self._initialize_dimensions()


    def _initialize_dimensions(self):
        for dim in self.dimensions:
            encoder = TextEncoder().to(self.device)
            predictor = DocumentValuePredictor(512, 1024).to(self.device)

            # Load model weights
            model_path = os.path.join(self.model_dir, f"{self.model_prefix}{dim}.pt")
            if os.path.exists(model_path):
                predictor.load_state_dict(torch.load(model_path, map_location=self.device))
                self.logger.log("DocumentMRQModelLoaded", {"dimension": dim, "path": model_path})
            else:
                self.logger.log("DocumentMRQModelMissing", {"dimension": dim, "path": model_path})

            # Load regression tuner if available
            tuner_path = os.path.join(self.model_dir, f"{self.model_prefix}{dim}_tuner.json")
            tuner = RegressionTuner(dimension=dim, logger=self.logger)
            if os.path.exists(tuner_path):
                tuner.load(tuner_path)
                self.logger.log("DocumentMRQTunerLoaded", {"dimension": dim, "path": tuner_path})
            else:
                self.logger.log("DocumentMRQTunerMissing", {"dimension": dim, "path": tuner_path})

            self.models[dim] = (encoder, predictor)
            self.regression_tuners[dim] = tuner
            self.min_score_by_dim[dim] = 0.0
            self.max_score_by_dim[dim] = 1.0

    def normalize_score(self, raw_score: float, dimension: str) -> float:
        min_val = self.min_score_by_dim.get(dimension, 0.0)
        max_val = self.max_score_by_dim.get(dimension, 1.0)
        return (raw_score - min_val) / (max_val - min_val + 1e-6)

    def score(self, goal_text: str, document_text: str, dimension: str) -> float:
        if dimension not in self.models:
            self.logger.log("DocumentMRQMissingDimension", {"dimension": dimension})
            return 0.0

        encoder, predictor = self.models[dimension]
        encoder.eval()
        predictor.eval()

        prompt_emb = torch.tensor(
            self.memory.embedding.get_or_create(goal_text), device=self.device
        ).unsqueeze(0)
        doc_emb = torch.tensor(
            self.memory.embedding.get_or_create(document_text), device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            zsa = encoder(prompt_emb, doc_emb)
            raw_score = predictor(zsa).item()

        norm_score = self.normalize_score(raw_score, dimension)

        tuner = self.regression_tuners.get(dimension)
        if tuner:
            tuned = tuner.transform(norm_score)
            self.logger.log("DocumentMRQTunedScore", {
                "dimension": dimension,
                "raw": norm_score,
                "tuned": tuned
            })
            return tuned

        return norm_score

    def train_tuner(self, dimension: str, mrq_score: float, llm_score: float):
        tuner = self.regression_tuners.get(dimension)
        if tuner:
            tuner.train_single(mrq_score, llm_score)

    def save_model(self, path_prefix: str):
        for dim, (encoder, predictor) in self.models.items():
            torch.save(predictor.state_dict(), f"{path_prefix}_{dim}.pt")
            self.regression_tuners[dim].save(f"{path_prefix}_{dim}_tuner.json")

    def load_model(self, path_prefix: str):
        for dim in self.dimensions:
            _, predictor = self.models[dim]
            predictor.load_state_dict(torch.load(f"{path_prefix}_{dim}.pt", map_location=self.device))
            self.regression_tuners[dim].load(f"{path_prefix}_{dim}_tuner.json")
