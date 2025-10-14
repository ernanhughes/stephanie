# stephanie/scoring/mrq/mrq_scorer.py
from __future__ import annotations

import os

import torch

from stephanie.constants import GOAL
from stephanie.data.score_bundle import ScoreBundle
from stephanie.data.score_result import ScoreResult
from stephanie.evaluator.hypothesis_value_predictor import \
    HypothesisValuePredictor
from stephanie.scoring.model.mrq_model import MRQModel
from stephanie.scoring.model.text_encoder import TextEncoder
from stephanie.scoring.scorer.base_scorer import BaseScorer
from stephanie.scoring.transforms.regression_tuner import RegressionTuner
from stephanie.utils.file_utils import load_json
from stephanie.utils.model_locator import ModelLocator


class MRQScorer(BaseScorer):
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.model_type = "mrq"
        self.embedding_type = self.memory.embedding.name
        self.dim = memory.embedding.dim
        self.hdim = memory.embedding.hdim
        self.target_type = cfg.get("target_type", "document")
        self.model_path = cfg.get("model_path", "models")
        self.version = cfg.get("model_version", "v1")

        self.models = {}
        self.model_meta = {}
        self.tuners = {}

        self.dimensions = cfg.get("dimensions", [])
        self._load_models(self.dimensions)

    def _load_models(self, dimensions):
        for dim in dimensions:
            locator = ModelLocator(
                root_dir=self.model_path,
                embedding_type=self.embedding_type,
                model_type=self.model_type,
                target_type=self.target_type,
                dimension=dim,
                version=self.version,
            )

            encoder = TextEncoder(self.dim, self.hdim)
            predictor = HypothesisValuePredictor(self.dim, self.hdim)
            model = MRQModel(encoder, predictor, self.memory.embedding, device=self.device)
            model.load_weights(locator.encoder_file(), locator.model_file())
            self.models[dim] = model

            meta = load_json(locator.meta_file()) if os.path.exists(locator.meta_file()) else {"min_value": 0, "max_value": 100}
            self.model_meta[dim] = meta

            tuner_path = locator.tuner_file()
            if os.path.exists(tuner_path):
                tuner = RegressionTuner(dimension=dim)
                tuner.load(tuner_path)
                self.tuners[dim] = tuner

    def score(self, context: dict, scorable, dimensions: list[str]) -> ScoreBundle:
        goal = context.get(GOAL, {})
        goal_text = goal.get("goal_text")
        results = {}

        for dim in dimensions:
            if isinstance(dim, dict):
                dimension_name = dim.get("name")
            else:
                dimension_name = dim
            model = self.models.get(dimension_name)
            if not model:
                continue

            q_value = model.predict(goal_text, scorable.text)

            meta = self.model_meta.get(dimension_name, {"min_value": 0, "max_value": 100})
            tuner = self.tuners.get(dimension_name)

            if tuner:
                scaled = tuner.transform(q_value)
            else:
                norm = torch.sigmoid(torch.tensor(q_value)).item()
                if norm < 0.01 or norm > 0.99:
                    self.logger.log("QValueOutlier", {"dimension": dim, "q_value": q_value})
                scaled = norm * (meta["max_value"] - meta["min_value"]) + meta["min_value"]

            final_score = round(max(min(scaled, meta["max_value"]), meta["min_value"]), 4)

            attributes = {
                "q_value": round(q_value, 4),
                "normalized_score": round(scaled, 4),
                "energy": q_value,
            }

            results[dimension_name] = ScoreResult(
                dimension=dimension_name,
                score=final_score,
                source=self.model_type,
                rationale=f"Q={round(q_value, 4)}",
                weight=1.0,
                attributes=attributes,)

        return ScoreBundle(results=results)
