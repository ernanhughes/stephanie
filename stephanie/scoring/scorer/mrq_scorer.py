# stephanie/scoring/mrq/mrq_scorer.py
from __future__ import annotations

import os
from typing import Dict, List, Union

import torch

from stephanie.constants import GOAL, GOAL_TEXT
from stephanie.data.score_bundle import ScoreBundle
from stephanie.data.score_result import ScoreResult
from stephanie.scoring.model.mrq_model import MRQModel
from stephanie.scoring.model.text_encoder import TextEncoder
from stephanie.scoring.model.value_predictor import ValuePredictor  # <-- matches trainer
from stephanie.scoring.scorer.base_scorer import BaseScorer
from stephanie.scoring.transforms.regression_tuner import RegressionTuner
from stephanie.utils.file_utils import load_json
from stephanie.utils.model_locator import ModelLocator


class MRQScorer(BaseScorer):
    """
    MRQ scorer: computes a pairwise logit for (goal, output) via encoder→predictor,
    then scales to the dimension's range using either a RegressionTuner (on prob) or sigmoid.
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.model_type = "mrq"
        self.embedding_type = self.memory.embedding.name
        self.dim = memory.embedding.dim
        self.hdim = memory.embedding.hdim

        self.target_type = cfg.get("target_type", "document")
        self.model_path = cfg.get("model_path", "models")
        self.version = cfg.get("model_version", "v1")

        self.models: Dict[str, MRQModel] = {}
        self.model_meta: Dict[str, dict] = {}
        self.tuners: Dict[str, RegressionTuner] = {}

        self.dimensions: List[str] = cfg.get("dimensions", [])
        self._load_models(self.dimensions)

    # ---------- loading ----------

    def _locator(self, dim: str) -> ModelLocator:
        return ModelLocator(
            root_dir=self.model_path,
            embedding_type=self.embedding_type,
            model_type=self.model_type,
            target_type=self.target_type,
            dimension=dim,
            version=self.version,
        )

    def _load_models(self, dimensions: List[str]) -> None:
        for dim in dimensions:
            loc = self._locator(dim)

            # build modules consistent with trainer
            encoder = TextEncoder(dim=self.dim, hdim=self.hdim).to(self.device)
            predictor = ValuePredictor(zsa_dim=self.dim, hdim=self.hdim).to(self.device)

            # load weights saved by trainer
            encoder.load_state_dict(torch.load(loc.encoder_file(), map_location=self.device))
            predictor.load_state_dict(torch.load(loc.model_file(), map_location=self.device))

            model = MRQModel(
                encoder=encoder,
                predictor=predictor,
                embedding_store=self.memory.embedding,
                device=self.device,
            )
            self.models[dim] = model

            # meta + tuner
            meta = load_json(loc.meta_file()) if os.path.exists(loc.meta_file()) else {"min_value": 0.0, "max_value": 100.0}
            self.model_meta[dim] = meta

            tuner_path = loc.tuner_file()
            if os.path.exists(tuner_path):
                t = RegressionTuner(dimension=dim)
                t.load(tuner_path)
                self.tuners[dim] = t

    # ---------- scoring ----------

    def _scale(self, dim: str, logit: float, meta: dict) -> float:
        """
        Trainer fed the tuner with probabilities (sigmoid), not logits.
        Respect that here: tuner.transform(prob) if present, else sigmoid→[min,max].
        """
        min_v = float(meta.get("min_value", 0.0))
        max_v = float(meta.get("max_value", 100.0))
        prob = torch.sigmoid(torch.tensor(logit, dtype=torch.float32)).item()

        if dim in self.tuners:
            val = float(self.tuners[dim].transform(prob))
        else:
            val = prob * (max_v - min_v) + min_v

        # clamp into the declared domain
        return max(min(val, max_v), min_v)

    def score(self, context: dict, scorable, dimensions: List[Union[str, dict]]) -> ScoreBundle:
        goal = context.get(GOAL, {})
        goal_text = goal.get(GOAL_TEXT, "") or ""

        # precompute embeddings once per call
        g_np = self.memory.embedding.get_or_create(goal_text)
        o_np = self.memory.embedding.get_or_create(scorable.text)
        g = torch.tensor(g_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        o = torch.tensor(o_np, dtype=torch.float32, device=self.device).unsqueeze(0)

        results: Dict[str, ScoreResult] = {}

        for dim in dimensions:
            dim_name = dim.get("name") if isinstance(dim, dict) else dim
            model = self.models.get(dim_name)
            if model is None:
                continue

            with torch.no_grad():
                # encoder(project goal, output) → z; predictor(z) → logit (scalar)
                z = model.encoder(g, o)                # [1, D]
                logit = float(model.predictor(z).view(-1)[0].item())
                prob = float(torch.sigmoid(torch.tensor(logit)).item())

            meta = self.model_meta.get(dim_name, {"min_value": 0.0, "max_value": 100.0})
            scaled = self._scale(dim_name, logit, meta)
            final_score = round(scaled, 4)

            # attributes for diagnostics
            attributes = {
                "q_value": round(logit, 6),        # raw logit
                "prob": round(prob, 6),             # sigmoid(logit)
                "energy": logit,                    # alias kept for continuity
                "min_value": meta.get("min_value", 0.0),
                "max_value": meta.get("max_value", 100.0),
            }

            results[dim_name] = ScoreResult(
                dimension=dim_name,
                score=final_score,
                source=self.model_type,
                rationale=f"logit={logit:.4f}, prob={prob:.4f}",
                weight=1.0,
                attributes=attributes,
            )

        return ScoreBundle(results=results)
