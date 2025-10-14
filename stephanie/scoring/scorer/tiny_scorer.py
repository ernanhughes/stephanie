# stephanie/scoring/tiny_scorer.py
from __future__ import annotations


import logging
import os
from typing import Dict, Any

import torch

from stephanie.constants import GOAL, GOAL_TEXT
from stephanie.data.score_bundle import ScoreBundle
from stephanie.data.score_result import ScoreResult
from stephanie.scoring.model.tiny_recursion import TinyRecursionModel
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorer.base_scorer import BaseScorer
from stephanie.utils.file_utils import load_json

_logger = logging.getLogger(__name__)


class TinyScorer(BaseScorer):
    """
    Scorer that uses a trained TinyRecursionModel (TRM) to evaluate goal/document pairs.
    Tiny model runs a few recursive refinement steps in embedding space and predicts a quality score.
    """
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.model_type = "tiny"  # identifies scorer type in results

        # Embedding interface (shared with HRM)
        self.embedding_type = self.memory.embedding.name
        self.dim = self.memory.embedding.dim

        # Config
        self.target_type = cfg.get("target_type", "document")
        self.model_path = cfg.get("model_path", "models")
        self.version = cfg.get("model_version", "v1")
        self.dimensions = cfg.get("dimensions", [])

        # Optional output scaling (keep consistent with HRM if desired)
        self.clip_0_100 = cfg.get("clip_0_100", True)

        # Containers for per-dimension models and metadata
        self.models: Dict[str, TinyRecursionModel] = {}
        self.model_meta: Dict[str, Dict[str, Any]] = {}

        # Attempt to load models up-front
        self._load_models(self.dimensions)

    # -------------------------
    # Loading
    # -------------------------
    def _load_models(self, dimensions):
        """
        Loads TRM models/metadata per dimension using the same locator pattern as HRMScorer.
        Expects weights saved with suffix `_tiny.pt` and a companion meta.json.
        """
        for dimension in dimensions:
            try:
                locator = self.get_locator(dimension)

                model_file_path = locator.model_file(suffix="_tiny.pt")  # mirror HRM's "_hrm.pt"
                meta_file_path = locator.meta_file()

                if not os.path.exists(model_file_path):
                    self.logger.log("TinyScorerModelError", {
                        "message": "Tiny model file not found.",
                        "path": model_file_path,
                        "dimension": dimension,
                    })
                    continue

                # Load meta if present
                if os.path.exists(meta_file_path):
                    self.model_meta[dimension] = load_json(meta_file_path)
                    self.logger.log("TinyScorerMetaLoaded", {
                        "dimension": dimension,
                        "meta": {k: self.model_meta[dimension].get(k) for k in (
                            "input_dim","d_model","dz","steps","n_blocks","dropout"
                        )}
                    })
                else:
                    self.model_meta[dimension] = {}
                    self.logger.log("TinyScorerWarning", {
                        "message": "Tiny meta file not found. Using defaults.",
                        "path": meta_file_path,
                        "dimension": dimension,
                    })

                # Reconstruct tiny model config from meta (defaults are safe)
                tm_cfg = {
                    # Tiny model consumes concatenated (ctx_emb, doc_emb)
                    "input_dim": self.model_meta[dimension].get("input_dim", self.dim * 2),
                    "d_model":   self.model_meta[dimension].get("d_model", 256),
                    "dz":        self.model_meta[dimension].get("dz", 256),
                    "steps":     self.model_meta[dimension].get("steps", 6),
                    "n_blocks":  self.model_meta[dimension].get("n_blocks", 2),
                    "dropout":   self.model_meta[dimension].get("dropout", 0.1),
                    "use_halting": self.model_meta[dimension].get("use_halting", True),
                    "out_dim":   self.model_meta[dimension].get("out_dim", 1),  # scalar quality
                }

                model = TinyRecursionModel(tm_cfg, logger=self.logger)
                model.to(self.device)
                state = torch.load(model_file_path, map_location=self.device)
                model.load_state_dict(state)
                model.eval()

                self.models[dimension] = model

                self.logger.log("TinyScorerModelLoaded", {
                    "dimension": dimension,
                    "model_path": model_file_path,
                    "device": str(self.device),
                })

            except Exception as e:
                self.logger.log("TinyScorerInitError", {
                    "message": "Failed to load tiny model.",
                    "dimension": dimension,
                    "error": str(e),
                })

    # -------------------------
    # Scoring
    # -------------------------
    def score(self, context: dict, scorable: Scorable, dimensions: list[str]) -> ScoreBundle:
        """
        Scores a single scorable item using TinyRecursionModel per dimension.
        Returns a ScoreBundle with ScoreResult per requested dimension.
        """
        results: Dict[str, ScoreResult] = {}

        goal = context.get(GOAL, {})
        goal_text = goal.get(GOAL_TEXT, "")
        doc_text = scorable.text

        if not goal_text or not doc_text:
            self.logger.log("TinyScorerWarning", {
                "message": "Missing goal_text or scorable text.",
                "goal": goal.get("id", "unknown"),
                "scorable_id": scorable.id,
            })
            return ScoreBundle(results={})

        # 1) Embeddings
        ctx_emb_np = self.memory.embedding.get_or_create(goal_text)
        doc_emb_np = self.memory.embedding.get_or_create(doc_text)

        # 2) To tensors + concat (B=1)
        ctx_emb = torch.tensor(ctx_emb_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        doc_emb = torch.tensor(doc_emb_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        x_input = torch.cat([ctx_emb, doc_emb], dim=-1)  # shape [1, dim*2]

        # 3) Per-dimension evaluation
        for dimension in dimensions:
            model = self.models.get(dimension)
            if not model:
                self.logger.log("TinyScorerError", {
                    "message": f"Tiny model not found for dimension '{dimension}'. Skipping.",
                    "goal_id": goal.get("id", "unknown"),
                    "scorable_id": scorable.id,
                })
                continue

            try:
                with torch.no_grad():
                    # forward returns (score_pred, intermediates)
                    # intermediates may include: z_final, halt_p, steps_used, y0, yT
                    y_pred, intermediates = model(x_input)

                raw = float(y_pred.squeeze().item())
                score = max(0.0, min(100.0, raw)) if self.clip_0_100 else raw

                # Extract helpful attributes if available
                halt_p = self._get(intermediates, "halt_p")
                steps_used = self._get(intermediates, "steps_used")
                z_final = self._get(intermediates, "z_final")
                y0 = self._get(intermediates, "y0")
                yT = self._get(intermediates, "yT")

                z_mag = float(torch.norm(z_final, p=2).item()) if isinstance(z_final, torch.Tensor) else None
                stability = None
                if isinstance(y0, torch.Tensor) and isinstance(yT, torch.Tensor):
                    denom = torch.norm(y0, p=2).item() + 1e-9
                    stability = float(max(0.0, 1.0 - (torch.norm(yT - y0, p=2).item() / denom)))

                rationale = (
                    f"Tiny[{dimension}] raw={round(raw,4)} "
                    f"| halt_p={round(float(halt_p),4) if halt_p is not None else 'NA'} "
                    f"| steps={int(steps_used) if steps_used is not None else 'NA'} "
                    f"| z_mag={round(z_mag,4) if z_mag is not None else 'NA'} "
                    f"| stability={round(stability,4) if stability is not None else 'NA'}"
                )

                attributes = {
                    "raw_score": round(raw, 4),
                    "halt_p": float(halt_p) if halt_p is not None else None,
                    "steps_used": int(steps_used) if steps_used is not None else None,
                    "z_magnitude": z_mag,
                    "stability": stability,
                    # Keep alignment with HRM attributes where possible:
                    "q_value": raw,      # same convention as HRMScorer
                    "energy": raw,       # simple proxy; plug in critic later
                }

                results[dimension] = ScoreResult(
                    dimension=dimension,
                    score=score,
                    source=self.model_type,
                    rationale=rationale,
                    weight=1.0,
                    attributes=attributes,
                )

                _logger.debug(
                    "TinyScorerEvaluated "
                    f"dimension: {dimension} "
                    f"goal_id: {goal.get('id','unknown')} "
                    f"scorable_id: {scorable.id} "
                    f"raw: {raw} score: {score} halt_p: {halt_p} steps: {steps_used}"
                )

            except Exception as e:
                self.logger.log("TinyScorerError", {
                    "message": "Exception during Tiny scoring.",
                    "dimension": dimension,
                    "goal_id": goal.get("id", "unknown"),
                    "scorable_id": scorable.id,
                    "error": str(e),
                })

        return ScoreBundle(results=results)

    # -------------------------
    # Utils
    # -------------------------
    @staticmethod
    def _get(d: Dict[str, Any], key: str):
        try:
            return d.get(key)
        except Exception:
            return None

    def __repr__(self):
        loaded = {k: (v is not None) for k, v in self.models.items()}
        return f"<TinyScorer(model_type={self.model_type}, loaded={loaded})>"
