import os

import torch
import torch.nn.functional as F

from stephanie.models.score import ScoreORM
from stephanie.scoring.base_scorer import BaseScorer
from stephanie.scoring.model.ebt_model import EBTModel
from stephanie.scoring.model.text_encoder import TextEncoder
from stephanie.scoring.scorable import Scorable
from stephanie.data.score_bundle import ScoreBundle
from stephanie.data.score_result import ScoreResult
from stephanie.scoring.transforms.regression_tuner import RegressionTuner
from stephanie.utils.file_utils import load_json
from stephanie.utils.model_locator import ModelLocator


class EBTScorer(BaseScorer):
    def __init__(self, cfg, memory, logger):
        super().__init__(cfg, memory, logger)
        self.model_type = "ebt"
        self.embedding_type = memory.embedding.type
        self.dim = memory.embedding.dim
        self.hdim = memory.embedding.hdim

        self.target_type = cfg.get("target_type", "document")
        self.model_path = cfg.get("model_path", "models")
        self.version = cfg.get("model_version", "v1")
        self.dimensions = cfg.get("dimensions", [])

        self.models = {}
        self.model_meta = {}
        self.tuners = {}

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

            model = EBTModel(
                embedding_dim=self.dim,
                hidden_dim=self.hdim,
                num_actions=3,
                device=self.device
            ).to(self.device)

            model.encoder.load_state_dict(
                torch.load(locator.encoder_file(), map_location=self.device)
            )
            model.q_head.load_state_dict(
                torch.load(locator.q_head_file(), map_location=self.device)
            )
            model.v_head.load_state_dict(
                torch.load(locator.v_head_file(), map_location=self.device)
            )
            model.pi_head.load_state_dict(
                torch.load(locator.pi_head_file(), map_location=self.device)
            )
            
            model.eval()
            self.models[dim] = model

            meta = load_json(locator.meta_file()) if os.path.exists(locator.meta_file()) else {"min_value": 0, "max_value": 100}
            self.model_meta[dim] = meta

            if os.path.exists(locator.tuner_file()):
                tuner = RegressionTuner(dimension=dim)
                tuner.load(locator.tuner_file())
                self.tuners[dim] = tuner

    def score(self, goal: dict, scorable: Scorable, dimensions: list[str]) -> ScoreBundle:
        goal_text = goal.get("goal_text")
        results = {}

        for dim in dimensions:
            model = self.models.get(dim)
            if model is None:
                continue

            ctx_emb = torch.tensor(
                self.memory.embedding.get_or_create(goal_text), device=self.device
            ).unsqueeze(0)
            doc_emb = torch.tensor(
                self.memory.embedding.get_or_create(scorable.text), device=self.device
            ).unsqueeze(0)

            with torch.no_grad():
                result = model(ctx_emb, doc_emb)

            q_value = result["q_value"].item()
            v_value = result["state_value"].item() if "state_value" in result else 0.0
            policy_logits = result.get("action_logits", torch.zeros(1, 3)).cpu().squeeze().tolist()

            uncertainty = abs(q_value - v_value)
            advantage = q_value - v_value

            action_probs = F.softmax(torch.tensor(policy_logits), dim=-1)
            entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8)).item()

            meta = self.model_meta.get(dim, {"min_value": 0, "max_value": 100})
            if dim in self.tuners:
                scaled_score = self.tuners[dim].transform(q_value)
            else:
                normalized = torch.sigmoid(torch.tensor(q_value)).item()
                scaled_score = normalized * (meta["max_value"] - meta["min_value"]) + meta["min_value"]

            final_score = round(max(min(scaled_score, meta["max_value"]), meta["min_value"]), 4)

            rationale = f"Q={q_value:.4f}, V={v_value:.4f}, Δ={uncertainty:.3f}, H={entropy:.3f}"

            attributes = {
                "q_value": round(q_value, 4),
                "v_value": round(v_value, 4),
                "normalized_score": round(scaled_score, 4),
                "final_score": final_score,
                "energy": q_value,
                "state_value": v_value,
                "policy_logits": policy_logits,
                "uncertainty": uncertainty,
                "entropy": entropy,
                "advantage": advantage,
            }

            results[dim] = ScoreResult(
                dimension=dim,
                score=final_score,
                source=self.model_type,
                rationale=rationale,
                weight=1.0, 
                attributes=attributes,
            )

        return ScoreBundle(results=results)
