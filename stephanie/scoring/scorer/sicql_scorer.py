# stephanie/scoring/scorer/sicql_scorer.py
from __future__ import annotations

import os
from typing import Dict, List

import torch
import torch.nn.functional as F

from stephanie.constants import GOAL, GOAL_TEXT
from stephanie.data.score_bundle import ScoreBundle
from stephanie.data.score_result import ScoreResult
from stephanie.scoring.model.in_context_q import InContextQModel
from stephanie.scoring.model.policy_head import PolicyHead
from stephanie.scoring.model.q_head import QHead
from stephanie.scoring.model.text_encoder import TextEncoder
from stephanie.scoring.model.v_head import VHead
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorer.base_scorer import BaseScorer
from stephanie.scoring.transforms.regression_tuner import RegressionTuner
from stephanie.utils.file_utils import load_json
from stephanie.utils.model_locator import ModelLocator


class SICQLScorer(BaseScorer):
    """
    Inference-time scorer for SICQL (Q/V/π). Produces a single scalar score per dimension,
    plus useful attributes (Q, V, logits, entropy, advantage, optional zsa).
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.model_type = "sicql"
        self.embedding_type = self.memory.embedding.name
        self.dim = memory.embedding.dim
        self.hdim = memory.embedding.hdim

        self.target_type = cfg.get("target_type", "document")
        self.model_path = cfg.get("model_path", "models")
        self.version = cfg.get("model_version", "v1")
        self.return_zsa = bool(cfg.get("return_zsa", False))

        self.models: Dict[str, InContextQModel] = {}
        self.model_meta: Dict[str, dict] = {}
        self.tuners: Dict[str, RegressionTuner] = {}

        self.dimensions = cfg.get("dimensions", [])
        self._load_models(self.dimensions)

    # ---------- helpers ----------

    def _ensure_flat_logits(self, t: torch.Tensor) -> List[float]:
        """
        Accepts (1, A) or (A,) → returns python list length A.
        """
        t = t.detach().cpu()
        if t.ndim > 1:
            t = t.view(-1)
        return t.tolist()

    def _scale_with_tuner_or_sigmoid(self, dim: str, q_value: float, meta: dict) -> float:
        """
        If a RegressionTuner exists, use it. Otherwise, sigmoid → [0..1] → [min,max].
        """
        min_val = float(meta.get("min_value", 0.0))
        max_val = float(meta.get("max_value", 100.0))
        if dim in self.tuners and self.tuners[dim] is not None:
            s = float(self.tuners[dim].transform(q_value))
        else:
            s01 = torch.sigmoid(torch.tensor(q_value, dtype=torch.float32)).item()
            s = s01 * (max_val - min_val) + min_val
        # clamp to declared domain
        return max(min(s, max_val), min_val)

    def _get_locator(self, dim: str) -> ModelLocator:
        return ModelLocator(
            root_dir=self.model_path,
            embedding_type=self.embedding_type,
            model_type=self.model_type,
            target_type=self.target_type,
            dimension=dim,
            version=self.version,
        )

    # ---------- load ----------

    def _load_models(self, dimensions):
        for dim in dimensions:
            loc = self._get_locator(dim)

            # build modules
            encoder = TextEncoder(dim=self.dim, hdim=self.hdim).to(self.device)
            q_head  = QHead(zsa_dim=self.dim, hdim=self.hdim).to(self.device)
            v_head  = VHead(zsa_dim=self.dim, hdim=self.hdim).to(self.device)
            pi_head = PolicyHead(zsa_dim=self.dim, hdim=self.hdim, num_actions=3).to(self.device)

            # load weights (best-effort)
            encoder.load_state_dict(torch.load(loc.encoder_file(), map_location=self.device))
            q_head.load_state_dict(torch.load(loc.q_head_file(), map_location=self.device))
            v_head.load_state_dict(torch.load(loc.v_head_file(), map_location=self.device))
            pi_head.load_state_dict(torch.load(loc.pi_head_file(), map_location=self.device))

            model = InContextQModel(
                encoder=encoder,
                q_head=q_head,
                v_head=v_head,
                pi_head=pi_head,
                embedding_store=self.memory.embedding,
                device=self.device,
            )
            self.models[dim] = model

            meta = load_json(loc.meta_file()) if os.path.exists(loc.meta_file()) else {"min_value": 0.0, "max_value": 100.0}
            self.model_meta[dim] = meta

            tuner_path = loc.tuner_file()
            if os.path.exists(tuner_path):
                tuner = RegressionTuner(dimension=dim)
                tuner.load(tuner_path)
                self.tuners[dim] = tuner

    # ---------- score API ----------

    def _score_core(self, context: dict, scorable: Scorable, dimensions: list[str]) -> ScoreBundle:
        goal = context.get(GOAL, {})
        goal_text = goal.get(GOAL_TEXT, "") or ""

        # precompute embeddings once per call
        prompt_emb_np = self.memory.embedding.get_or_create(goal_text)
        output_emb_np = self.memory.embedding.get_or_create(scorable.get("text"))

        prompt_emb = torch.tensor(prompt_emb_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        output_emb = torch.tensor(output_emb_np, dtype=torch.float32, device=self.device).unsqueeze(0)

        results: Dict[str, ScoreResult] = {}

        for dim in dimensions:
            model = self.models.get(dim)
            if model is None:
                # no model loaded → skip dimension
                continue

            with torch.no_grad():
                outs = model(prompt_emb, output_emb)
                q_value = float(outs["q_value"].view(-1)[0].item())
                v_value = float(outs["state_value"].view(-1)[0].item())
                logits  = self._ensure_flat_logits(outs["action_logits"])

                # policy metrics
                probs_t = F.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1)
                entropy = float(-(probs_t * torch.log(probs_t + 1e-8)).sum().item())
                advantage = q_value - v_value

                zsa_tensor = None
                if self.return_zsa:
                    zsa_tensor = model.encoder(prompt_emb, output_emb)  # already no_grad

            meta = self.model_meta.get(dim, {"min_value": 0.0, "max_value": 100.0})
            final_score = round(self._scale_with_tuner_or_sigmoid(dim, q_value, meta), 4)

            rationale = f"Q={q_value:.4f}, V={v_value:.4f}, Δ={abs(advantage):.3f}, H={entropy:.3f}"

            attributes = {
                "q_value": q_value,
                "energy": q_value,            # legacy alias
                "state_value": v_value,
                "policy_logits": logits,
                "uncertainty": abs(advantage),
                "entropy": entropy,
                "advantage": advantage,
            }
            if self.return_zsa and zsa_tensor is not None:
                # pass tensor; downstream should handle serialization if needed
                attributes["zsa"] = zsa_tensor
                rationale += f", zsa_dim={int(zsa_tensor.shape[-1])}"

            results[dim] = ScoreResult(
                dimension=dim,
                source=self.name,
                score=final_score,
                rationale=rationale,
                weight=1.0,
                attributes=attributes,
            )

        return ScoreBundle(results=results)

    # ---------- convenience ----------

    def __call__(self, goal: dict, scorable: Scorable, dimension: str):
        """
        Quick raw forward for a single dimension (debugging / analysis).
        """
        model = self.models.get(dimension)
        if model is None:
            raise ValueError(f"Model for dimension '{dimension}' not loaded.")

        prompt_emb = torch.tensor(self.memory.embedding.get_or_create(goal.get("goal_text", "")),
                                  device=self.device, dtype=torch.float32).unsqueeze(0)
        output_emb = torch.tensor(self.memory.embedding.get_or_create(scorable.text),
                                  device=self.device, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            outs = model(prompt_emb, output_emb)
            if self.return_zsa and "zsa" not in outs:
                outs["zsa"] = model.encoder(prompt_emb, output_emb)
        return outs

    def get_model(self, dimension: str) -> InContextQModel:
        model = self.models.get(dimension)
        if model is None:
            raise ValueError(f"Model for dimension '{dimension}' not loaded.")
        return model
