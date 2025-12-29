# stephanie/scoring/scorer/sicql_scorer.py
from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

from stephanie.constants import GOAL, GOAL_TEXT
from stephanie.data.score_bundle import ScoreBundle
from stephanie.data.score_result import ScoreResult
from stephanie.scoring.model.sicql import (InContextQModel, PolicyHead, QHead,
                                           VHead)
from stephanie.scoring.model.text_encoder import TextEncoder
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorer.base_scorer import BaseScorer
from stephanie.scoring.transforms.regression_tuner import RegressionTuner
from stephanie.utils.file_utils import load_json
from stephanie.utils.model_locator import ModelLocator
from stephanie.scoring.scorer.model_health import audit_load_state_dict, merge_load_audits

import logging
log = logging.getLogger(__name__)

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

    def _clamp01(self, x: float) -> float:
        return 0.0 if x != x else max(0.0, min(1.0, float(x)))

    def _scale_with_tuner_or_sigmoid(self, dim: str, q_value: float, meta: dict) -> tuple[float, float]:
        """
        Always produce:
        - score01 in [0,1]  (authoritative, returned as ScoreResult.score)
        - score_scaled in [min,max] (kept as attribute for display)
        """
        min_val = float(meta.get("min_value", 0.0))
        max_val = float(meta.get("max_value", 100.0))
        denom = (max_val - min_val) if (max_val - min_val) != 0 else 1.0

        if dim in self.tuners and self.tuners[dim] is not None:
            # tuner likely outputs in scaled units (often 0..100)
            score_scaled = float(self.tuners[dim].transform(q_value))
            score01 = self._clamp01((score_scaled - min_val) / denom)
        else:
            # base path: sigmoid(q) gives score01 directly
            score01 = float(torch.sigmoid(torch.tensor(q_value, dtype=torch.float32)).item())
            score01 = self._clamp01(score01)
            score_scaled = score01 * denom + min_val

        # clamp scaled into declared range
        score_scaled = max(min(score_scaled, max_val), min_val)
        return score01, score_scaled

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
            ok_enc = _safe_load(encoder, loc.encoder_file(), name=f"SICQL encoder[{dim}]", device=self.device, strict=True)
            ok_q   = _safe_load(q_head,  loc.q_head_file(),  name=f"SICQL QHead[{dim}]",  device=self.device, strict=True)
            ok_v   = _safe_load(v_head,  loc.v_head_file(),  name=f"SICQL VHead[{dim}]",  device=self.device, strict=True)
            ok_pi  = _safe_load(pi_head, loc.pi_head_file(), name=f"SICQL Policy[{dim}]", device=self.device, strict=True)

            load_ok = ok_enc and ok_q and ok_v and ok_pi
            if not load_ok:
                # This is the important part: fail fast with a clear message.
                # (Or set a flag and skip this dimension.)
                raise RuntimeError(f"SICQL model load failed for dim={dim} (see logs above for shape mismatch)")

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
        output_emb_np = self.memory.embedding.get_or_create(scorable.text)

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
            final_score, score_scaled = self._scale_with_tuner_or_sigmoid(dim, q_value, meta)

            rationale = f"Q={q_value:.4f}, V={v_value:.4f}, Δ={abs(advantage):.3f}, H={entropy:.3f}"

            attributes = {
                "score_scaled": score_scaled,
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


def _peek_shapes(sd: Dict[str, Any], keys: Tuple[str, ...]) -> Dict[str, Tuple[int, ...]]:
    out = {}
    for k in keys:
        v = sd.get(k)
        if isinstance(v, torch.Tensor):
            out[k] = tuple(v.shape)
    return out

def _safe_load(module, path: str, *, name: str, device: str, strict: bool = True) -> bool:
    """
    Loads a checkpoint with useful shape logging.
    Returns True on success, False on RuntimeError (shape mismatch, etc).
    """
    sd = torch.load(path, map_location=device)

    # log a few likely keys so we can see in/out dims immediately
    peek_keys = (
        "encoder.0.weight", "encoder.0.bias",         # if TextEncoder uses nn.Sequential
        "model.0.weight", "model.0.bias",             # QHead
        "net.0.weight", "net.0.bias",                 # VHead
        "linear.0.weight", "linear.0.bias",           # PolicyHead
    )
    peek = _peek_shapes(sd, peek_keys)
    if peek:
        log.info("%s shapes=%s ckpt=%s", name, peek, path)
    else:
        # fall back: show a couple first keys
        some = list(sd.keys())[:6]
        log.info("%s ckpt=%s keys_preview=%s", name, path, some)

    try:
        module.load_state_dict(sd, strict=strict)
        log.info("%s loaded ok strict=%s", name, strict)
        return True
    except RuntimeError as e:
        log.warning("%s FAILED to load strict=%s err=%s", name, strict, str(e).splitlines()[0])
        return False
