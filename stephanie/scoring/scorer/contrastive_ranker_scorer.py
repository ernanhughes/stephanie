"""
Contrastive (pairwise) ranker scorer.

This module implements a goal-conditioned (or goal-agnostic) Siamese preference
model that can (a) emit absolute scores by comparing a candidate to a per-dimension
baseline, and (b) perform direct A-vs-B pairwise comparisons.

# Dependencies on disk per dimension (via BaseScorer.get_locator(dim)):
- <dim>.meta.json           # {"hidden_dim": int, "baseline": str, "min_value": float, "max_value": float, optional "null_context": str}
- <dim>.scaler.joblib       # sklearn StandardScaler (or compatible) trained on [ctx||doc] features
- <dim>.pt                  # torch state_dict for PreferenceRanker
- <dim>.tuner.joblib        # RegressionTuner mapping raw logits → calibrated absolute score

# Config keys (typical):
- dimensions: [ "alignment", "relevance", ... ]
- goal_conditioned: true|false   # default true; if false, uses a null (fixed) context for comparisons

Author: your team
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
from joblib import load
from torch import nn
from tqdm import tqdm

from stephanie.data.score_bundle import ScoreBundle
from stephanie.data.score_result import ScoreResult
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorer.base_scorer import BaseScorer
from stephanie.scoring.transforms.regression_tuner import RegressionTuner
from stephanie.utils.file_utils import load_json


class PreferenceRanker(nn.Module):
    """
    Simple Siamese encoder + comparator.
    NOTE: Must match the trainer's architecture exactly.
    """
    def __init__(self, embedding_dim: int = 1024, hidden_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.comparator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
        """
        emb_a/b: (B, D) tensors already scaled/normalized as per training.
        Returns logits (B,) where higher means "a preferred over b".
        """
        feat_a = self.encoder(emb_a)
        feat_b = self.encoder(emb_b)
        combined = torch.cat([feat_a, feat_b], dim=1)
        return self.comparator(combined).squeeze(1)


class ContrastiveRankerScorer(BaseScorer):
    """
    Goal-conditioned contrastive scorer (with goal-agnostic fallback).

    - score(context, scorable, dims): absolute score per dimension by comparing
      [ctx||doc] against [ctx||baseline] then calibrating via RegressionTuner.
    - compare(context, a, b, dims): pairwise decision A vs B (probabilities per dim).

    Implementation notes:
    - Uses CPU by default. If you want CUDA, wire a device + state_dict accordingly.
    - Embeddings are pulled from memory.embedding; we assume deterministic dimension.
    """

    def __init__(self, cfg: dict, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.model_type = "contrastive_ranker"

        # Runtime containers keyed by dimension
        self.models: Dict[str, tuple] = {}        # dim -> (scaler, model)
        self.tuners: Dict[str, RegressionTuner] = {}   # dim -> tuner
        self.metas: Dict[str, dict] = {}          # dim -> metadata
        self.baselines: Dict[str, np.ndarray] = {}     # dim -> baseline embedding

        # Config
        self.goal_conditioned: bool = bool(getattr(self.cfg, "goal_conditioned", True))
        self.device = torch.device("cpu")  # Keep on CPU unless explicitly changed

        # Lazily created, fixed "null" context for goal-agnostic ops
        self._ctx_null: Optional[np.ndarray] = None

        # Load all configured dimensions (fail-soft if some are missing)
        self._load_all_dimensions()

    # ---------- helpers ---------- #

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))

    def _prep_pair_tensor(
        self,
        scaler,
        ctx_emb: np.ndarray,
        doc_emb: np.ndarray,
    ) -> torch.Tensor:
        """
        Build the exact input your trainer used: concatenate [ctx_emb || doc_emb],
        apply the joblib scaler, return as (1, D) float32 tensor on self.device.
        """
        pair = np.concatenate([ctx_emb, doc_emb]).reshape(1, -1)
        pair_scaled = scaler.transform(pair)
        return torch.tensor(pair_scaled, dtype=torch.float32, device=self.device)

    def _safe_embed(self, text: str) -> np.ndarray:
        """Fetch an embedding and return as float32 numpy array."""
        vec = self.memory.embedding.get_or_create(text or "")
        return np.asarray(vec, dtype=np.float32)

    def _load_all_dimensions(self) -> None:
        """
        Preload artifacts per dimension. If a dimension is mis-configured,
        we log and skip that dimension rather than failing the whole scorer.
        """
        dims = getattr(self, "dimensions", []) or []
        if not dims:
            self.logger.log("ContrastiveRankerNoDimensions", {})
            return

        for dim in tqdm(dims, desc="Loading contrastive rankers"):
            try:
                locator = self.get_locator(dim)

                # Meta
                meta = load_json(locator.meta_file()) or {}
                hidden_dim = int(meta.get("hidden_dim", 256))
                baseline_text = meta.get("baseline") or ""
                min_value = float(meta.get("min_value", 0.0))
                max_value = float(meta.get("max_value", 1.0))
                self.metas[dim] = {
                    "hidden_dim": hidden_dim,
                    "baseline": baseline_text,
                    "min_value": min_value,
                    "max_value": max_value,
                    "null_context": meta.get("null_context") or meta.get("baseline_context"),
                }

                # Scaler
                scaler = load(locator.scaler_file())
                input_dim = int(getattr(scaler, "mean_", np.zeros(1)).shape[0])

                # Model
                model = PreferenceRanker(embedding_dim=input_dim, hidden_dim=hidden_dim)
                state = torch.load(locator.model_file(suffix=".pt"), map_location=self.device)
                model.load_state_dict(state)
                model.eval()
                model.to(self.device)
                self.models[dim] = (scaler, model)

                # Tuner
                tuner = RegressionTuner(dimension=dim, logger=self.logger)
                tuner.load(locator.tuner_file())
                self.tuners[dim] = tuner

                # Baseline embedding
                baseline_emb = self._safe_embed(baseline_text) if baseline_text else np.zeros(
                    self.memory.embedding.dim, dtype=np.float32
                )
                self.baselines[dim] = baseline_emb

                self.logger.log("ContrastiveRankerDimReady", {
                    "dimension": dim, "input_dim": input_dim, "hidden_dim": hidden_dim
                })

            except Exception as e:
                self.logger.log("ContrastiveRankerDimLoadError", {"dimension": dim, "error": str(e)})

    def _get_context_embedding(self, context: Optional[dict]) -> np.ndarray:
        """
        Return the goal embedding if available; otherwise a consistent null context
        (from meta if present, else zeros). This ensures fairness in A vs B when
        no goal is supplied.
        """
        if self.goal_conditioned:
            goal_text = ((context or {}).get("goal") or {}).get("goal_text", "")
            if goal_text:
                return self._safe_embed(goal_text)

        if self._ctx_null is None:
            # Try a meta-provided null context first
            meta_ctx = None
            for m in self.metas.values():
                meta_ctx = m.get("null_context") or None
                if meta_ctx:
                    break
            self._ctx_null = self._safe_embed(meta_ctx) if meta_ctx else np.zeros(
                self.memory.embedding.dim, dtype=np.float32
            )
        return self._ctx_null

    # ---------- scoring API ---------- #

    def _score_core(self, context: dict, scorable: Scorable, dimensions: List[str]) -> ScoreBundle:
        """
        Absolute scoring by baseline comparison:
        For each dimension d:
        - build [ctx||doc] vs [ctx||baseline_d]
        - run model(doc, baseline) → raw logit
        - calibrate via RegressionTuner → tuned_score
        - clamp to [min_value, max_value] from meta

        Returns a ScoreBundle with per-dimension ScoreResult.
        """
        ctx_emb = self._get_context_embedding(context)
        doc_emb = self._safe_embed(scorable.text)

        results: Dict[str, ScoreResult] = {}
        for dim in dimensions:
            if dim not in self.models or dim not in self.tuners or dim not in self.metas:
                self.logger.log("ContrastiveRankerMissingDim", {"dimension": dim})
                continue

            scaler, model = self.models[dim]
            tuner = self.tuners[dim]
            meta = self.metas[dim]
            baseline_emb = self.baselines[dim]

            # Build pairs and tensors
            t_doc = self._prep_pair_tensor(scaler, ctx_emb, doc_emb)
            t_base = self._prep_pair_tensor(scaler, ctx_emb, baseline_emb)

            # Forward pass (doc vs baseline)
            with torch.no_grad():
                raw = float(model(t_doc, t_base).item())

            # Calibrate and clamp
            tuned = float(tuner.transform(raw))
            min_v, max_v = meta["min_value"], meta["max_value"]
            final = float(np.clip(tuned, min_v, max_v))

            attributes = {
                "raw_score": round(raw, 6),
                "normalized_score": round(tuned, 6),
                "final_score": final,
                "baseline_used": bool(np.any(baseline_emb)),
            }

            results[dim] = ScoreResult(
                dimension=dim,
                score=final,
                weight=1.0,
                source=self.model_type,
                rationale=f"PrefScore(raw={raw:.4f}→tuned={tuned:.3f} clamped[{min_v},{max_v}])",
                attributes=attributes,
            )

        return ScoreBundle(results=results)

    def compare(
        self,
        context: Optional[dict],
        a: Scorable,
        b: Scorable,
        dimensions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Native pairwise compare A vs B under the (possibly goal-conditioned) context.

        Returns:
            {
              "winner": "a"|"b",
              "score_a": float,  # mean preference P(A>B) across used dims
              "score_b": float,
              "mode": "pairwise",
              "scorer": "contrastive_ranker",
              "per_dimension": [
                 {"dimension": dim, "p_ab": ..., "p_ba": ..., "pref_a": ..., "raw_ab": ..., "raw_ba": ...},
                 ...
              ]
            }
        """
        ctx_emb = self._get_context_embedding(context)
        a_emb = self._safe_embed(a.text)
        b_emb = self._safe_embed(b.text)

        dims = dimensions or getattr(self, "dimensions", None) or list(self.models.keys())
        per_dim: List[Dict[str, Any]] = []
        pref_sum = 0.0
        used = 0

        for dim in dims:
            if dim not in self.models:
                self.logger.log("ContrastiveRankerMissingDim", {"dimension": dim})
                continue

            scaler, model = self.models[dim]
            model.eval()

            t_a = self._prep_pair_tensor(scaler, ctx_emb, a_emb)
            t_b = self._prep_pair_tensor(scaler, ctx_emb, b_emb)

            with torch.no_grad():
                raw_ab = float(model(t_a, t_b).item())  # logit “A over B”
                raw_ba = float(model(t_b, t_a).item())  # logit “B over A”

            p_ab = self._sigmoid(raw_ab)               # P(A > B)
            p_ba = self._sigmoid(raw_ba)               # P(B > A)
            # symmetric preference (robust to slight antisymmetry violations)
            pref_a = 0.5 * (p_ab + (1.0 - p_ba))

            per_dim.append({
                "dimension": dim,
                "p_ab": round(p_ab, 6),
                "p_ba": round(p_ba, 6),
                "pref_a": round(pref_a, 6),
                "raw_ab": round(raw_ab, 6),
                "raw_ba": round(raw_ba, 6),
            })
            pref_sum += pref_a
            used += 1

        pref_a_mean = float(pref_sum / used) if used else 0.5
        winner = "a" if pref_a_mean >= 0.5 else "b"

        result = {
            "winner": winner,
            "score_a": pref_a_mean,
            "score_b": float(1.0 - pref_a_mean),
            "mode": "pairwise",
            "scorer": self.model_type,
            "per_dimension": per_dim,
        }
        if self.logger:
            self.logger.log("ContrastivePairwiseDecision", result)
        return result
