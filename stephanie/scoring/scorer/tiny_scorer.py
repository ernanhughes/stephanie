# stephanie/scoring/tiny_scorer.py
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

import torch

from stephanie.constants import GOAL, GOAL_TEXT
from stephanie.data.score_bundle import ScoreBundle
from stephanie.data.score_result import ScoreResult
from stephanie.scoring.model.tiny_recursion import TinyRecursionModel
from stephanie.scoring.scorer.base_scorer import BaseScorer
from stephanie.utils.file_utils import load_json

_logger = logging.getLogger(__name__)


class TinyScorer(BaseScorer):
    """
    Scorer that uses a trained TinyRecursionModel (TRM) to evaluate goal/document pairs.
    Tiny runs a few recursive refinement steps in embedding space and predicts a quality score,
    plus rich auxiliary diagnostics (entropy, certainty/uncertainty, sensitivity, agreement, etc).

    ### SCM: this scorer now also emits a fixed, aligned set of Shared Core Metrics under `scm.*`.
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.model_type = "tiny"  # identifies scorer type in results

        # Embedding interface (shared with HRM)
        self.embedding_type = self.memory.embedding.name
        self.dim = self.memory.embedding.dim

        # Config
        self.target_type = cfg.get("target_type", "conversation_turn")
        self.model_path = cfg.get("model_path", "models")
        self.version = cfg.get("model_version", "v1")
        self.dimensions: List[str] = cfg.get("dimensions", [])

        # Optional output scaling (keep consistent with HRM if desired)
        self.clip_0_100 = cfg.get("clip_0_100", True)

        # Attr verbosity: "minimal" | "standard" | "full"
        self.attr_level = (cfg.get("tiny_attr_level") or "standard").lower()

        # Containers for per-dimension models and metadata
        self.models: Dict[str, TinyRecursionModel] = {}
        self.model_meta: Dict[str, Dict[str, Any]] = {}

        # Attempt to load models up-front
        self._load_models(self.dimensions)

    # -------------------------
    # Loading
    # -------------------------
    def _load_models(self, dimensions: List[str]):
        for dim in dimensions:
            locator = self.get_locator(dim)

            # Resolve files
            model_path = locator.model_file(suffix="_tiny.pt")
            meta_path = locator.meta_file()

            if not os.path.exists(model_path):
                self.logger.log(
                    "TinyScorerModelMissing",
                    {"dimension": dim, "path": model_path},
                )
                continue

            # Pull hyperparams from meta (prefer the “safe” ones if you saved them)
            meta: Dict[str, Any] = {}
            if os.path.exists(meta_path):
                try:
                    meta = load_json(meta_path) or {}
                except Exception as e:
                    self.logger.log(
                        "TinyScorerMetaLoadError", {"dimension": dim, "error": str(e)}
                    )

            # Defaults aligned with trainer
            cfg_meta = meta.get("cfg", {}) if isinstance(meta, dict) else {}
            n_layers = int(cfg_meta.get("n_layers", 2))
            n_recursions = int(cfg_meta.get("n_recursions", 6))
            use_attn = bool(cfg_meta.get("use_attention", False))
            dropout = float(cfg_meta.get("dropout", 0.1))
            attn_heads = int(cfg_meta.get("attn_heads", 4))
            step_scale = float(cfg_meta.get("step_scale", 0.1))
            cons_mask_p = float(cfg_meta.get("consistency_mask_p", 0.10))
            len_norm_L = float(cfg_meta.get("len_norm_L", 512.0))
            vocab_size = int(cfg_meta.get("vocab_size", 101))

            # Allow model flags to be toggled by meta (default True)
            enable_agree_head = bool(cfg_meta.get("enable_agree_head", True))
            enable_causal_sens_head = bool(cfg_meta.get("enable_causal_sens_head", True))

            # Instantiate exactly like trainer did
            model = TinyRecursionModel(
                d_model=self.dim,
                n_layers=n_layers,
                n_recursions=n_recursions,
                vocab_size=vocab_size,
                use_attention=use_attn,
                dropout=dropout,
                attn_heads=attn_heads,
                step_scale=step_scale,
                consistency_mask_p=cons_mask_p,
                len_norm_L=len_norm_L,
                enable_agree_head=enable_agree_head,
                enable_causal_sens_head=enable_causal_sens_head,
            ).to(self.device)

            # Load weights
            state = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state, strict=False)
            model.eval()

            self.models[dim] = model
            self.model_meta[dim] = meta
            self.logger.log(
                "TinyScorerModelLoaded",
                {"dimension": dim, "model_path": model_path, "device": str(self.device)},
            )

    # -------------------------
    # Scoring
    # -------------------------
    def _score_core(self, context: dict, scorable, dimensions: List[str]) -> ScoreBundle:
        goal = context.get(GOAL, {})
        goal_text = goal.get(GOAL_TEXT, "")
        results: Dict[str, ScoreResult] = {}

        # 1) embeddings
        x_np = self.memory.embedding.get_or_create(goal_text)
        y_np = self.memory.embedding.get_or_create(scorable.text)
        x = torch.tensor(x_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        y = torch.tensor(y_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        z = torch.zeros_like(x)  # neutral third stream
        seq_len = torch.zeros(x.size(0), dtype=torch.int32, device=self.device)

        for dim in dimensions:
            model = self.models.get(dim)
            if model is None:
                self.logger.log("TinyModelMissing", {"dimension": dim})
                continue

            try:
                with torch.no_grad():
                    _, halt_logits, _, aux = model(
                        x, y, z, seq_len=seq_len, return_aux=True
                    )

                # Core metrics
                raw01 = float(max(0.0, min(1.0, _tf(aux.get("score")))))
                # "uncertainty" key carries certainty in current model; prefer certainty01 if present.
                # prefer certainty01; fall back to (1 - uncertainty) if that’s present; else 0.5
                if "certainty01" in aux:
                    certainty01 = _tf(aux["certainty01"])
                elif "uncertainty01" in aux:
                    certainty01 = 1.0 - _tf(aux["uncertainty01"])
                elif "uncertainty" in aux:
                    certainty01 = 1.0 - _tf(aux["uncertainty"])
                else:
                    certainty01 = 0.5
                entropy = _tf(aux.get("entropy_aux"))
                halt_prob = _sigmoid_mean(halt_logits)

                # Meta & scaling
                meta = self.model_meta.get(dim, {})
                final_score = _tf(aux.get("score"))
                tiny_score01 = raw01
                tiny_score100 = round(_safe_scale_0_100(tiny_score01, meta), 4)

                # Attributes (base set)
                attrs: Dict[str, Any] = {
                    "tiny.score01": tiny_score01,
                    "tiny.score100": tiny_score100,
                    "raw01": tiny_score01,  # backward-compat shortcut if others read it
                    "entropy": float(entropy),
                    "certainty01": float(certainty01),
                    "halt_prob": float(halt_prob) if halt_prob is not None else None,
                    # Useful meta (so downstream knows Tiny’s context)
                    "n_recursions": int(meta.get("cfg", {}).get("n_recursions", 6)),
                    "use_attention": bool(meta.get("cfg", {}).get("use_attention", False)),
                    "dropout": float(meta.get("cfg", {}).get("dropout", 0.1)),
                }

                # Standard / Full diagnostics
                if self.attr_level in ("standard", "full"):
                    attrs.update(_extract_standard_aux(aux))
                    # Include bridge heads if present
                    if "agree01" in aux and isinstance(aux["agree01"], torch.Tensor):
                        attrs["agree01"] = float(_tf(aux["agree01"]))
                    if "sens01" in aux and isinstance(aux["sens01"], torch.Tensor):
                        attrs["sens01"] = float(_tf(aux["sens01"]))

                if self.attr_level == "full":
                    attrs.update(_extract_full_aux(aux))
                    # include benign summaries of logits for debugging
                    if "score_logit" in aux:
                        attrs["score_logit_mean"] = float(_tf(aux["score_logit"]))
                    if "aux3_logits" in aux and isinstance(aux["aux3_logits"], torch.Tensor):
                        al = aux["aux3_logits"]
                        attrs["aux3_logits_l1_mean"] = float(al.abs().mean().item())

                # === SCM: build shared-core metrics (fixed, aligned columns) ===
                scm = _build_scm_from_tiny_attrs(attrs)
                attrs.update(scm)

                # Optional: duplicate Tier-1 dimension scores under tiny.* for PHOS
                # (This guarantees rows_for_df has hrm./tiny. columns even if native metrics differ.)
                for dname in ("reasoning", "knowledge", "clarity", "faithfulness", "coverage"):
                    key = f"scm.{dname}.score01"
                    if key in scm:
                        v01 = float(scm[key])
                        attrs[f"tiny.{dname}.score01"]  = v01
                        attrs[f"tiny.{dname}.score100"] = round(v01 * 100.0, 4)
                        attrs[f"tiny.{dname}"] = float(scm[key])

                rationale = (
                    f"tiny[{dim}] raw01={float(raw01):.4f}, "
                    f"H={float(entropy):.3f}, C={float(certainty01):.3f}, "
                    f"halt_p={float(halt_prob) if halt_prob is not None else -1:.3f}"
                )

                # Build a flat vector so GAP can align columns deterministically
                vector = _tiny_build_vector(attrs)

                results[dim] = ScoreResult(
                    dimension=dim,
                    score=tiny_score01,
                    source=self.model_type,
                    rationale=rationale,
                    weight=1.0,
                    attributes={**attrs, "vector": vector["vector"], "columns": vector["columns"], "values": vector["values"]},
                )

            except Exception as e:
                self.logger.log("TinyScoreError", {"dimension": dim, "error": str(e)})

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

def _take_scalar(t):
    # works with tensor or float
    if isinstance(t, torch.Tensor):
        return float(t.detach().mean().cpu().item())
    return float(t)


# -------------------------
# Helpers
# -------------------------

def _tf(v):
    """Tensor/array/number → scalar float (mean) with safe fallback."""
    if v is None:
        return 0.0
    if isinstance(v, torch.Tensor):
        # handle both scalar and vector tensors
        return v.detach().float().mean().item()
    try:
        return float(v)
    except Exception:
        return 0.0


def _sigmoid_mean(v):
    if v is None:
        return None
    if isinstance(v, torch.Tensor):
        return torch.sigmoid(v.detach()).mean().item()
    return float(v)


def _safe_scale_0_100(raw: float, meta: dict | None) -> float:
    # raw in [0,1] (trainer learned target01). Scale to meta range or 0..100.
    if not meta:
        return float(max(0.0, min(1.0, raw)) * 100.0)
    lo = float(meta.get("min_value", 0.0))
    hi = float(meta.get("max_value", 100.0))
    return float(max(lo, min(hi, lo + (hi - lo) * max(0.0, min(1.0, raw)))))

def _tiny_build_vector(attrs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Produce vector/columns/values for ScoringProcessor alignment.
    Includes tiny raw stats + SCM + mirrored tiny.{dim} scores.
    """
    vec: Dict[str, float] = {}
    # a few raw-ish tiny stats that show up often
    vec["tiny.score01"]        = float(attrs.get("tiny.score01", 0.0))
    vec["tiny.score100"]       = float(attrs.get("tiny.score100", 0.0))
    vec["tiny.certainty01"]    = float(attrs.get("certainty01", 0.5))
    vec["tiny.entropy_mean"]   = float(attrs.get("entropy", 0.0))
    if "halt_prob" in attrs and attrs["halt_prob"] is not None:
        vec["tiny.halt_prob"] = float(attrs["halt_prob"])

    # SCM (fixed schema)
    for k in [
        "scm.reasoning.score01","scm.knowledge.score01","scm.clarity.score01",
        "scm.faithfulness.score01","scm.coverage.score01","scm.aggregate01",
        "scm.uncertainty01","scm.ood_hat01","scm.consistency01",
        "scm.length_norm01","scm.temp01","scm.agree_hat01",
    ]:
        if k in attrs:
            vec[k] = float(attrs[k])

    # mirror five dimensions under tiny.* for PHOS
    for d in ("reasoning","knowledge","clarity","faithfulness","coverage"):
        k = f"scm.{d}.score01"
        if k in attrs:
            v01 = float(attrs[k])
            vec[f"tiny.{d}.score01"]  = v01
            vec[f"tiny.{d}.score100"] = round(v01 * 100.0, 4)
            vec[f"tiny.{d}"]          = v01

    cols = list(vec.keys())
    vals = [vec[c] for c in cols]
    return {"vector": vec, "columns": cols, "values": vals}


def _extract_standard_aux(aux: Dict[str, Any]) -> Dict[str, float]:
    """
    Balanced diagnostics, all in [0,1] where sensible.
    """
    out: Dict[str, float] = {}

    # confidence triplet
    if "aux3_probs" in aux and isinstance(aux["aux3_probs"], torch.Tensor):
        p = aux["aux3_probs"].detach().float()
        out["aux3_p_bad"]  = float(p[..., 0].mean().item())
        out["aux3_p_mid"]  = float(p[..., 1].mean().item())
        out["aux3_p_good"] = float(p[..., 2].mean().item())

    # calibration / OOD
    out["temp01"] = float(_tf(aux.get("temp01")))

    # OOD (prefer ood_hat01)
    if "ood_hat01" in aux:
        out["ood_hat01"] = float(_tf(aux["ood_hat01"]))
    elif "ood_hat" in aux:  # backward compat
        out["ood_hat01"] = float(_tf(aux["ood_hat"]))

    # Robustness & sensitivity
    out["consistency_hat"] = float(_tf(aux.get("consistency_hat")))
    out["jacobian_fd"]     = float(_tf(aux.get("jacobian_fd")))

    # Reconstruction / comprehension
    out["recon_sim"]  = float(_tf(aux.get("recon_sim")))
    out["disagree_hat"] = float(_tf(aux.get("disagree_hat")))

    # Length proxy (prefer 0..1)
    if "length_norm01" in aux:
        out["length_norm01"] = float(_tf(aux["length_norm01"]))
    else:
        # derive from tanh-normalized len_effect if present
        if "len_effect" in aux:
            le = float(_tf(aux["len_effect"]))
            out["length_norm01"] = float(max(0.0, min(1.0, (le + 1.0) * 0.5)))
        else:
            out["length_norm01"] = 0.0

    # SAE sparsity
    out["concept_sparsity"] = float(_tf(aux.get("concept_sparsity")))
    return out


def _extract_full_aux(aux: Dict[str, Any]) -> Dict[str, float]:
    """
    Max detail: expose raw-ish signals but summarized safely.
    Avoid huge vectors; keep scalar summaries.
    """
    out: Dict[str, float] = {}

    # Summaries of raw heads
    for k in ("log_var", "consistency_logit", "disagree_logit"):
        if k in aux and isinstance(aux[k], torch.Tensor):
            t = aux[k].detach()
            out[f"{k}_mean"] = float(t.mean().item())

    # Reconstructive detail
    if "y_recon" in aux and isinstance(aux["y_recon"], torch.Tensor):
        yr = aux["y_recon"].detach()
        out["y_recon_norm_mean"] = float(yr.norm(dim=-1).mean().item())

    # Concept vector magnitude (post-SAE)
    if "concept_vec" in aux and isinstance(aux["concept_vec"], torch.Tensor):
        c = aux["concept_vec"].detach()
        out["concept_vec_l2_mean"] = float((c.pow(2).sum(-1).sqrt()).mean().item())
    return out


# === SCM mapping from Tiny aux → aligned scm.* columns =======================

def _build_scm_from_tiny_attrs(attrs: Dict[str, Any]) -> Dict[str, float]:
    """Dimension-specific SCM mapping for Tiny using model dynamics only."""
    # Core signals (clamped)
    certainty = float(attrs.get("certainty01", 0.5))
    unc01     = 1.0 - max(0.0, min(1.0, certainty))
    cons01    = max(0.0, min(1.0, float(attrs.get("consistency_hat", 0.5))))
    ood01     = max(0.0, min(1.0, float(attrs.get("ood_hat", 0.0))))
    len01     = max(0.0, min(1.0, float(attrs.get("len_effect", 0.0))))
    temp01    = max(0.0, min(1.0, float(attrs.get("temp01", 0.0))))
    agree01   = max(0.0, min(1.0, float(attrs.get("agree01", 0.5))))

    # Extra signals
    recon_sim      = max(0.0, min(1.0, float(attrs.get("recon_sim", 0.5))))
    concept_sparse = max(0.0, min(1.0, float(attrs.get("concept_sparsity", 0.5))))
    p_bad          = max(0.0, min(1.0, float(attrs.get("aux3_p_bad", 0.5))))
    token_ok       = 1.0 - p_bad  # “clarity-ish” proxy: lower bad prob → clearer

    dim_scores: Dict[str, float] = {}
    # Reasoning: stability/consistency + low uncertainty + agreement
    dim_scores["reasoning"] = 0.60*cons01 + 0.30*(1.0-unc01) + 0.10*agree01
    # Knowledge: in-distribution + reconstruction + low uncertainty
    dim_scores["knowledge"] = 0.50*(1.0-ood01) + 0.30*recon_sim + 0.20*(1.0-unc01)
    # Clarity: token “goodness” + shorter/normalized length + consistency
    dim_scores["clarity"] = 0.50*token_ok + 0.30*(1.0-len01) + 0.20*cons01
    # Faithfulness: reconstruction + consistency + low uncertainty
    dim_scores["faithfulness"] = 0.50*recon_sim + 0.30*cons01 + 0.20*(1.0-unc01)
    # Coverage: concept activity + low uncertainty + in-distribution
    dim_scores["coverage"] = 0.40*concept_sparse + 0.40*(1.0-unc01) + 0.20*(1.0-ood01)

    # Clamp to [0,1]
    for k in dim_scores:
        v = dim_scores[k]
        dim_scores[k] = float(min(1.0, max(0.0, v)))

    scm: Dict[str, float] = {f"scm.{k}.score01": dim_scores[k]
                             for k in ("reasoning","knowledge","clarity","faithfulness","coverage")}
    scm["scm.aggregate01"]   = float(sum(dim_scores.values())/5.0)
    scm["scm.uncertainty01"] = float(unc01)
    scm["scm.ood_hat01"]     = float(ood01)
    scm["scm.consistency01"] = float(cons01)
    scm["scm.length_norm01"] = float(len01)
    scm["scm.temp01"]        = float(temp01)
    scm["scm.agree_hat01"]   = float(agree01)
    return scm
