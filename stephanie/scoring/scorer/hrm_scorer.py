# stephanie/scoring/hrm_scorer.py

import logging
import os
from typing import Any, Dict, List

import torch

from stephanie.constants import GOAL, GOAL_TEXT
from stephanie.data.score_bundle import ScoreBundle
from stephanie.data.score_result import ScoreResult
from stephanie.scoring.model.hrm_model import HRMModel
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorer.base_scorer import BaseScorer
from stephanie.utils.file_utils import load_json  # To load meta file

_logger = logging.getLogger(__name__)

class HRMScorer(BaseScorer):
    """
    Scorer that uses a trained Hierarchical Reasoning Model (HRM) to evaluate
    goal/document pairs. The HRM performs internal multi-step reasoning to
    produce a quality score.

    This scorer also emits the fixed Shared Core Metrics (SCM) under `scm.*`
    so GAP analysis always has aligned columns vs Tiny.
    """
    def __init__(self, cfg, memory, container, logger, enable_plugins: bool = True):
        super().__init__(cfg, memory, container, logger, enable_plugins=enable_plugins)
        self.model_type = "hrm"  # identifies scorer type

        # Embedding interface (shared with Tiny)
        self.embedding_type = self.memory.embedding.name
        self.dim = self.memory.embedding.dim

        # Config
        self.target_type = cfg.get("target_type", "conversation_turn")
        self.model_path = cfg.get("model_path", "models")
        self.version = cfg.get("model_version", "v1")
        self.dimensions: List[str] = cfg.get("dimensions", [])

        # Per-dimension models + meta
        self.models: Dict[str, HRMModel] = {}
        self.model_meta: Dict[str, Dict[str, Any]] = {}

        # Load models up-front
        self._load_models(self.dimensions)

    # -------------------------------------------------------------------------
    # Loading
    # -------------------------------------------------------------------------
    def _load_models(self, dimensions: List[str]):
        """
        Loads trained HRM models + metadata using ModelLocator.
        """
        for dimension in dimensions:
            try:
                locator = self.get_locator(dimension)

                model_file_path = locator.model_file(suffix="_hrm.pt")
                meta_file_path = locator.meta_file()

                if not os.path.exists(model_file_path):
                    self.logger.log("HRMScorerModelError", {
                        "message": "HRM model file not found.",
                        "path": model_file_path,
                        "dimension": dimension,
                    })
                    continue

                # Load meta (optional)
                meta: Dict[str, Any] = {}
                if os.path.exists(meta_file_path):
                    try:
                        meta = load_json(meta_file_path) or {}
                        self.logger.log("HRMScorerMetaLoaded", {
                            "dimension": dimension, "has_meta": True
                        })
                    except Exception as e:
                        self.logger.log("HRMScorerMetaLoadError", {
                            "dimension": dimension, "error": str(e)
                        })
                else:
                    self.logger.log("HRMScorerWarning", {
                        "message": "HRM meta file not found. Using defaults.",
                        "path": meta_file_path,
                        "dimension": dimension
                    })

                self.model_meta[dimension] = meta

                # Reconstruct HRM config consistent with training defaults
                cfg_meta = meta.get("cfg", {}) if isinstance(meta, dict) else {}
                hrm_cfg = {
                    "input_dim": meta.get("input_dim", self.dim * 2),
                    "h_dim": meta.get("h_dim", 256),
                    "l_dim": meta.get("l_dim", 128),
                    "output_dim": meta.get("output_dim", 1),
                    "n_cycles": meta.get("n_cycles", cfg_meta.get("n_cycles", 4)),
                    "t_steps": meta.get("t_steps", cfg_meta.get("t_steps", 4)),
                }

                model = HRMModel(hrm_cfg, logger=self.logger).to(self.device)
                state = torch.load(model_file_path, map_location=self.device)
                model.load_state_dict(state, strict=False)
                model.eval()

                self.models[dimension] = model

                self.logger.log("HRMScorerModelLoaded", {
                    "dimension": dimension,
                    "model_path": model_file_path,
                    "device": str(self.device),
                })

            except Exception as e:
                self.logger.log("HRMScorerInitError", {
                    "message": "Failed to load HRM model.",
                    "dimension": dimension,
                    "model_type": self.model_type,
                    "error": str(e),
                })

    # -------------------------------------------------------------------------
    # Scoring
    # -------------------------------------------------------------------------
    def score(self, context: dict, scorable: Scorable, dimensions: List[str]) -> ScoreBundle:
        """
        Scores a single scorable item against a goal using the HRM models per dimension.
        Returns: ScoreBundle with ScoreResult for each dimension.
        """
        results: Dict[str, ScoreResult] = {}

        goal = context.get(GOAL, {})
        goal_text = goal.get(GOAL_TEXT, "")
        doc_text = scorable.text

        if not goal_text or not doc_text:
            self.logger.log("HRMScorerWarning", {
                "message": "Missing goal_text or scorable text.",
                "goal": goal.get("id", "unknown"),
                "scorable_id": scorable.id
            })
            return ScoreBundle(results={})

        # Embeddings
        ctx_emb_np = self.memory.embedding.get_or_create(goal_text)
        doc_emb_np = self.memory.embedding.get_or_create(doc_text)
        ctx_emb = torch.tensor(ctx_emb_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        doc_emb = torch.tensor(doc_emb_np, dtype=torch.float32, device=self.device).unsqueeze(0)

        # HRM input: concat (x, y) — keep consistent with training
        x_input = torch.cat([ctx_emb, doc_emb], dim=-1)

        for dimension in dimensions:
            model = self.models.get(dimension)
            if not model:
                self.logger.log("HRMScorerError", {
                    "message": f"HRM model not found for dimension '{dimension}'. Skipping.",
                    "goal_id": goal.get("id", "unknown"),
                    "scorable_id": scorable.id
                })
                continue

            try:
                with torch.no_grad():
                    # HRM forward returns (y_pred, intermediate_states)
                    y_pred, intermediate = model(x_input)

                raw_score = float(y_pred.squeeze().item())

                # Pull a few useful magnitudes if present (robust to None / non-tensors)
                zL_mag = _safe_norm(intermediate.get("zL_final"))
                zH_mag = _safe_norm(intermediate.get("zH_final"))

                rationale = (
                    f"HRM[{dimension}] raw={raw_score:.4f} | "
                    f"zL_mag={_fmt_opt4(zL_mag)}, zH_mag={_fmt_opt4(zH_mag)}"
                )

                # Native attrs (kept for drill-down)
                attributes: Dict[str, Any] = {
                    "raw_score": raw_score,
                    "zL_magnitude": zL_mag,
                    "zH_magnitude": zH_mag,
                    # historical keys used in your blog glossary
                    "q_value": raw_score,
                    "energy": raw_score,  # swap out later if you add real energy
                }

                # Optionally proxy extra HRM diagnostics if your HRMModel exposes them
                for k in ("entropy", "len_effect", "ood_hat", "consistency_hat", "temp01", "disagree_hat", "agree_hat"):
                    if k in intermediate:
                        v = intermediate[k]
                        attributes[k] = float(v.detach().mean().item() if isinstance(v, torch.Tensor) else float(v))

                # === SCM: derive shared-core metrics from HRM internals/attrs ===
                scm = _build_scm_from_hrm(attrs=attributes, intermediate=intermediate)
                attributes.update(scm)

                # Mirror the five core scores under hrm.* for PHOS rows
                for dname in ("reasoning", "knowledge", "clarity", "faithfulness", "coverage"):
                    key = f"scm.{dname}.score01"
                    if key in scm:
                        attributes[f"hrm.{dname}"] = float(scm[key])

                results[dimension] = ScoreResult(
                    dimension=dimension,
                    score=raw_score,
                    source=self.model_type,
                    rationale=rationale,
                    weight=1.0,
                    attributes=attributes,
                )
                _logger.debug(
                    "HRMScorerEvaluated "
                    f"dimension={dimension} goal_id={goal.get('id', 'unknown')} "
                    f"scorable_id={scorable.id} raw_score={raw_score} "
                    f"zL_magnitude={zL_mag} zH_magnitude={zH_mag}"
                )

            except Exception as e:
                self.logger.log("HRMScorerError", {
                    "message": "Exception during HRM scoring.",
                    "dimension": dimension,
                    "goal_id": goal.get("id", "unknown"),
                    "scorable_id": scorable.id,
                    "error": str(e)
                })

        return ScoreBundle(results=results)

    def __repr__(self):
        loaded = {k: (v is not None) for k, v in self.models.items()}
        return f"<HRMScorer(model_type={self.model_type}, loaded={loaded})>"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _fmt_opt4(x):
    return "NA" if x is None else f"{float(x):.4f}"

def _safe_norm(t):
    if t is None:
        return None
    if isinstance(t, torch.Tensor):
        try:
            return float(torch.norm(t.detach().float(), p=2).item())
        except Exception:
            return None
    try:
        return float(t)
    except Exception:
        return None


# === SCM mapping from HRM internals → aligned scm.* columns ==================

_SCM_DIMS = ("reasoning", "knowledge", "clarity", "faithfulness", "coverage")

def _clip01(x: float) -> float:
    try:
        return float(min(1.0, max(0.0, x)))
    except Exception:
        return 0.0

def _build_scm_from_hrm(*, attrs: Dict[str, Any], intermediate: Dict[str, Any]) -> Dict[str, float]:
    """Dimension-specific SCM mapping for HRM using internal signals only."""
    def _clip01(x: float) -> float:
        try: return float(min(1.0, max(0.0, x)))
        except Exception: return 0.0

    # Uncertainty: prefer entropy→log_var→energy
    if "entropy" in attrs:
        unc = _clip01(float(attrs["entropy"]))
    elif "log_var" in intermediate and isinstance(intermediate["log_var"], torch.Tensor):
        unc = _clip01(float(torch.sigmoid(intermediate["log_var"].detach()).mean().item()))
    elif "energy" in attrs:
        unc = _clip01(float(attrs["energy"]) / 100.0)
    else:
        unc = 0.5

    # Consistency: prefer consistency_hat; else zL/zH cosine proxy
    if "consistency_hat" in attrs:
        cons = _clip01(float(attrs["consistency_hat"]))
    elif "zL_final" in intermediate and "zH_final" in intermediate:
        zL, zH = intermediate["zL_final"], intermediate["zH_final"]
        if isinstance(zL, torch.Tensor) and isinstance(zH, torch.Tensor):
            try:
                cos = torch.nn.functional.cosine_similarity(
                    zL.detach().flatten(1), zH.detach().flatten(1), dim=-1
                ).mean().item()
                cons = _clip01(0.5*(cos+1.0))
            except Exception:
                cons = 0.5
        else:
            cons = 0.5
    else:
        cons = 0.5

    ood     = _clip01(float(attrs.get("ood_hat", 0.0)))
    length  = _clip01(float(attrs.get("len_effect", 0.0)))
    temp01  = _clip01(float(attrs.get("temp01", 0.0)))
    agree   = _clip01(float(attrs.get("agree_hat", 1.0 - float(attrs.get("disagree_hat", 0.5)))))

    # Extra HRM signals
    recon_sim = _clip01(float(attrs.get("recon_sim", 0.5)))
    zL_mag    = float(attrs.get("zL_magnitude", 0.0) or 0.0)
    zH_mag    = float(attrs.get("zH_magnitude", 0.0) or 0.0)
    # Reasoning depth proxy: high-level vs low-level magnitude
    reasoning_ratio = 0.0 if zL_mag <= 0 else min(1.0, (zH_mag/(zL_mag+1e-6))/10.0)

    dim_scores: Dict[str, float] = {}
    dim_scores["reasoning"]    = 0.50*reasoning_ratio + 0.30*cons + 0.20*(1.0-unc)
    dim_scores["knowledge"]    = 0.50*(1.0-ood) + 0.30*recon_sim + 0.20*(1.0-unc)
    dim_scores["clarity"]      = 0.50*(1.0-length) + 0.30*cons + 0.20*(1.0-unc)
    dim_scores["faithfulness"] = 0.50*recon_sim + 0.30*cons + 0.20*(1.0-unc)
    dim_scores["coverage"]     = 0.40*reasoning_ratio + 0.40*(1.0-unc) + 0.20*(1.0-ood)

    for k in dim_scores:
        dim_scores[k] = _clip01(dim_scores[k])

    scm: Dict[str, float] = {f"scm.{k}.score01": dim_scores[k]
                             for k in ("reasoning","knowledge","clarity","faithfulness","coverage")}
    scm["scm.aggregate01"]   = float(sum(dim_scores.values())/5.0)
    scm["scm.uncertainty01"] = float(unc)
    scm["scm.ood_hat01"]     = float(ood)
    scm["scm.consistency01"] = float(cons)
    scm["scm.length_norm01"] = float(length)
    scm["scm.temp01"]        = float(temp01)
    scm["scm.agree_hat01"]   = float(agree)
    return scm
