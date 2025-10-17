# stephanie/scoring/scorer/knowledge_scorer.py
from __future__ import annotations

import os

from stephanie.constants import GOAL
from stephanie.data.score_bundle import ScoreBundle
from stephanie.data.score_result import ScoreResult
from stephanie.scoring.model.knowledge import KnowledgeModel
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorer.base_scorer import BaseScorer
from stephanie.scoring.transforms.regression_tuner import RegressionTuner
from stephanie.utils.file_utils import load_json
from stephanie.utils.model_locator import ModelLocator


class KnowledgeScorer(BaseScorer):
    """
    Loads the trained KnowledgeModel and returns a ScoreBundle with 'knowledge'.
    Compatible with MRQ-style call sites.
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.model_type = "knowledge"
        self.dim = self.memory.embedding.dim
        self.hdim = self.memory.embedding.hdim
        self.embedding_type = self.memory.embedding.name
        self.target_type = cfg.get("target_type", "document")
        self.model_root = cfg.get("model_path", "models")
        self.version = cfg.get("model_version", "v1")

        self.aux_features = cfg.get(
            "aux_features",
            [
                "human_stars",
                "pseudo_stars",
                "artifact_quality",
                "turn_pos_ratio",
                "has_retrieval",
                "retrieval_fidelity",
                "text_len_norm",
            ],
        )

        # Create model shell (weights loaded in _load_weights)
        self.model = KnowledgeModel(
            dim=self.dim,
            hdim=self.hdim,
            embedding_store=self.memory.embedding,   # ← use memory.embedding
            aux_feature_names=self.aux_features,
            device=str(self.device),
        )

        self.meta = {"min_value": 0.0, "max_value": 1.0, "aux_features": self.aux_features}
        self.tuner = None
        self._load_weights()

    def _load_weights(self):
        loc = ModelLocator(
            root_dir=self.model_root,
            embedding_type=self.embedding_type,
            model_type=self.model_type,
            target_type=self.target_type,
            dimension="knowledge",
            version=self.version,
        )

        # Preferred: explicit paths (matches your trainer’s save)
        enc_path   = loc.encoder_file()
        head_h     = loc.model_file()      # human head
        head_a     = loc.q_head_file()     # AI head
        auxproj    = loc.auxproj_file()

        # Load model weights
        self.model = KnowledgeModel.load(
            dim=self.dim,
            hdim=self.hdim,
            embedding_store=self.memory.embedding,
            aux_feature_names=self.aux_features,
            device=str(self.device),
            encoder_path=enc_path,
            head_h_path=head_h,
            head_a_path=head_a,
            auxproj_path=auxproj,
        )

        # Load meta/manifest if present
        meta_path = loc.meta_file()
        if os.path.exists(meta_path):
            self.meta = load_json(meta_path)

        # Optional tuner (keep only if you trained a tuner on probabilities; else omit)
        tuner_path = loc.tuner_file()
        if os.path.exists(tuner_path):
            t = RegressionTuner(dimension="knowledge")
            t.load(tuner_path)
            self.tuner = t

    def _aux_from_context(self, context: dict, scorable) -> dict:
        tm = getattr(scorable, "meta", {}) or {}
        return {
            "human_stars": tm.get("stars", 0.0),
            "pseudo_stars": tm.get("pseudo_stars", 0.0),
            "artifact_quality": tm.get("artifact_quality", 0.0),
            "turn_pos_ratio": tm.get("order_index", 0) / max(1, tm.get("conv_length", 1)),
            "has_retrieval": 1.0 if tm.get("has_retrieval") else 0.0,
            "retrieval_fidelity": tm.get("retrieval_fidelity", 0.0),
            "text_len_norm": min(1.0, len(getattr(scorable, "text", "") or "") / 2000.0),
            # If you propagate this from your pair builder, blending will prefer human head:
            # "has_similar_human": bool(tm.get("has_similar_human", False)),
        }

    def score(self, context: dict, scorable: Scorable, dimensions=["knowledge"]) -> ScoreBundle:
        goal_text = context.get(GOAL, {}).get("goal_text") or ""
        candidate = getattr(scorable, "text", "") or ""

        if not goal_text or not candidate:
            neutral = 0.5
            res = ScoreResult(
                dimension="knowledge",
                score=neutral,
                source="knowledge",
                rationale="missing_goal_or_text",
                weight=1.0,
                attributes={"normalized_score": neutral, "aux_used": self.aux_features},
            )
            return ScoreBundle(results={"knowledge": res})

        meta = self._aux_from_context(context, scorable)

        # One model call gives you the blended prob + attribution
        p, comps = self.model.predict(
            goal_text,
            candidate,
            meta=meta,
            return_components=True,
        )

        # Optional post-calibration (only if your tuner is prob-calibrated)
        if self.tuner is not None and hasattr(self.tuner, "transform_prob"):
            p = float(self.tuner.transform_prob(float(p)))

        # Clamp
        lo, hi = self.meta.get("min_value", 0.0), self.meta.get("max_value", 1.0)
        p = max(lo, min(hi, float(p)))

        # High-disagreement routing (use model-provided head probs)
        head_gap = abs(comps["human_prob"] - comps["ai_prob"])
        if head_gap > 0.25:
            try:
                self.memory.casebooks.add_scorable(
                    case_id=getattr(scorable, "case_id", None),
                    pipeline_run_id=0,
                    role="uncertain_candidate",
                    text=getattr(scorable, "text", ""),
                    meta={
                        "agent_name": "knowledge_scorer",
                        "section_name": getattr(scorable, "section_name", None),
                        "paper_id": getattr(scorable, "paper_id", None),
                        "human_logit": comps["human_logit"],
                        "ai_logit": comps["ai_logit"],
                        "sigmoid_gap": head_gap,
                    },
                )
            except Exception:
                pass

        # Build result (pass through the model’s diagnostics)
        attrs = {
            "probability": round(p, 4),
            "human_prob": round(comps["human_prob"], 4),
            "ai_prob": round(comps["ai_prob"], 4),
            "human_logit": round(comps["human_logit"], 4),
            "ai_logit": round(comps["ai_logit"], 4),
            "head_gap": round(head_gap, 4),
            "alpha_human_weight": round(comps["alpha_human_weight"], 4),
            "has_similar_human": bool(comps["has_similar_human"]),
            "human_component": round(comps["human_component"], 4),
            "ai_component": round(comps["ai_component"], 4),
            "human_fraction": round(comps["human_fraction"], 4),
            "ai_fraction": round(comps["ai_fraction"], 4),
            "aux_used": list(self.aux_features),
        }

        res = ScoreResult(
            dimension="knowledge",
            score=round(p, 4),
            source="knowledge",
            rationale=f"blended_prob={round(p, 4)}",
            weight=1.0,
            attributes=attrs,
        )
        return ScoreBundle(results={"knowledge": res})


    def _log_disagreement(self, scorable, s_h, s_a, gap):
        """Log high-disagreement cases for human review."""
        try:
            self.memory.casebooks.add_scorable(
                case_id=getattr(scorable, "case_id", None),
                pipeline_run_id=0,
                role="uncertain_candidate",
                text=getattr(scorable, "text", ""),
                meta={
                    "agent_name": "knowledge_scorer",
                    "section_name": getattr(scorable, "section_name", None),
                    "paper_id": getattr(scorable, "paper_id", None),
                    "human_logit": s_h,
                    "ai_logit": s_a,
                    "sigmoid_gap": gap,
                },
            )
        except Exception:
            pass
