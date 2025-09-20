import os, torch
from stephanie.data.score_bundle import ScoreBundle
from stephanie.data.score_result import ScoreResult
from stephanie.scoring.scorer.base_scorer import BaseScorer
from stephanie.scoring.transforms.regression_tuner import RegressionTuner
from stephanie.utils.file_utils import load_json
from stephanie.utils.model_locator import ModelLocator
from stephanie.constants import GOAL
from stephanie.scoring.model.knowledge import KnowledgeModel

class KnowledgeScorer(BaseScorer):
    """
    Loads the trained KnowledgeModel and returns a ScoreBundle with 'knowledge'.
    Compatible with your MRQ scoring call-sites.
    """
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.model_type   = "knowledge"
        self.embedding    = memory.embedding
        self.dim          = self.embedding.dim
        self.hdim         = self.embedding.hdim
        self.embedding_type = self.embedding.name
        self.target_type  = cfg.get("target_type", "document")
        self.model_path   = cfg.get("model_path", "models")
        self.version      = cfg.get("model_version", "v1")
        self.aux_features = cfg.get("aux_features", [
            "human_stars", "pseudo_stars", "artifact_quality",
            "turn_pos_ratio", "has_retrieval", "retrieval_fidelity",
            "text_len_norm"
        ])

        self.model = KnowledgeModel(self.dim, self.hdim, self.embedding, self.aux_features, device=str(self.device))
        self.meta  = {"min_value": 0.0, "max_value": 1.0, "aux_features": self.aux_features}
        self.tuner = None
        self._load_weights()

    def _load_weights(self):
        loc = ModelLocator(
            root_dir=self.model_path, embedding_type=self.embedding_type,
            model_type=self.model_type, target_type=self.target_type,
            dimension="knowledge", version=self.version
        )
        # encoder.pt, model.pt, q_head_file() repurposed for aux_proj
        enc_path, pred_path, aux_path = loc.encoder_file(), loc.model_file(), loc.q_head_file()
        self.model.load(enc_path, pred_path, aux_path)

        meta_path = loc.meta_file()
        if os.path.exists(meta_path):
            self.meta = load_json(meta_path)

        tuner_path = loc.tuner_file()
        if os.path.exists(tuner_path):
            t = RegressionTuner(dimension="knowledge")
            t.load(tuner_path)
            self.tuner = t

    def _aux_from_context(self, context: dict, scorable) -> dict:
        m = {}
        turn_meta = getattr(scorable, "meta", {}) or {}
        m.update({
            "human_stars": turn_meta.get("stars", 0.0),
            "pseudo_stars": turn_meta.get("pseudo_stars", 0.0),
            "artifact_quality": turn_meta.get("artifact_quality", 0.0),
            "turn_pos_ratio": turn_meta.get("order_index", 0) / max(1, turn_meta.get("conv_length", 1)),
            "has_retrieval": 1.0 if turn_meta.get("has_retrieval") else 0.0,
            "retrieval_fidelity": turn_meta.get("retrieval_fidelity", 0.0),
            "text_len_norm": min(1.0, len(getattr(scorable, "text", "") or "") / 2000.0),
        })
        return m

    def score(self, context: dict, scorable, dimensions=None) -> ScoreBundle:
        goal = context.get(GOAL, {})
        goal_text = goal.get("goal_text") or context.get("title") or ""
        meta = self._aux_from_context(context, scorable)
        q = self.model.predict(goal_text, scorable.text, meta=meta)
        # Map “diff-style” scalar to [0,1]. Use tuner if present, else sigmoid as a fallback.
        scaled = self.tuner.transform(q) if self.tuner else float(torch.sigmoid(torch.tensor(q)).item())
        lo, hi = self.meta.get("min_value", 0.0), self.meta.get("max_value", 1.0)
        scaled = max(min(scaled, hi), lo)

        res = ScoreResult(
            dimension="knowledge",
            score=round(scaled, 4),
            source="knowledge",
            rationale=f"Q={round(q,4)}",
            weight=1.0,
            attributes={"q_value": round(q,4), "normalized_score": round(scaled,4), "aux_used": self.aux_features}
        )
        return ScoreBundle(results={"knowledge": res})
