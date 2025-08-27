# stephanie/ranking/scorable_ranker.py
import math
import time
from typing import Dict, List, Any
from sklearn.metrics.pairwise import cosine_similarity

from stephanie.scoring.base_scorer import BaseScorer
from stephanie.models.scorable_embedding import ScorableEmbeddingORM
from stephanie.models.evaluation import EvaluationORM
from stephanie.scoring.scorable import Scorable


class ScorableRanker(BaseScorer):
    """
    Universal ranking engine for all scoreables (documents, prompts, plan traces, etc.)
    Works like SICQLScorer but produces a ranking score instead of a direct prediction.
    """

    def __init__(self, cfg, memory, logger):
        super().__init__(cfg, memory, logger)
        self.model_type = "scorable_rank"

        # Embedding backend details
        self.embedding_type = memory.embedding.type
        self.dim = memory.embedding.dim 
        self.hdim = memory.embedding.hdim

        # Config
        self.target_type = cfg.get("target_type", "document")
        self.weights = cfg.get("weights", {
            "similarity": 0.4,
            "reward": 0.3,
            "recency": 0.2,
            "adaptability": 0.1,
        })

        self.logger.log("ScorableRankerInitialized", {
            "target_type": self.target_type,
            "embedding_type": self.embedding_type,
            "weights": self.weights
        })

    # --- Component Scorers ---
    def _similarity(self, scorable: Scorable, query_emb: List[float]) -> float:
        emb = (
            self.memory.session.query(ScorableEmbeddingORM)
            .filter_by(document_id=scorable.id, document_type=scorable.target_type)
            .first()
        )
        if not emb:
            return 0.0
        scorable_emb = self.memory.embedding.get_by_id(emb.embedding_id)
        return float(cosine_similarity([query_emb], [scorable_emb])[0][0])

    def _reward(self, scorable: Scorable) -> float:
        eval_rec = (
            self.memory.session.query(EvaluationORM)
            .filter_by(target_id=scorable.id, target_type=scorable.target_type)
            .order_by(EvaluationORM.created_at.desc())
            .first()
        )
        return float(eval_rec.score) if eval_rec else 0.0

    def _recency(self, scorable: Scorable) -> float:
        created_at = getattr(scorable, "created_at", None)
        if not created_at:
            return 0.5
        age_sec = time.time() - created_at.timestamp()
        return math.exp(-age_sec / (60 * 60 * 24 * 30))  # 30-day decay

    def _adaptability(self, scorable: Scorable) -> float:
        meta = getattr(scorable, "meta", {})
        return float(meta.get("reuse_count", 0)) / float(meta.get("attempt_count", 1))

    # --- Main Ranker ---
    def score(self, query_text: str, scoreables: List[Scorable]) -> List[Dict[str, Any]]:
        """
        Return ranked list of scoreables for a given query text.
        """
        query_emb = self.memory.embedding.get_or_create(query_text)

        ranked = []
        for sc in scoreables:
            sim = self._similarity(sc, query_emb)
            reward = self._reward(sc)
            recency = self._recency(sc)
            adaptability = self._adaptability(sc)

            score = (
                self.weights["similarity"] * sim +
                self.weights["reward"] * reward +
                self.weights["recency"] * recency +
                self.weights["adaptability"] * adaptability
            )

            ranked.append({
                "scorable": sc,
                "rank": score,
                "components": {
                    "similarity": sim,
                    "reward": reward,
                    "recency": recency,
                    "adaptability": adaptability,
                }
            })

        ranked = sorted(ranked, key=lambda x: x["rank"], reverse=True)

        self.logger.log("ScorableRankCompleted", {
            "query": query_text[:80],
            "n_scoreables": len(scoreables),
            "top_rank": ranked[0]["rank"] if ranked else None
        })

        return ranked
