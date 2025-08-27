# stephanie/ranking/scorable_ranker.py
import math
import time
from typing import Dict, List
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

    def score(
        self,
        query: str,
        candidates: List[Scorable],
        extra_signals: Dict[str, float] = None,
    ) -> List[Dict]:
        """
        Rank candidates for a query and log results in DB.
        :param query: the task / state / goal text
        :param candidates: list of Scorable objects
        :param extra_signals: optional dict of {signal_name: weight}
        """
        extra_signals = extra_signals or {}

        query_emb = self.memory.embedding.get_or_create(query)

        results = []
        for cand in candidates:
            cand_emb = self.memory.embedding.get_or_create(cand.text)

            sim = float(cosine_similarity([query_emb], [cand_emb])[0][0])
            rew = self._reward(cand)
            rec = self._recency(cand)
            adapt = self._adaptability(cand)

            score = (
                self.weights["similarity"] * sim
                + self.weights["reward"] * rew
                + self.weights["recency"] * rec
                + self.weights["adaptability"] * adapt
            )

            record = {
                "query_text": query,
                "scorable_id": str(cand.id),
                "scorable_type": cand.target_type,
                "rank_score": score,
                "components": {
                    "similarity": sim,
                    "reward": rew,
                    "recency": rec,
                    "adaptability": adapt,
                },
                "embedding_type": self.embedding_type,
            }
            results.append(record)

        # Sort by score
        results.sort(key=lambda r: r["rank_score"], reverse=True)

        # Persist all ranks
        self.memory.scorable_ranks.bulk_insert(results)

        if self.logger:
            self.logger.log(
                "ScorableRanked",
                {"query": query[:80], "count": len(results), "top_score": results[0]["rank_score"]},
            )

        return results
