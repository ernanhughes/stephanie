import math
import time
from typing import Dict, List

from sklearn.metrics.pairwise import cosine_similarity

from stephanie.scoring.base_scorer import BaseScorer
from stephanie.scoring.scorable import Scorable
from stephanie.models.evaluation import EvaluationORM
from stephanie.models.score import ScoreORM


class ScorableRanker(BaseScorer):
    """
    Universal ranking engine for all Scoreables (documents, prompts, plan traces, etc.).
    Produces an EvaluationORM per candidate with component signals stored as ScoreORMs.
    """

    def __init__(self, cfg, memory, logger):
        super().__init__(cfg, memory, logger)
        self.model_type = "scorable_rank"

        # Embedding backend
        self.embedding_type = memory.embedding.type
        self.dim = memory.embedding.dim
        self.hdim = memory.embedding.hdim

        # Config
        self.target_type = cfg.get("target_type", "document")
        self.weights = cfg.get(
            "weights",
            {"similarity": 0.4, "reward": 0.3, "recency": 0.2, "adaptability": 0.1},
        )

        self.logger.log(
            "ScorableRankerInitialized",
            {
                "target_type": self.target_type,
                "embedding_type": self.embedding_type,
                "weights": self.weights,
            },
        )

    # --- Component scorers ---
    def _similarity(self, query_emb, cand_emb) -> float:
        return float(cosine_similarity([query_emb], [cand_emb])[0][0])

    def _reward(self, scorable: Scorable) -> float:
        eval_rec = (
            self.memory.session.query(EvaluationORM)
            .filter_by(target_id=scorable.id, target_type=scorable.target_type)
            .order_by(EvaluationORM.created_at.desc())
            .first()
        )
        return float(eval_rec.scores.get("avg", 0)) if eval_rec else 0.0

    def _recency(self, scorable: Scorable) -> float:
        created_at = getattr(scorable, "created_at", None)
        if not created_at:
            return 0.5
        age_sec = time.time() - created_at.timestamp()
        return math.exp(-age_sec / (60 * 60 * 24 * 30))  # 30-day half-life

    def _adaptability(self, scorable: Scorable) -> float:
        meta = getattr(scorable, "meta", {}) or {}
        return float(meta.get("reuse_count", 0)) / float(meta.get("attempt_count", 1))

    # --- Main API ---
    def rank(
        self,
        query: Scorable,
        candidates: List[Scorable],
        extra_signals: Dict[str, float] = None,
    ) -> List[EvaluationORM]:
        """
        Rank candidates for a query. Returns EvaluationORM objects persisted in DB.
        :param query: The query Scorable (defines ranking context).
        :param candidates: List of Scorable objects to rank.
        :param extra_signals: Optional dict of {signal_name: value}.
        """
        extra_signals = extra_signals or {}

        # Get embeddings
        query_emb = self.memory.embedding.get_or_create(query.text)

        evals = []
        for cand in candidates:
            cand_emb = self.memory.embedding.get_or_create(cand.text)

            # Component signals
            components = {
                "similarity": self._similarity(query_emb, cand_emb),
                "reward": self._reward(cand),
                "recency": self._recency(cand),
                "adaptability": self._adaptability(cand),
            }
            components.update(extra_signals)

            # Weighted sum
            rank_score = sum(
                components[k] * self.weights.get(k, 0) for k in components
            )

            # Persist EvaluationORM
            evaluation = EvaluationORM(
                goal_id=None,
                target_type=cand.target_type,
                target_id=str(cand.id),
                query_type=query.target_type,
                query_id=str(query.id),
                embedding_type=self.embedding_type,
                agent_name="ScorableRanker",
                source="ranking",
                model_name="scorable_ranker",
                evaluator_name="ScorableRanker",
                strategy="weighted_signals",
                scores={"rank_score": rank_score, **components},
                extra_data={"components": components, "weights": self.weights},
            )
            self.memory.session.add(evaluation)
            self.memory.session.flush()  # assign ID before creating ScoreORMs

            # Store component scores as ScoreORMs
            for dim, val in components.items():
                score = ScoreORM(
                    evaluation_id=evaluation.id,
                    dimension=dim,
                    score=float(val),
                    weight=self.weights.get(dim, 0),
                    source="scorable_ranker",
                )
                self.memory.session.add(score)

            evals.append(evaluation)

        # Commit all
        self.memory.session.commit()

        if self.logger:
            self.logger.log(
                "ScorableRankCompleted",
                {
                    "query": query.text[:80],
                    "count": len(candidates),
                    "top_score": max(e.scores["rank_score"] for e in evals)
                    if evals
                    else None,
                },
            )

        return evals
