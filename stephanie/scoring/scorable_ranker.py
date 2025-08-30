import math
import time
from typing import Dict, List, Optional

from sklearn.metrics.pairwise import cosine_similarity

from stephanie.scoring.base_scorer import BaseScorer
from stephanie.scoring.scorable import Scorable
from stephanie.models.evaluation import EvaluationORM
from stephanie.data.score_bundle import ScoreBundle
from stephanie.data.score_result import ScoreResult

class ScorableRanker(BaseScorer):
    """
    Universal ranking engine for all Scoreables (documents, prompts, plan traces, etc.).
    Integrates with the standard BaseScorer API:
      - score(context, scorable, dimensions) → dict
      - rank(query, candidates, extra_signals) → List[EvaluationORM]
    """

    def __init__(self, cfg, memory, logger):
        super().__init__(cfg, memory, logger)
        self.model_type = "scorable_rank"

        # Embedding backend
        self.embedding_type = self.memory.embedding.name
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
            .filter_by(scorable_id=scorable.id, scorable_type=scorable.target_type)
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


    def score(self, context: dict, scorable: Scorable, dimensions=None) -> dict:
        """
        Score a single scorable relative to the current goal/query in context.
        Returns a dict with rank_score + component scores.
        """
        goal_text = context.get("goal", {}).get("goal_text", "")
        query_emb = self.memory.embedding.get_or_create(goal_text)
        cand_emb = self.memory.embedding.get_or_create(scorable.text)

        # Component signals
        components = {
            "similarity": self._similarity(query_emb, cand_emb),
            "reward": self._reward(scorable),
            "recency": self._recency(scorable),
            "adaptability": self._adaptability(scorable),
        }

        rank_score = sum(
            components[k] * self.weights.get(k, 0) for k in components
        )

        result = {
            "scorable_id": scorable.id,
            "scorable_type": scorable.target_type,
            "rank_score": rank_score,
            "components": components,
        }

        return result

    # --- Multi-candidate ranking ---
    def rank(self, query: Scorable, candidates: list[Scorable], context: dict, extra_signals: Optional[Dict[str, float]] = None):
        bundles = []
        extra_signals = extra_signals or {}
        query_emb = self.memory.embedding.get_or_create(query.text)

        for cand in candidates:
            cand_emb = self.memory.embedding.get_or_create(cand.text)

            components = {
                "similarity": self._similarity(query_emb, cand_emb),
                "reward": self._reward(cand),
                "recency": self._recency(cand),
                "adaptability": self._adaptability(cand),
            }
            components.update(extra_signals)

            rank_score = sum(
                components[k] * self.weights.get(k, 0) for k in components
            )

            bundle = ScoreBundle(results={
                "rank_score": ScoreResult(
                    dimension="rank_score",
                    score=rank_score,
                    weight=1.0,
                    source="scorable_ranker",
                    rationale="Weighted combo",
                    attributes=components,
                )
            })
            bundle.meta = query.to_dict()

            # Persist via EvaluationStore
            self.memory.evaluations.save_bundle(
                bundle=bundle,
                scorable=cand, 
                context=context,
                cfg=self.cfg,
                source="scorable_ranker",
                embedding_type=self.memory.embedding.name,
                evaluator=self.model_type
            ) 

            bundles.append(bundle)

        return bundles


    def to_report_dict(self, query: Scorable, results: list) -> dict:
        """
        Convert ranking results (ScoreBundles or dicts) into a SYS-friendly report dict.
        """

        if not results:
            return {
                "event": "scorable_ranking",
                "step": "ScorableRanker",
                "details": f"No candidates ranked for query '{query.text[:50]}...'",
                "query": {
                    "id": query.id,
                    "type": query.target_type,
                    "text": query.text[:200],
                },
                "candidates": 0,
            }

        # Normalize results → always extract rank_score
        def extract(result):
            if hasattr(result, "to_dict"):
                d = result.to_dict()
                return {
                    "rank_score": d["results"]["rank_score"].score,
                    "components": d["results"]["rank_score"].attributes,
                    "scorable_id": getattr(d.get("scorable"), "id", None),
                    "scorable_type": getattr(d.get("scorable"), "target_type", None),
                }
            elif isinstance(result, dict):
                return {
                    "rank_score": result.get("rank_score")
                    or result.get("scores", {}).get("rank_score"),
                    "components": result.get("components", {}),
                    "scorable_id": result.get("scorable_id"),
                    "scorable_type": result.get("scorable_type"),
                }
            return {"rank_score": 0, "components": {}, "scorable_id": None, "scorable_type": None}

        normalized = [extract(r) for r in results]
        top_result = results[0]
        return {
            "event": "scorable_ranking",
            "step": "ScorableRanker",
            "details": f"Ranked {len(normalized)} candidates for query '{query.text[:50]}...'",
            "query": {
                "id": query.id,
                "type": query.target_type,
                "text": query.text[:200],
            },
            "top_score": top_result.get("rank_score"),
            "top_candidate": {
                "id": top_result.get("scorable_id"),
                "type": top_result.get("scorable_type"),
                "scores": top_result.get("components"),
            },
        }
