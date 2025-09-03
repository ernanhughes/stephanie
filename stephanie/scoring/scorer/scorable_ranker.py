import math
import random
import time
from typing import Dict, List, Optional

from sklearn.metrics.pairwise import cosine_similarity

from stephanie.data.score_bundle import ScoreBundle
from stephanie.data.score_result import ScoreResult
from stephanie.models.evaluation import EvaluationORM
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorer.base_scorer import BaseScorer


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

        # CBR literature weights: sim=0.45, value=0.30, recency=0.10, diversity=0.10, adapt=0.05
        self.weights = cfg.get(
            "weights",
            {
                "similarity": 0.45,
                "value": 0.30,
                "recency": 0.10,
                "diversity": 0.10,
                "adaptability": 0.05,
            },
        )

        # CBR-specific parameters
        self.lambda_recency = cfg.get(
            "rank_recency_lambda", 0.02
        )  # days decay
        self.epsilon = cfg.get("rank_epsilon", 0.10)  # exploration probability
        self.top_k = cfg.get("rank_top_k", 8)  # max to return
        self.normalize_similarity = cfg.get("normalize_similarity", True)
        self.logger.log(
            "ScorableRankerInitialized",
            {
                "target_type": self.target_type,
                "embedding_type": self.embedding_type,
                "weights": self.weights,
                "normalize_similarity": self.normalize_similarity,
                "cbr_parameters": {
                    "lambda_recency": self.lambda_recency,
                    "epsilon": self.epsilon,
                    "top_k": self.top_k,
                },
            },
        )

    # --- Component scorers ---
    def _similarity(self, query_emb, cand_emb) -> float:
        sim = float(cosine_similarity([query_emb], [cand_emb])[0][0])
        if self.normalize_similarity:
            sim = (sim + 1.0) / 2.0  # [-1,1] -> [0,1]
        return sim

    def _reward(self, scorable: Scorable) -> float:
        eval_rec = (
            self.memory.session.query(EvaluationORM)
            .filter_by(
                scorable_id=scorable.id, scorable_type=scorable.target_type
            )
            .order_by(EvaluationORM.created_at.desc())
            .first()
        )
        return float(eval_rec.scores.get("avg", 0)) if eval_rec else 0.0

    def _adaptability(self, scorable: Scorable, context: dict) -> float:
        """Enhanced adaptability signal with tool compatibility"""
        # Base adaptability from reuse metrics
        base = 0.5
        meta = getattr(scorable, "meta", {}) or {}
        if "reuse_count" in meta and "attempt_count" in meta:
            base = float(meta["reuse_count"]) / max(
                1, float(meta["attempt_count"])
            )

        # Tool compatibility (if context is provided)
        tool_compat = 1.0
        if context:
            tool_compat = self._tool_compatibility(scorable, context)

        # Weighted combination
        return 0.7 * base + 0.3 * tool_compat

    def _diversity(
        self, scorable: Scorable, selected: List[Scorable]
    ) -> float:
        """Maximal Marginal Relevance (MMR) diversity penalty"""
        if not selected:
            return 1.0

        # Get similarity to most similar already-selected case
        selected_embs = [
            self.memory.embedding.get_or_create(s.text) for s in selected
        ]
        cand_emb = self.memory.embedding.get_or_create(scorable.text)

        # Handle potential embedding retrieval failures
        if cand_emb is None or not selected_embs:
            return 1.0

        similarities = cosine_similarity([cand_emb], selected_embs)[0]
        max_sim = float(max(similarities))

        # Diversity = 1 - max similarity to selected
        return 1.0 - max_sim

    def _tool_compatibility(self, scorable: Scorable, context: dict) -> float:
        """Measure compatibility with current available tools"""
        # Extract available tools from context
        available_tools = context.get("available_tools", [])

        # Get tools used in this scorable (if it's a plan trace)
        used_tools = []
        if hasattr(scorable, "meta") and scorable.meta:
            used_tools = scorable.meta.get("tools", [])
        elif hasattr(scorable, "tools"):
            used_tools = scorable.tools

        if not used_tools:
            return 1.0

        return len(set(available_tools) & set(used_tools)) / len(
            set(used_tools)
        )

    def score(
        self, context: dict, scorable: Scorable, dimensions=None
    ) -> dict:
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
            "adaptability": self._adaptability(scorable, context=context),
        }

        rank_score = sum(
            components[k] * self.weights.get(k, 0) for k in components
        )

        result = {
            "scorable_id": scorable.id,
            "scorable_type": scorable.target_type,
            "rank_score": rank_score,
            "text": scorable.text,
            "components": components,
        }

        return result

    # --- Multi-candidate ranking ---
    def rank(
        self,
        query: Scorable,
        candidates: list[Scorable],
        context: dict,
        extra_signals: Optional[Dict[str, float]] = None,
    ):
        # Precompute query embedding once
        query_emb = self.memory.embedding.get_or_create(query.text)
        if query_emb is None:
            self.logger.log(
                "RankingError", {"error": "Failed to get query embedding"}
            )
            return []

        # Score all candidates
        scored = []
        for cand in candidates:
            cand_emb = self.memory.embedding.get_or_create(cand.text)
            if cand_emb is None:
                continue

            components = {
                "similarity": self._similarity(query_emb, cand_emb),
                "value": self._value(
                    cand
                ),  # Renamed from "reward" for clarity
                "recency": self._recency(cand),
                "adaptability": self._adaptability(cand, context),
            }

            scored.append((cand, components))

        if not scored:
            return []

        # Apply MMR + ε-greedy selection
        selected = []
        remaining = scored.copy()
        bundles = []

        # We'll select up to top_k candidates
        while remaining and len(selected) < self.top_k:
            if random.random() < self.epsilon:
                # Exploration: pick randomly from top 20%
                top_20 = sorted(
                    remaining,
                    key=lambda x: self._calculate_rank_score(x[1]),
                    reverse=True,
                )[: max(1, len(remaining) // 5)]
                pick = random.choice(top_20)
                remaining.remove(pick)
                selected.append(pick)
            else:
                # Exploitation: pick highest score with diversity penalty
                best = None
                best_score = -float("inf")

                for cand, components in remaining:
                    # Diversity penalty: reduce score based on similarity to already-picked
                    if selected:
                        selected_embs = [
                            self.memory.embedding.get_or_create(s[0].text)
                            for s in selected
                            if self.memory.embedding.get_or_create(s[0].text)
                            is not None
                        ]
                        cand_emb = self.memory.embedding.get_or_create(
                            cand.text
                        )

                        if cand_emb is not None and selected_embs:
                            similarities = cosine_similarity(
                                [cand_emb], selected_embs
                            )[0]
                            div_penalty = float(max(similarities))
                            diversity = 1.0 - div_penalty
                        else:
                            diversity = 1.0
                    else:
                        diversity = 1.0

                    # Store diversity for final scoring
                    components["diversity"] = diversity

                    # Calculate final score with diversity
                    score = self._calculate_rank_score(components)

                    if score > best_score:
                        best = (cand, components)
                        best_score = score

                if best:
                    remaining.remove(best)
                    selected.append(best)

        # Convert selected items to dicts
        for cand, components in selected:
            rank_score = self._calculate_rank_score(components)

            # Create ScoreBundle (still persist in memory)
            bundle = ScoreBundle(
                results={
                    "rank_score": ScoreResult(
                        dimension="rank_score",
                        score=rank_score,
                        weight=1.0,
                        source="scorable_ranker",
                        rationale="Weighted combo with diversity",
                        attributes=components,
                    )
                }
            )
            bundle.meta = query.to_dict()
            self.memory.evaluations.save_bundle(
                bundle=bundle,
                scorable=cand,
                context=context,
                cfg=self.cfg,
                source="scorable_ranker",
                embedding_type=self.memory.embedding.name,
                evaluator=self.model_type,
            )

            # Append normalized dict instead of raw bundle
            bundles.append({
                "scorable_id": cand.id,
                "scorable_type": cand.target_type,
                "agent_name": getattr(cand, "meta", {}).get("agent_name"),
                "rank_score": rank_score,
                "text": cand.text,
                "components": components,
            })

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
                    "scorable_type": getattr(
                        d.get("scorable"), "target_type", None
                    ),
                }
            elif isinstance(result, dict):
                return {
                    "rank_score": result.get("rank_score")
                    or result.get("scores", {}).get("rank_score"),
                    "components": result.get("components", {}),
                    "scorable_id": result.get("scorable_id"),
                    "scorable_type": result.get("scorable_type"),
                }
            return {
                "rank_score": 0,
                "components": {},
                "scorable_id": None,
                "scorable_type": None,
            }

        normalized = [extract(r) for r in results]
        top_result = normalized[0]  # <-- use normalized, not results
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

    # scorable_ranker.py
    def _value(self, scorable: Scorable) -> float:
        return self.memory.scores.get_value_signal_for_target(
            target_id=str(scorable.id),
            target_type=scorable.target_type,
            prefer=("hrm", "HRM", "rank_score"),
            normalize=True,
            default=0.0,
        )

    def _calculate_rank_score(self, components: dict) -> float:
        """Calculate final rank score with proper weighting"""
        return (
            self.weights["similarity"] * components.get("similarity", 0.0)
            + self.weights["value"] * components.get("value", 0.0)
            + self.weights["recency"] * components.get("recency", 0.0)
            + self.weights["diversity"] * components.get("diversity", 0.0)
            + self.weights["adaptability"]
            * components.get("adaptability", 0.0)
        )

    def _recency(self, scorable: Scorable) -> float:
        """
        Exponential decay with a configurable half-life.
        Tries scorable.created_at → PlanTrace.created_at → latest Evaluation.created_at.
        Returns a value in (0,1], ~1 for very recent, ~0 for very old; 0.5 if unknown.
        """
        halflife_days = self.cfg.get("recency_halflife_days", 30)

        created_at = getattr(scorable, "created_at", None)

        # Fallback 1: try PlanTrace by id
        if created_at is None:
            try:
                pt = getattr(self.memory, "plan_traces", None)
                if pt:
                    pt_row = pt.get_by_trace_id(str(scorable.id))
                    if pt_row and getattr(pt_row, "created_at", None):
                        created_at = pt_row.created_at
            except Exception:
                pass

        # Fallback 2: latest Evaluation timestamp
        if created_at is None:
            try:
                eval_rec = (
                    self.memory.session.query(EvaluationORM)
                    .filter_by(
                        scorable_id=str(scorable.id),
                        scorable_type=scorable.target_type,
                    )
                    .order_by(EvaluationORM.created_at.desc())
                    .first()
                )
                if eval_rec:
                    created_at = eval_rec.created_at
            except Exception:
                pass

        if created_at is None:
            return 0.5  # neutral when we can’t tell

        age_sec = time.time() - created_at.timestamp()
        # choose tau so that score halves every halflife_days
        tau = (halflife_days * 24 * 3600) / math.log(2)
        return math.exp(-age_sec / tau)
