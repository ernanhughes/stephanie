# stephanie/agents/proximity.py
import numpy as np

from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.mixins.scoring_mixin import ScoringMixin
from stephanie.constants import DATABASE_MATCHES, GOAL, GOAL_TEXT
from stephanie.scoring.scorable_factory import ScorableFactory, TargetType
from stephanie.scoring.scorer.proximity_scorer import ProximityScorer


class ProximityAgent(ScoringMixin, BaseAgent):
    """
    Agent wrapper for the ProximityScorer.
    Produces:
      - direct proximity scores for given hypotheses
      - database-side nearest neighbors (similar hypotheses, etc.)
      - clusters of related hypotheses (graft candidates)
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.similarity_threshold = cfg.get("similarity_threshold", 0.75)
        self.max_graft_candidates = cfg.get("max_graft_candidates", 3)
        self.top_k_database_matches = cfg.get("top_k_database_matches", 5)

        self.scorer = ProximityScorer(self.cfg, memory=self.memory, logger=self.logger)
        self.metrics = cfg.get("metrics", {})

    async def run(self, context: dict) -> dict:
        documents = self.get_scorables(context)
        proximity_results = []

        # --- score incoming hypotheses against goal
        for document in documents:
            scorable = ScorableFactory.from_dict(document, TargetType.HYPOTHESIS)
            score = self.scorer.score(
                context=context,
                scorable=scorable,
                metrics=self.metrics,
            )
            self.logger.log("ProximityScoreComputed", score.to_dict())
            proximity_results.append(score.to_dict())

        # --- fetch DB-side neighbors for the goal
        goal = context.get(GOAL)
        goal_text = goal.get(GOAL_TEXT) if goal else None
        db_texts = []
        if goal_text:
            try:
                db_texts = self.memory.hypotheses.get_similar(
                    goal_text, limit=self.top_k_database_matches
                )
            except Exception as e:
                if self.logger:
                    self.logger.log("ProximityDBFetchFailed", {"error": str(e)})

        # --- cluster hypotheses for grafting
        graft_candidates = self._find_graft_candidates(proximity_results)
        clusters = self._cluster_hypotheses(graft_candidates)

        # --- write results back to context
        context[self.output_key] = {"scores": proximity_results,
                                      DATABASE_MATCHES: db_texts,
                                      "graft_clusters": clusters}
        return context

    # ---------------------------
    # Internal helpers
    # ---------------------------

    def _cosine(self, a, b) -> float:
        """Cosine similarity between two vectors."""
        a = np.array(list(a), dtype=float)
        b = np.array(list(b), dtype=float)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def _find_graft_candidates(self, proximity_results: list[dict]) -> list[tuple[str, str]]:
        """
        Identify hypothesis pairs above threshold as candidates for clustering.
        Expects `proximity_results` items with `source_text` and `score` fields.
        """
        candidates = []
        for res in proximity_results:
            # assume the scorer returns structure like:
            # { "dimension": "avg_similarity", "score": float,
            #   "source_text": "...", "target_text": "..." }
            sim = res.get("score", 0.0)
            if sim >= self.similarity_threshold:
                src = res.get("source_text")
                tgt = res.get("target_text")
                if src and tgt:
                    candidates.append((src, tgt))

        # limit number of pairs if configured
        return candidates[: self.max_graft_candidates]

    def _cluster_hypotheses(self, graft_candidates: list[tuple[str, str]]) -> list[list[str]]:
        """
        Group hypotheses into clusters based on shared graft candidates.
        """
        clusters = []

        for h1, h2 in graft_candidates:
            found = False
            for cluster in clusters:
                if h1 in cluster or h2 in cluster:
                    if h1 not in cluster:
                        cluster.append(h1)
                    if h2 not in cluster:
                        cluster.append(h2)
                    found = True
                    break
            if not found:
                clusters.append([h1, h2])

        # merge overlapping clusters
        merged_clusters = []
        for cluster in clusters:
            merged = False
            for mc in merged_clusters:
                if set(cluster) & set(mc):
                    mc.extend(cluster)
                    merged = True
                    break
            if not merged:
                merged_clusters.append(list(set(cluster)))

        return merged_clusters
