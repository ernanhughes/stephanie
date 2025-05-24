# co_ai/agents/proximity.py
import itertools

import numpy as np

from co_ai.agents.base import BaseAgent
from co_ai.constants import GOAL, HYPOTHESES


class ProximityAgent(BaseAgent):
    """
    The Proximity Agent calculates similarity between hypotheses and builds a proximity graph.

    From the paper:
    > 'The Proximity agent calculates the similarity between research hypotheses and builds a proximity graph...'
    """

    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

        self.similarity_threshold = cfg.get("similarity_threshold", 0.75)
        self.max_graft_candidates = cfg.get("max_graft_candidates", 3)
        self.top_k_database_matches = self.cfg.get("top_k_database_matches", 5)

    async def run(self, context: dict) -> dict:
        """
        Run proximity analysis on current hypotheses.

        Args:
            context: Dictionary with keys:
                - hypotheses: list of hypothesis strings
                - ranked: list of (hypothesis, score)
                - evolved: list of hypotheses
                - feedback: optional previous feedback
        Returns:
            dict with keys:
                - clusters: list of hypothesis groups
                - graft_candidates: list of high-similarity pairs
                - proximity_graph: list of (h1, h2, similarity)
        """
        current_goal = self.extract_goal_text(context.get(GOAL))
        current_hypotheses = self.get_hypotheses(context)

        # Fetch historical hypotheses from DB
        db_hypotheses = self.memory.hypotheses.get_similar(
            current_goal, top_k=self.top_k_database_matches
        )

        db_texts = [h["text"] for h in db_hypotheses]

        self.logger.log(
            "DatabaseHypothesesMatched",
            {
                GOAL: current_goal[:60],
                "matches": [
                    {
                        "text": h["text"][:100],
                        "similarity": h["similarity"],
                        "source": h.get("source"),
                    }
                    for h in db_hypotheses
                ],
            },
        )

        # Combine current and past hypotheses
        all_hypotheses = list(set(current_hypotheses + db_texts))

        if not all_hypotheses.__len__():
            self.logger.log("NoHypothesesForProximity", {"reason": "empty_input"})
            return context

        # Compute pairwise similarity between all hypotheses
        similarities = self._compute_similarity_matrix(all_hypotheses)

        # Log proximity graph
        self.logger.log(
            "ProximityGraphComputed",
            {
                "total_pairs": len(similarities),
                "threshold": self.similarity_threshold,
                "top_matches": [
                    {"pair": (h1[:60], h2[:60]), "score": sim}
                    for h1, h2, sim in similarities[:3]
                ],
            },
        )

        # Identify grafting candidates
        graft_candidates = [
            (h1, h2) for h1, h2, sim in similarities if sim >= self.similarity_threshold
        ]
        # Cluster similar hypotheses
        clusters = self._cluster_hypotheses(graft_candidates)

        # Store results in context
        context[self.output_key] = {
            "clusters": clusters,
            "graft_candidates": graft_candidates,
            "database_matches": db_hypotheses,
            "proximity_graph": similarities,
        }

        return context

    def _compute_similarity_matrix(self, hypotheses: list[str]) -> list[tuple]:
        """Compute pairwise cosine similarity between hypotheses"""
        vectors = []
        valid_hypotheses = []

        for h in hypotheses:
            vec = self.memory.embedding.get_or_create(h)
            if vec is None:
                self.logger.log("MissingEmbedding", {"hypothesis_snippet": h[:60]})
                continue
            vectors.append(vec)
            valid_hypotheses.append(h)

        similarities = []

        for i, j in itertools.combinations(range(len(valid_hypotheses)), 2):
            h1 = valid_hypotheses[i]
            h2 = valid_hypotheses[j]
            sim = self._cosine(vectors[i], vectors[j])
            similarities.append((h1, h2, sim))

        # Sort by similarity
        similarities.sort(key=lambda x: x[2], reverse=True)
        return similarities

    def _cosine(self, a, b):
        """Compute cosine similarity between two vectors"""
        a = np.array(list(a), dtype=float)
        b = np.array(list(b), dtype=float)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def _cluster_hypotheses(self, graft_candidates: list[tuple]) -> list[list[str]]:
        """Build clusters of similar hypotheses"""
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

        # Merge overlapping clusters
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
