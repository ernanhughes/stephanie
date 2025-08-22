# stephanie/scoring/proximity_scorer.py
import numpy as np
from stephanie.scoring.base_scorer import BaseScorer
from stephanie.utils import compute_similarity_matrix
from stephanie.data.score_result import ScoreResult
from stephanie.data.score_bundle import ScoreBundle
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine, euclidean, cityblock

def jaccard_similarity(a: str, b: str) -> float:
    set_a, set_b = set(a.lower().split()), set(b.lower().split())
    return len(set_a & set_b) / len(set_a | set_b) if set_a | set_b else 0.0

def tfidf_cosine(a: str, b: str) -> float:
    vec = TfidfVectorizer().fit([a, b])
    mat = vec.transform([a, b]).toarray()
    return 1 - cosine(mat[0], mat[1])

def cosine_similarity(v1, v2) -> float:
    return 1 - cosine(v1, v2)

def euclidean_similarity(v1, v2) -> float:
    return 1 / (1 + euclidean(v1, v2))

def manhattan_similarity(v1, v2) -> float:
    return 1 / (1 + cityblock(v1, v2))


# Registry of all available metrics
METRIC_REGISTRY = {
    "jaccard": lambda a, b, *_: jaccard_similarity(a, b),
    "tfidf": lambda a, b, *_: tfidf_cosine(a, b),
    "cosine": lambda _, __, v1, v2: cosine_similarity(v1, v2),
    "euclidean": lambda _, __, v1, v2: euclidean_similarity(v1, v2),
    "manhattan": lambda _, __, v1, v2: manhattan_similarity(v1, v2),
}

class ProximityScorer:
    """
    Proximity-based scorer for hypotheses.
    Uses multiple similarity/distance metrics as dimensions.
    """

    def __init__(self, cfg, memory=None, logger=None):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger
        self.similarity_threshold = cfg.get("similarity_threshold", 0.75)
        self.max_pairs = cfg.get("max_graft_candidates", 5)
        self.top_k_db = cfg.get("top_k_database_matches", 10)
        self.use_db = cfg.get("use_database_search", True)
        self.model_type = "proximity"

    def score(self, context: dict, scorable, metrics: list[dict] = {}) -> ScoreBundle:
        """
        Compute proximity-based scores for a given scorable in context.

        Args:
            context (dict): Pipeline or agent context containing "hypotheses".
            scorable: The scorable object (must have .text).
            metrics (list[dict]): Metric definitions from config.

        Returns:
            ScoreBundle: Containing one ScoreResult per proximity metric.
        """
        text = scorable.text

        # Step 1. Collect hypotheses
        hypotheses = []
        if context and "hypotheses" in context:
            hypotheses = [h["text"] for h in context["hypotheses"]]
        if text and text not in hypotheses:
            hypotheses.append(text)

        if not hypotheses:
            empty_result = ScoreResult(
                dimension="avg_similarity",
                score=0.0,
                source=self.model_type,
                rationale="No hypotheses to compare.",
                weight=1.0,
                attributes={"pair_count": 0},
            )
            return ScoreBundle(results={"avg_similarity": empty_result})

        # Step 2. Compute similarity matrix
        similarities = compute_similarity_matrix(hypotheses, self.memory, self.logger)

        # Step 3. Compute metrics
        metrics = self._compute_metrics(similarities, hypotheses, text)

        # Step 4. Wrap metrics as ScoreResults
        results = {}
        for dim, meta in metrics.items():
            results[dim] = ScoreResult(
                dimension=dim,
                score=meta["score"],
                source=self.model_type,
                rationale=meta["rationale"],
                weight=meta["weight"],
                attributes=meta.get("attributes", {}),
            )

        return ScoreBundle(results=results)

    def _compute_metrics(self, similarities, hypotheses, target_text):
        """Turn similarity matrix into structured dimensions."""
        if not similarities:
            return {
                "avg_similarity": {
                    "score": 0.0,
                    "rationale": "No similarity pairs found.",
                    "weight": 1.0,
                    "attributes": {"pair_count": 0},
                }
            }

        sims = [s for _, _, s in similarities]
        pair_count = len(sims)

        # Core stats
        avg_sim = float(np.mean(sims))
        max_sim = float(np.max(sims))
        min_sim = float(np.min(sims))
        sim_var = float(np.var(sims))
        sims_sorted = sorted(sims, reverse=True)[: self.max_pairs]
        avg_topk = float(np.mean(sims_sorted)) if sims_sorted else 0.0

        graft_pairs = [(h1, h2) for h1, h2, s in similarities if s >= self.similarity_threshold]
        graft_count = len(graft_pairs)

        # Cluster measures
        cluster_density = graft_count / pair_count if pair_count > 0 else 0.0
        redundancy_index = avg_topk  # high top-k mean → redundancy
        novelty_index = 1.0 - min_sim  # inverse of worst similarity

        # DB measures (optional)
        db_scores = {}
        if self.use_db and self.memory:
            db_scores = self._compute_db_proximity(target_text)

        metrics = {
            # Embedding stats
            "avg_similarity": {
                "score": avg_sim,
                "rationale": f"Average pairwise similarity across {pair_count} pairs.",
                "weight": 1.2,
                "attributes": {"pair_count": pair_count},
            },
            "max_similarity": {
                "score": max_sim,
                "rationale": "Strongest semantic overlap between any two hypotheses.",
                "weight": 1.0,
            },
            "min_similarity": {
                "score": min_sim,
                "rationale": "Weakest semantic overlap (outlier detection).",
                "weight": 0.5,
            },
            "similarity_variance": {
                "score": sim_var,
                "rationale": "Variance of similarity scores (spread/uncertainty).",
                "weight": 0.7,
            },
            "avg_topk_similarity": {
                "score": avg_topk,
                "rationale": f"Average similarity among top-{self.max_pairs} pairs.",
                "weight": 1.1,
            },
            "graft_pair_count": {
                "score": graft_count,
                "rationale": f"Number of pairs ≥ {self.similarity_threshold}.",
                "weight": 0.9,
                "attributes": {"graft_pairs": graft_pairs},
            },
            # Cluster / diversity
            "cluster_density": {
                "score": cluster_density,
                "rationale": "Fraction of pairs above threshold (cluster tightness).",
                "weight": 1.0,
            },
            "redundancy_index": {
                "score": redundancy_index,
                "rationale": "High top-k similarity → risk of redundancy.",
                "weight": 0.8,
            },
            "novelty_index": {
                "score": novelty_index,
                "rationale": "Encourages new hypotheses (low overlap).",
                "weight": 0.9,
            },
        }

        # Merge DB metrics if available
        metrics.update(db_scores)
        return metrics

    def _compute_db_proximity(self, target_text):
        """Compute database match metrics for hypothesis proximity."""
        matches = self.memory.embedding.search_related(
            target_text, top_k=self.top_k_db
        )
        if not matches:
            return {
                "db_match_score": {
                    "score": 0.0,
                    "rationale": "No DB matches found.",
                    "weight": 1.0,
                },
                "db_overlap_count": {
                    "score": 0,
                    "rationale": "No overlaps above threshold.",
                    "weight": 0.7,
                },
                "db_diversity": {
                    "score": 0.0,
                    "rationale": "No DB diversity measurable.",
                    "weight": 0.6,
                },
            }

        scores = [m.similarity for m in matches]
        avg_match = float(np.mean(scores))
        overlap_count = sum(1 for s in scores if s >= self.similarity_threshold)
        db_div = float(np.std(scores))  # diversity from DB history

        return {
            "db_match_score": {
                "score": avg_match,
                "rationale": f"Average similarity to top-{self.top_k_db} DB matches.",
                "weight": 1.0,
            },
            "db_overlap_count": {
                "score": overlap_count,
                "rationale": f"Count of DB matches ≥ {self.similarity_threshold}.",
                "weight": 0.7,
            },
            "db_diversity": {
                "score": db_div,
                "rationale": "Spread of DB match similarities (diversity).",
                "weight": 0.6,
            },
        }
