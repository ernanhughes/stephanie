# stephanie/agents/proximity.py
import itertools

import numpy as np

from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.mixins.scoring_mixin import \
    ScoringMixin  # Adjust path if needed
from stephanie.constants import (DATABASE_MATCHES, GOAL, GOAL_TEXT,
                                 PIPELINE_RUN_ID, TEXT)
from stephanie.models import EvaluationORM
from stephanie.scoring.proximity_scorer import ProximityScorer
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorable_factory import TargetType
from stephanie.utils import compute_similarity_matrix


class ProximityAgent(ScoringMixin, BaseAgent):
    """
    The Proximity Agent calculates similarity between hypotheses and builds a proximity graph.
    """

    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.similarity_threshold = cfg.get("similarity_threshold", 0.75)
        self.max_graft_candidates = cfg.get("max_graft_candidates", 3)
        self.top_k_database_matches = cfg.get("top_k_database_matches", 5)

    async def run(self, context: dict) -> dict:
        goal = context.get(GOAL)
        goal_text = goal.get(GOAL_TEXT)
        current_hypotheses = self.get_hypotheses(context)

        db_texts = self.memory.hypotheses.get_similar(
            goal_text, limit=self.top_k_database_matches
        )
        self.logger.log(
            "DatabaseHypothesesMatched",
            {
                GOAL: goal,
                "matches": [{"text": h[:100]} for h in db_texts],
            },
        )

        hypotheses_texts = [h.get(TEXT) for h in current_hypotheses]
        all_hypotheses = list(set(hypotheses_texts + db_texts))

        if not all_hypotheses:
            self.logger.log("NoHypothesesForProximity", {"reason": "empty_input"})
            return context

        similarities = compute_similarity_matrix(all_hypotheses, self.memory, self.logger)
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

        graft_candidates = [
            (h1, h2) for h1, h2, sim in similarities if sim >= self.similarity_threshold
        ]
        clusters = self._cluster_hypotheses(graft_candidates)

        context[self.output_key] = {
            "clusters": clusters,
            "graft_candidates": graft_candidates,
            DATABASE_MATCHES: db_texts,
            "proximity_graph": similarities,
        }

        top_similar = similarities[: self.max_graft_candidates]
        to_merge = {
            GOAL: goal,
            "most_similar": "\n".join(
                [
                    f"{i + 1}. {h1} ↔ {h2} (sim: {score:.2f})"
                    for i, (h1, h2, score) in enumerate(top_similar)
                ]
            ),
        }

        merged = {**context, **to_merge}
        summary_prompt = self.prompt_loader.load_prompt(self.cfg, merged)

        summary_output = self.call_llm(summary_prompt, merged)
        context["proximity_summary"] = summary_output

        scorable = Scorable(text=summary_output, target_type=TargetType.HYPOTHESIS)
        score_result = self.score_item(
            scorable=scorable,
            context=context,
            metrics="proximity",  # Must match your config key: `proximity_score_config`
            scorer=ProximityScorer(self.cfg, self.memory, self.logger, self.call_llm),
        )
        score = score_result["score"]

        self.logger.log(
            "ProximityAnalysisScored",
            {
                "score": score,
                "analysis": summary_output[:300],
            },
        )

        # Compute additional dimensions
        cluster_count = len(clusters)
        top_k_sims = [sim for _, _, sim in similarities[: self.max_graft_candidates]]
        avg_top_k_sim = sum(top_k_sims) / len(top_k_sims) if top_k_sims else 0.0
        graft_count = len(graft_candidates)

        # Format as new score schema
        structured_scores = {
            "stage": "proximity",
            "dimensions": {
                "proximity_score": {
                    "score": score,
                    "rationale": summary_output,
                    "weight": 1.0,
                },
                "cluster_count": {
                    "score": cluster_count,
                    "rationale": f"Total unique clusters of hypotheses: {cluster_count}",
                    "weight": 0.5,
                },
                "avg_similarity_top_k": {
                    "score": avg_top_k_sim,
                    "rationale": f"Average similarity among top-{self.max_graft_candidates} pairs.",
                    "weight": 0.8,
                },
                "graft_pair_count": {
                    "score": graft_count,
                    "rationale": f"Pairs exceeding similarity threshold ({self.similarity_threshold}).",
                    "weight": 0.7,
                },
            },
        }
        structured_scores["final_score"] = (
            round(
                sum(
                    dim["score"] * dim["weight"]
                    for dim in structured_scores["dimensions"].values()
                )
                / sum(
                    dim["weight"] for dim in structured_scores["dimensions"].values()
                ),
                2,
            ),
        )

        # Save per-hypothesis score
        for hypothesis in current_hypotheses:
            score_obj = EvaluationORM(
                goal_id=goal.get("goal_id"),
                target_type="hypothesis",
                target_id=hypothesis.get("id"),
                agent_name=self.name,
                model_name=self.model_name,
                evaluator_name=self.name,
                extra_data={"summary": summary_output},
                scores=structured_scores, 
                pipeline_run_id=context.get(PIPELINE_RUN_ID),
            )
            self.memory.evaluations.insert(score_obj)

        return context

    def _cosine(self, a, b):
        a = np.array(list(a), dtype=float)
        b = np.array(list(b), dtype=float)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def _cluster_hypotheses(self, graft_candidates: list[tuple]) -> list[list[str]]:
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
