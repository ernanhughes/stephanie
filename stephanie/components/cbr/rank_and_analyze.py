# stephanie/cbr/rank_and_analyze.py
from typing import Dict, List, Tuple

from stephanie.constants import GOAL
from stephanie.data.score_corpus import ScoreCorpus
from stephanie.scoring.scorable import Scorable, ScorableType
from stephanie.utils.score_utils import score_scorable


class DefaultRankAndAnalyze:
    def __init__(self, cfg, memory, container, logger, ranker, mars=None):
        self.cfg = cfg 
        self.memory = memory
        self.logger = logger
        self.ranker = ranker
        self.mars = mars
        self.dimensions = cfg.get("dimensions", ["alignment"])
        self.enabled_scorers = cfg.get("enabled_scorers", ["sicql", "hrm"])
        self.scoring = None

    def _normalize(self, item) -> dict:
        if isinstance(item, dict):
            r = dict(item)
            r.setdefault("rank", r.get("position") or r.get("order") or 0)
            return r
        return {
            "id": getattr(item, "id", None)
            or getattr(item, "scorable_id", None),
            "text": getattr(item, "text", "") or getattr(item, "content", ""),
            "rank": getattr(item, "rank", 0),
        }

    def run(
        self, context, scorables: List[dict]
    ) -> Tuple[List[dict], Dict, Dict, List[str], Dict]:
        goal = context[GOAL]
        ranked, bundles = [], {}

        if scorables:
            query = Scorable(
                id=goal["id"],
                text=goal["goal_text"],
                target_type=ScorableType.GOAL,
            )
            scorables = [
                Scorable(
                    id=h.get("id"),
                    text=h.get("text", ""),
                    target_type=h.get("target_type", ScorableType.HYPOTHESIS),
                )
                for h in scorables
            ]
            ranked_raw = (
                self.ranker.rank(
                    query=query, candidates=scorables, context=context
                )
                or []
            )
            ranked = [self._normalize(r) for r in ranked_raw]
            for s in scorables:
                try:
                    bundles[s.id] = score_scorable(
                        context,
                        s,
                        self.enabled_scorers,
                        self.dimensions,
                        self.scoring,
                    )
                except Exception:
                    pass

        corpus = ScoreCorpus(bundles=bundles)
        mars_results, recommendations = {}, []
        if self.mars:
            mars_results = self.mars.calculate(corpus, context=context) or {}
            recommendations = (
                self.mars.generate_recommendations(mars_results) or []
            )

        scores_payload = {}
        for sid, bundle in bundles.items():
            if hasattr(bundle, "to_dict"):
                try:
                    scores_payload[sid] = bundle.to_dict()
                except Exception:
                    scores_payload[sid] = {}
            else:
                scores_payload[sid] = {}

        return ranked, corpus, mars_results, recommendations, scores_payload
