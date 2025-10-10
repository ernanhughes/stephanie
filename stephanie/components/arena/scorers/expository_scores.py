# stephanie/arena/scorers/expository_scores.py
from __future__ import annotations

from stephanie.components.arena.plugins.interfaces import JobCtx, Scorer
from stephanie.components.arena.plugins.registry import register_scorer


@register_scorer
class ExpositoryHeuristicScorer:
    name = "expository_heuristic"

    def score(self, ctx: JobCtx, m):
        # m = metrics dict from PlayResult
        # keep interpretable sub-scores
        return {
            "primary": 0.7*m.get("r_solve", 0.0) + 0.2*m.get("coherence", 0.0) + 0.1*m.get("readability", 0.0),
            "kept_bonus": 0.1 if m.get("kept", 0.0) > 0 else 0.0
        }

@register_scorer
class LengthPreferenceScorer:
    name = "length_pref"
    def __init__(self, target_words: int = 700):
        self.target_words = target_words
    def score(self, ctx: JobCtx, m):
        # optional: look up draft by id to compute exact word count, or include in metrics
        # placeholder: n/a if not available
        return {"primary": 0.0}
