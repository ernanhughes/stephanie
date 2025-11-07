# stephanie/utils/score_utils.py
from __future__ import annotations

import logging

from stephanie.data.score_bundle import ScoreBundle
from stephanie.scoring.scorable import Scorable

logger = logging.getLogger(__name__)


def clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

def safe_mean(xs):
    xs = list(xs or [])
    return 0.0 if not xs else float(sum(xs) / len(xs))

def weighted_mean(vals, weights):
    vals = list(vals or [])
    weights = list(weights or [])
    if not vals:
        return 0.0
    if not weights or len(weights) != len(vals):
        weights = [1.0] * len(vals)
    s = float(sum(max(0.0, w) for w in weights)) or 1.0
    return float(sum(v * max(0.0, w) for v, w in zip(vals, weights)) / s)

def median(xs):
    xs = sorted(list(xs or []))
    n = len(xs)
    if n == 0: 
        return 0.0
    m = n // 2
    return float(xs[m]) if n % 2 else float(0.5 * (xs[m - 1] + xs[m]))

def score_scorable(context: dict, scorable: Scorable, scorer_names: str, dimensions: list, scorer) -> tuple:
    score_results = {}
    for scorer_name in scorer_names:
        try:
            bundle = scorer.score(
                scorer_name,
                context=context,
                scorable=scorable,
                dimensions=dimensions
            )
            for dim, result in bundle.results.items():
                # ensure the result carries its dimension and source
                if not getattr(result, "dimension", None):
                    result.dimension = dim
                if not getattr(result, "source", None):
                    result.source = scorer_name  # fallback if scorer didn't set it

                # use a composite key to avoid overwriting, but keep result.dimension == dim
                key = f"{dim}::{result.source}"
                score_results[key] = result
        except Exception as e:
            logger.error(f"ScorerError scorer: {scorer_name}, scorable_id: {scorable.id}, error: {str(e)}")
            continue

    bundle = ScoreBundle(results=dict(score_results))

    return bundle
