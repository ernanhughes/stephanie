
import logging

from stephanie.data.score_bundle import ScoreBundle
from stephanie.scoring.scorable import Scorable

logger = logging.getLogger(__name__)

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
