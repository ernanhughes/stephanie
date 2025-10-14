# stephanie/agents/phoshrm.py
from __future__ import annotations

from stephanie.agents.base_agent import BaseAgent
from typing import Dict, List, Tuple, Iterable
import pandas as pd

from stephanie.constants import GOAL, GOAL_TEXT


class PhoshrmAgent(BaseAgent):
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)

    async def run(self, context: dict) -> dict:
        # Implement agent logic here
        return context
    


# stephanie/eval/score_matrix.py

def build_score_matrix(
    responses: List[str],
    *,
    goal_text: str,
    dimensions: List[str],
    scorers: Dict[str, object],   # {"hrm": HRMScorer(...), "tiny": TinyScorer(...)}
    logger=None,
) -> Tuple[pd.DataFrame, dict]:
    """
    Returns:
      df: MultiIndex columns (model, dimension), rows aligned with `responses`
      meta: lightweight run metadata
    """
    # Try Stephanie’s Scorable; shim if not available
    try:
        from stephanie.scoring.scorable import Scorable
        class _S(Scorable):  # keep id-ish behavior stable
            def __init__(self, text, idx): self.text=text; self.id=idx
    except Exception:
        class _S:
            def __init__(self, text, idx): self.text=text; self.id=idx

    rows = []
    for i, text in enumerate(responses):
        scorable = _S(text, i)
        context  = {GOAL: {GOAL_TEXT: goal_text}}
        row = {}
        for model_name, scorer in scorers.items():
            try:
                bundle = scorer.score(context, scorable, dimensions)
                for dim in dimensions:
                    r = bundle.results.get(dim)
                    if r is not None:
                        row[(model_name, dim)] = float(r.score)
            except Exception as e:
                if logger: logger.log("ScoreMatrixError", {"model": model_name, "err": str(e), "i": i})
        rows.append(row)

    # Expand to full column set
    cols = pd.MultiIndex.from_product([sorted(scorers.keys()), dimensions], names=["model","dimension"])
    df = pd.DataFrame(rows, index=range(len(responses)))
    df = df.reindex(columns=cols)

    meta = {
        "n": len(responses),
        "models": sorted(scorers.keys()),
        "dimensions": list(dimensions),
        "goal_text": goal_text,
        "nulls": int(df.isna().sum().sum()),
    }
    if logger: logger.log("ScoreMatrixBuilt", meta)
    return df, meta
