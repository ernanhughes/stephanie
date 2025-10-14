# stephanie/agents/phoshrm.py
from __future__ import annotations

from stephanie.agents.base_agent import BaseAgent
from typing import Dict, List, Tuple, Iterable
import pandas as pd
import numpy as np
import re
from datetime import datetime
import json
from pathlib import Path
from typing import List, Dict, Any
from stephanie.agents.base_agent import BaseAgent
from stephanie.services.zeromodel_service import ZeroModelService
from stephanie.analysis.vpm_differential_analyzer import (
    VPMDifferentialAnalyzer,
)
from stephanie.services.scoring_service import ScoringService
from stephanie.constants import PIPELINE_RUN_ID
from stephanie.scoring.scorable import ScorableType
from stephanie.services.workers.metrics_worker import MetricsWorker
from stephanie.services.workers.vpm_worker import VPMWorker
from stephanie.utils.emit_broadcaster import EmitBroadcaster
import time
from sqlalchemy import text
import logging
from tqdm import tqdm
import asyncio
from typing import Optional
from stephanie.agents.agentic_tree_search import (
    SolutionNode,
)  # assuming you have this
from stephanie.scoring.scorable import Scorable, ScorableFactory
from stephanie.services.workers.metrics_worker import MetricsWorkerInline
from stephanie.services.workers.vpm_worker import VPMWorkerInline

from stephanie.constants import GOAL, GOAL_TEXT


class PhoshrmAgent(BaseAgent):
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        self.run_id = context.get(PIPELINE_RUN_ID)
        goal_text = context["goal"]["goal_text"]
        self._goal_text = goal_text
        self.ref_goal_id = context["goal"].get("id", 0)

        # --- inline workers ---
        scoring_cfg = self.cfg.get("metrics", {})
        scorers = scoring_cfg.get("scorers", ["sicql", "mrq", "ebt"])
        dims = scoring_cfg.get("dimensions", ["alignment", "clarity", "implementability", "novelty", "relevance"])
        self.scorer: ScoringService = self.container.get("scoring")
        self.zm: ZeroModelService = self.container.get("zeromodel")
        metrics_worker = MetricsWorkerInline(self.scorer, scorers, dims)
        self.metric_names = []
        vpm_worker = VPMWorkerInline(self.zm, self.logger)

        # --- load bands ---
        good_rows = await self._gather_runs(self.ref_goal_id)
        good = [self._make_scorable(r["response"], 1.0, "good") for r in good_rows]
        medium_rows = self.memory.embedding.search_scorables_in_similarity_band(goal_text, ScorableType.RESPONSE, 0.15, 0.80, 300)
        medium = [self._make_scorable(r["text"], r["similarity"], "medium") for r in medium_rows]
        opposite_rows = self.memory.embedding.search_unrelated_scorables(goal_text, ScorableType.RESPONSE, top_k=300)
        opposite = [self._make_scorable(r["text"], r["score"], "opposite") for r in opposite_rows]

        datasets = {"good": good, "medium": medium, "opposite": opposite}
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
