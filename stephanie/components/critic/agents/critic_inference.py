# stephanie/components/critic/agents/critic_inference.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from stephanie.agents.base_agent import BaseAgent
from stephanie.components.critic.model.critic_model import CriticModel
from stephanie.constants import GOAL, GOAL_TEXT
from stephanie.data.score_bundle import ScoreBundle
from stephanie.data.score_result import ScoreResult
from stephanie.scoring.scorable import Scorable

log = logging.getLogger(__name__)


class CriticInferenceAgent(BaseAgent):
    """
    Production-grade scorer for the Tiny Critic model.

    This mirrors Stephanie’s other model scorers (SVMScorer, MRQInferenceAgent):
      - Loads model + meta (CriticModel)
      - Prepares features via scorable_processor
      - ScoreBundle output (dim="critic")
      - Supports batch scoring
      - Optional explanations
    """

    def __init__(self, cfg: dict, memory, container, logger):
        super().__init__(cfg, memory, container, logger)

        self.dimension = cfg.get("dimension", "critic")
        self.model_path = Path(cfg.get("model_path", "models/critic.joblib"))
        self.meta_path = Path(cfg.get("meta_path", "models/critic.meta.json"))
        self.include_explanations = bool(cfg.get("include_explanations", False))

        # Load Tiny Critic
        self.model = CriticModel.load(
            model_path=self.model_path,
            meta_path=self.meta_path,
        )

        logger.log("CriticInferenceAgentLoaded", {
            "dimension": self.dimension,
            "model_path": str(self.model_path),
            "meta_path": str(self.meta_path),
            "features": len(self.model.meta.feature_names),
        })

    # ------------------------------------------------------------
    async def run(self, context: dict) -> dict:
        """
        Input:
            context["scorables"] = list of Scorable
            context["scorable_features"] = list of {feature_name: value}

        Output:
            context["critic_scores"] = ScoreBundle
        """
        scorables: List[Scorable] = context.get(self.input_key) or []
        rows = context.get("scorable_features") or []

        if not scorables or not rows:
            log.warning(f"{self.name}: No scorables or no feature rows.")
            context[self.output_key] = ScoreBundle(results={})
            return context

        # Align rows → Tiny Critic probabilities
        probs = self.model.score_batch(rows)  # returns array of P(positive)

        # Construct ScoreBundle
        results = {}

        for scorable, p, row in zip(scorables, probs, rows):
            # “critic” produces a 0–100 score by convention (optional)
            score01 = float(p)
            score100 = round(score01 * 100, 3)

            rationale = f"Critic p={score01:.4f}"

            if self.include_explanations:
                top = self.model.explain_one(row)
                rationale += " | " + ", ".join(
                    f"{k}:{round(v,3)}" for k, v in top
                )

            results[scorable.id] = ScoreResult(
                dimension=self.dimension,
                score=score100,
                rationale=rationale,
                weight=1.0,
                source="critic",
            )

        bundle = ScoreBundle(results=results)
        context[self.output_key] = bundle

        # Log for telemetry
        self.logger.log("CriticScoresComputed", {
            "dimension": self.dimension,
            "count": len(results),
        })

        return context
