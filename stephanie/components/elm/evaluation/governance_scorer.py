# stephanie/scoring/governance/governance_scorer.py

from __future__ import annotations

import logging
from typing import Any, Dict, List, Protocol

from stephanie.data.score_bundle import ScoreBundle
from stephanie.data.score_result import ScoreResult
from stephanie.scoring.scorer.base_scorer import BaseScorer
from stephanie.scoring.scorable import Scorable

log = logging.getLogger(__name__)


# ----------------------------------------
# Provider Protocol
# ----------------------------------------

class GovernanceProvider(Protocol):
    def compute(
        self,
        context: dict,
        scorable: Scorable,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Returns:
            {
                "dimension_name": {
                    "score": float,
                    "rationale": str,
                    "attributes": dict
                }
            }
        """
        ...


# ----------------------------------------
# GovernanceScorer
# ----------------------------------------

class GovernanceScorer(BaseScorer):
    """
    Governance layer implemented as a standard Stephanie scorer.

    - Each provider emits dimensions
    - Dimensions become ScoreResult objects
    - Attributes store diagnostics
    - No custom reward vector
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)

        self.model_type = "governance"
        self.providers: List[GovernanceProvider] = cfg.get("providers", [])

        if not self.providers:
            log.warning("GovernanceScorer initialized with no providers")

    # ----------------------------------------
    # Core Scoring
    # ----------------------------------------

    def _score_core(
        self,
        context: dict,
        scorable: Scorable,
        dimensions: List[str]
    ) -> ScoreBundle:

        results: Dict[str, ScoreResult] = {}

        for provider in self.providers:
            try:
                provider_output = provider.compute(
                    context=context,
                    scorable=scorable,
                )

                for dim, payload in provider_output.items():

                    if dimensions and dim not in dimensions:
                        continue

                    score = float(payload.get("score", 0.0))
                    rationale = payload.get("rationale", "")
                    attributes = payload.get("attributes", {})

                    results[dim] = ScoreResult(
                        dimension=dim,
                        score=score,
                        weight=1.0,
                        rationale=rationale,
                        source=self.model_type,
                        attributes=attributes,
                    )

            except Exception as e:
                log.error(f"Governance provider failure: {e}")
                self.logger.log(
                    "GovernanceProviderError",
                    {"error": str(e)}
                )

        return ScoreBundle(results=results)
