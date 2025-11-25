# stephanie/components/scorable_processor/features/visicalc_basic_feature.py

from __future__ import annotations
import logging
from typing import Dict, Any

from stephanie.scoring.metrics.base_feature import BaseFeature
from stephanie.tools.visicalc_tool import VisiCalcTool

log = logging.getLogger(__name__)


class VisiCalcBasicFeature(BaseFeature):
    """
    Lightweight per-row VisiCalc feature.
    This runs AFTER metrics are computed and attaches:
        - visicalc_report  (frontier score summary)
        - visicalc_quality (optional future use)

    Full VisiCalc batch computation is handled by Critic / Nexus agents,
    not inside the ScorableProcessor.
    """

    name = "visicalc_basic"

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.tool = VisiCalcTool(cfg, memory, container, logger)
        self.enabled = bool(cfg.get("enabled", True))

    # -----------------------------------------------------
    async def apply(
        self,
        scorable,
        acc: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Apply VisiCalc lightweight logic for a single row.

        Expects:
            acc["metrics_columns"]
            acc["metrics_values"]

        Produces:
            acc["visicalc_report"]
        """

        if not self.enabled:
            return acc

        cols = acc.get("metrics_columns") or []
        vals = acc.get("metrics_values") or []

        if not cols or not vals:
            log.debug("[VisiCalcBasicFeature] no metrics for row; skipping")
            return acc

        # Delegate to the tool’s per-row VisiCalc
        acc = await self.tool.apply(scorable, acc, context)
        return acc
