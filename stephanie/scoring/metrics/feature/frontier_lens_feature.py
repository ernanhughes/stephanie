# stephanie/scoring/metrics/feature/frontier_lens_feature.py
from __future__ import annotations

import logging
from typing import Any, Dict

from stephanie.scoring.metrics.feature.base_feature import BaseFeature
from stephanie.tools.frontier_lens_tool import FrontierLensTool

log = logging.getLogger(__name__)


class FrontierLensFeature(BaseFeature):
    """
    Lightweight per-row FrontierLens feature.

    Runs AFTER metrics are computed and attaches a small summary:
        - frontier_lens_report  (frontier metric + value + metric count)

    Full episode-level FrontierLens analysis (report + features + VPM)
    is handled by FrontierLensGroupFeature or by dedicated Critic/Nexus
    agents using FrontierLensTool.apply_rows().
    """

    # New, stable name for config + registry
    name = "frontier_lens_basic"

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.tool = FrontierLensTool(cfg, memory, container, logger)
        self.enabled = bool(cfg.get("enabled", True))

    async def apply(
        self,
        scorable,
        acc: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Expects:
            acc["metrics_columns"]
            acc["metrics_values"]

        Produces (if metrics exist):
            acc["frontier_lens_report"] : Dict[str, Any]
        """
        if not self.enabled:
            return acc

        cols = acc.get("metrics_columns") or []
        vals = acc.get("metrics_values") or []

        if not cols or not vals:
            log.debug("[FrontierLensFeature] no metrics for row; skipping")
            return acc

        # Delegate to the tool’s per-row FrontierLens summary
        acc = await self.tool.apply(scorable, acc, context)
        return acc
