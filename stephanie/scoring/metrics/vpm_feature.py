# stephanie/features/vpm_feature.py
from __future__ import annotations

import logging
from typing import Any, Dict

from stephanie.scoring.metrics.base_feature import BaseFeature
from stephanie.tools.vpm_tool import VpmTool

log = logging.getLogger(__name__)


class VpmFeature(BaseFeature):
    name = "vpm"

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.tool = VpmTool(cfg, memory, container, logger)

    async def apply(
        self,
        scorable,
        context: Dict[str, Any],
    ):
        # VPM depends on metrics stored in acc
        metrics_columns = scorable.meta.get("metrics_columns") or []
        metrics_values = scorable.meta.get("metrics_values") or []

        out = await self.tool.apply(
            scorable,
            metrics_columns=metrics_columns,
            metrics_values=metrics_values,
        )
        scorable.meta.update(out)
        return scorable
