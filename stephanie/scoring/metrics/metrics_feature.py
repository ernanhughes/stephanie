# stephanie/features/metrics_feature.py
from __future__ import annotations

import logging
from typing import Dict, Any

from stephanie.scoring.metrics.base_feature import BaseFeature
from stephanie.tools.metrics_tool import MetricsTool

log = logging.getLogger(__name__)


class MetricsFeature(BaseFeature):
    name = "metrics"

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.tool = MetricsTool(cfg, memory, container, logger)

    async def apply(self, scorable, acc: Dict[str, Any], context: Dict[str, Any]):
        out = await self.tool.apply(scorable, context)
        acc.update(out)
        return acc
