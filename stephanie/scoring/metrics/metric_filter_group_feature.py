# stephanie/scoring/metrics/metric_filter_group_feature.py
from __future__ import annotations
from typing import Any, Dict, List
from stephanie.scoring.metrics.base_group_feature import BaseGroupFeature
from stephanie.tools.metric_filter_group_tool import MetricFilterGroupTool

class MetricFilterGroupFeature(BaseGroupFeature):
    name = "metric_filter"

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.tool = MetricFilterGroupTool(cfg, memory, container, logger)

    async def apply(self, rows: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self.enabled:
            return rows
        return await self.tool.apply(rows, context)
