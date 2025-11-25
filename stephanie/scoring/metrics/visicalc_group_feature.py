# stephanie/scoring/metrics/visicalc_group_feature.py
from __future__ import annotations
from typing import Any, Dict, List
from stephanie.scoring.metrics.base_group_feature import BaseGroupFeature
from stephanie.tools.visicalc_group_tool import VisiCalcGroupTool

class VisiCalcGroupFeature(BaseGroupFeature):
    name = "visicalc_group"
    requires = ["metric_filter"]  # optional: ensure filtered metrics exist first

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.tool = VisiCalcGroupTool(cfg, memory, container, logger)

    async def apply(self, rows: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self.enabled:
            return rows
        return await self.tool.apply(rows, context)
