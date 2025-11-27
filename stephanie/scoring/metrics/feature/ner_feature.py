# stephanie/scoring/metrics/feature/ner_feature.py
from __future__ import annotations

from stephanie.scoring.metrics.feature.base_feature import BaseFeature
from stephanie.tools.ner_tool import NerTool


class NerFeature(BaseFeature):
    name = "ner"

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.tool = NerTool(cfg, memory, container, logger)

    async def apply(self, scorable, context):
        return await self.tool.apply(scorable, context)
