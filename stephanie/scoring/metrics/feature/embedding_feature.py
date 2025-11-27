# stephanie/scoring/metrics/feature/embedding_feature.py
from __future__ import annotations

from stephanie.scoring.metrics.feature.base_feature import BaseFeature
from stephanie.tools.embedding_tool import EmbeddingTool


class EmbeddingFeature(BaseFeature):
    name = "embedding"

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.tool = EmbeddingTool(cfg, memory, container, logger)

    async def apply(self, scorable, context):
        return await self.tool.apply(scorable, context)
