# stephanie/scoring/metrics/embedding_feature.py
from __future__ import annotations

from stephanie.scoring.metrics.feature import Feature
from stephanie.tools.embedding_tool import EmbeddingTool

class EmbeddingFeature(Feature):
    name = "embedding"

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.tool = EmbeddingTool(cfg, memory, container, logger)

    async def apply(self, scorable, context):
        return await self.tool.apply(scorable, context)
