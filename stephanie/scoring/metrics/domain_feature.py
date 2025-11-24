# stephanie/features/domain_feature.py
from __future__ import annotations

from stephanie.features.base_feature import BaseFeature
from stephanie.tools.domain_tool import DomainTool

class DomainFeature(BaseFeature):
    name = "domain"

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.tool = DomainTool(cfg, memory, container, logger)

    async def apply(self, scorable, context):
        return await self.tool.apply(scorable, context)
