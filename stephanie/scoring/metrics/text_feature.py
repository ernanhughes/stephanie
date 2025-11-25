# stephanie/features/text_feature.py
from __future__ import annotations

from stephanie.scoring.metrics.base_feature import BaseFeature
from stephanie.tools.text_tool import TextTool


class TextFeature(BaseFeature):
    """
    Wraps TextTool as a ScorableProcessor Feature.
    
    Produces:
        acc["text_features"] = { ... computed text statistics ... }
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.enabled = bool(cfg.get("enabled", True))

        # TextTool already handles entropy/sentence toggles.
        self.tool = TextTool(cfg, memory, container, logger)

    async def apply(self, scorable, acc: dict, context: dict):
        if not self.enabled:
            return acc

        # Use TextTool.apply, which updates scorable.meta
        sc_after = await self.tool.apply(scorable, context)

        # Extract & copy meta → attach to acc
        stats = sc_after.meta.get("text_stats")
        if stats:
            acc["text_features"] = dict(stats)

        return acc
