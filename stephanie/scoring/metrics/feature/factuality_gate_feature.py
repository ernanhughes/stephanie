# stephanie/scoring/metrics/feature/factuality_gate_feature.py
from __future__ import annotations

from stephanie.scoring.metrics.feature.base_feature import BaseFeature
from stephanie.tools.factuality_gate_tool import FactualityGateTool


class FactualityGateFeature(BaseFeature):
    """
    Feature wrapper around FactualityGateTool.

    Use this on the *candidate blog scorable* in your scoring pipeline.
    Expects `context["source_text"]` (or similar) to contain the source paper.
    """

    name = "factuality_gate"

    def __init__(self, cfg, memory, container, logger):
        self.tool = FactualityGateTool(
            cfg["tools"]["factuality_gate"],
            memory,
            container,
            logger,
        )

    async def process_section(self, scorable, context):
        scorable = await self.tool.apply(scorable, context)
        return scorable
