from stephanie.scoring.metrics.feature.base_feature import BaseFeature
from stephanie.tools.summarization_tool import SummarizationTool


class SectionSummarizationFeature(BaseFeature):
    name = "section_summarizer"

    def __init__(self, cfg, memory, container, logger):
        self.summarizer = SummarizationTool(
            cfg["tools"]["section_summarizer"],
            memory,
            container,
            logger,
        )

    async def process_section(self, scorable, context):
        scorable = await self.summarizer.apply(scorable, context)
        return scorable
