# stephanie/components/information/agents/paper_pipeline.py
from __future__ import annotations

import logging
from typing import Any, Dict

from stephanie.agents.base_agent import BaseAgent
from stephanie.tools.paper_import_tool import PaperImportTool
from stephanie.components.information.paper.pipeline.pipeline_runner import PaperPipelineRunner

log = logging.getLogger(__name__)


class PaperPipelineAgent(BaseAgent):
    """ I
    High-level agent that runs the full paper pipeline:

        arxiv_id -> graph -> sections -> cross-paper links

    Expected context input keys:
        - "arxiv_id" (or "paper_arxiv_id")
        - optional: "max_refs", "max_similar"

    It writes:
        - context["paper_graph"]
        - context["paper_sections"]
        - context["section_matches"]
        - context["concept_clusters"]

    NOTE: For now this agent is arXiv-centric. If you want to iterate over a
    directory of local PDFs, you can:
        - treat each PDF as having an "arxiv_id" equal to its stem, or
        - build a small wrapper agent that uses PaperImportTask + SectionBuildTask
          directly for non-arxiv PDFs.
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(
            cfg=cfg, memory=memory, container=container, logger=logger
        )
        self.import_tool = PaperImportTool(
            self.cfg, self.memory, self.container, self.logger
        )
        self.runner = PaperPipelineRunner(
            cfg=self.cfg,
            memory=self.memory,
            container=self.container,
            logger=self.logger,
            import_tool=self.import_tool,
        )

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        context = await self.runner.run(context)
        return context
