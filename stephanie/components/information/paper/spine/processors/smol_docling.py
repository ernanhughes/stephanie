# stephanie/components/information/paper/spine/processors/smol_docling.py
from __future__ import annotations

from typing import Dict, List

from .base import BaseSpineProcessor, ProcessorResult
from stephanie.tools.smol_docling_tool import SmolDoclingTool


from stephanie.scoring.scorable import Scorable, ScorableType  # adjust import to your actual path


class SmolDoclingProcessor(BaseSpineProcessor):
    name = "smol_docling"

    def __init__(self, tool: SmolDoclingTool):
        self.tool = tool

    async def run(self, *, arxiv_id, pdf_path, elements, context):
        # Create a minimal scorable so the tool can resolve paper_id and attach meta
        sc = Scorable(
            id=str(arxiv_id),
            target_type=ScorableType.PAPER,  # or just "paper"
            text="",
            meta={},
        )

        sc = await self.tool.apply(sc, context)  # renders PDF -> doctags -> sc.meta["docling"]["pages"]

        pages = (sc.meta.get("docling") or {}).get("pages") or []
        if not pages:
            context["paper_spine_sections"] = []
            return elements, ProcessorResult(name=self.name, ran=True)

        sections = self.tool.build_semantic_sections(
            arxiv_id=str(arxiv_id),
            paper_role="root",                 # or pass from context if you track it
            pages=pages,
            run_id=context.get("run_id"),
        )

        context["paper_spine_sections"] = sections
        return elements, ProcessorResult(name=self.name, ran=True)
