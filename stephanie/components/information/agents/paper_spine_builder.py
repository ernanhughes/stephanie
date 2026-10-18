# stephanie/components/information/agents/paper_spine_builder.py
from __future__ import annotations

import logging
from typing import Any, Dict, List

from stephanie.agents.base_agent import BaseAgent
from stephanie.components.information.data import DocumentElement
from stephanie.components.information.data import assign_page_ranges_to_semantic_sections
from stephanie.tools.pdf_tool import extract_page_texts
from stephanie.components.information.utils.spine_dump import SpineDumper

from stephanie.components.information.paper.spine.sections import needs_page_fallback, make_page_sections
from stephanie.components.information.paper.spine.attach import build_spine
from stephanie.components.information.paper.spine.dump import dump_spine
from stephanie.components.information.paper.spine.signals import emit_processing_signals
from stephanie.components.information.paper.spine.processors import (
    ProcessorResult,
    SmolDoclingProcessor,
)
from pathlib import Path

log = logging.getLogger(__name__)


class PaperSpineBuilderAgent(BaseAgent):
    """
    Thin orchestrator:
      - resolve identity
      - get/normalize sections
      - run processors (docling)
      - fallback to per-page sections if needed
      - attach elements -> spine
      - dump + signals
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg=cfg, memory=memory, container=container, logger=logger)
        from stephanie.tools.smol_docling_tool import SmolDoclingTool

        try:
            self._docling_tool = SmolDoclingTool(cfg=getattr(cfg, "docling", {}), memory=memory, container=container, logger=logger)
            self._processors = [SmolDoclingProcessor(self._docling_tool)]
        except Exception as exc:
            # Docling is an optional enrichment processor (it needs a vision
            # transformers build). Without it the spine still builds from the
            # semantic sections and the page fallback below, so degrade instead
            # of failing the whole pipeline.
            log.warning(
                "PaperSpineBuilderAgent: docling processor unavailable (%s); "
                "continuing without figure/table element extraction",
                exc,
            )
            self._docling_tool = None
            self._processors = []
        self._spine_dumper = SpineDumper(cfg=getattr(cfg, "dump", {}), memory=memory, container=container, logger=logger)

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        arxiv_id = context.get("arxiv_id")
        pdf_path = context.get("paper_pdf_path")
        if pdf_path:
            pdf_path = Path(pdf_path)

        sections = context.get("paper_sections") or []
        page_texts: Dict[int, str] = {}
        if pdf_path:
            try:
                page_texts = extract_page_texts(str(pdf_path))
            except Exception as exc:
                # The PDF may not be cached locally (data/ is not versioned and
                # the graph may not have re-imported it). Page ranges and figure
                # attachment are enrichments; the spine/blog still build from
                # the semantic sections.
                log.warning(
                    "PaperSpineBuilderAgent: page text extraction unavailable for %s (%s); "
                    "continuing without page ranges",
                    pdf_path,
                    exc,
                )
        assign_page_ranges_to_semantic_sections(sections, page_texts)

        elements: List[DocumentElement] = context.get("paper_elements") or []
        proc_results: List[ProcessorResult] = []

        for p in self._processors:
            elements, r = await p.run(arxiv_id=arxiv_id, pdf_path=pdf_path, elements=elements, context=context)
            proc_results.append(r)

        spine_sections = context.get("paper_spine_sections") or list(sections)
        if needs_page_fallback(spine_sections):
            num_pages = max(page_texts.keys(), default=0)
            spine_sections = make_page_sections(
                arxiv_id=arxiv_id,
                paper_role="root",
                num_pages=num_pages,
                page_text_by_page=page_texts,
            )

        spine = build_spine(spine_sections, elements)
        context.update(
            {
                "paper_sections": sections,
                "paper_spine_sections": spine_sections,
                "paper_elements": elements,
                "paper_spine": spine,
            }
        )

        try:
            dumped = dump_spine(
                dumper=self._spine_dumper,
                arxiv_id=arxiv_id,
                sections=spine_sections,
                elements=elements,
                spine=spine,
                proc_results=proc_results,
            )
            context.setdefault("paper_processing_signals", {}).setdefault("spine_dump", {})["files"] = dumped
        except Exception:
            log.exception("PaperSpineBuilderAgent: dump failed")

        emit_processing_signals(context=context, proc_results=proc_results, sections=spine_sections, elements=elements)
        return context
