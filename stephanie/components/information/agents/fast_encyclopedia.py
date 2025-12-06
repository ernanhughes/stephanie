from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List

import logging

from stephanie.agents.base_agent import BaseAgent
from stephanie.components.information.models import ReasonedSection, ReasonedBlogResult

log = logging.getLogger(__name__)




# ---------------------------------------------------------------------
# Fast agent: no Memento, no extra LLM calls
# ---------------------------------------------------------------------


class FastEncyclopediaAgent(BaseAgent):
    """
    Stage 2: fast, zero-extra-LLM pass for the AI Encyclopedia.

    Runs after InformationIngestAgent.

    For each ingested paper (for now: first one only), it:

      - Resolves sections from:
          1) document_sections table (if available)
          2) doc_meta["sections"]
          3) falls back to a single 'Overview' section from blog_markdown
      - Builds a simple outline (one entry per section)
      - Assembles a markdown blog with:

            # <paper_title>
            (optional intro from blog_markdown)
            ## <section title>
            <section text>

    This is meant to be very fast and robust. No MementoAgent, no
    additional LLM calls: it reuses what ingest already produced.
    """

    def __init__(self, cfg: Dict[str, Any], memory, container, logger) -> None:
        super().__init__(cfg, memory, container, logger)

        # How many sections to keep at most
        self.max_sections: int = int(cfg.get("max_sections", 6))

        # Minimum length to consider a section "real"
        self.section_min_len: int = int(cfg.get("section_min_len", 80))

    # ------------------------------------------------------------------
    # Pipeline entrypoint
    # ------------------------------------------------------------------

    async def run(self, context: Dict[str, Any]) ->  Dict[str, Any]:
        ingest_block = context.get("information_ingest") or {}
        if ingest_block.get("status") != "ok":
            log.warning(
                "FastEncyclopediaAgent: information_ingest.status != 'ok', skipping."
            )
            context["reasoned_blog"] = {
                "status": "skipped",
                "reason": "information_ingest not ready",
            }
            return context

        documents = ingest_block.get("documents") or []
        if not documents:
            log.warning(
                "FastEncyclopediaAgent: no documents found in information_ingest."
            )
            context["reasoned_blog"] = {
                "status": "no_documents",
                "reason": "no documents to enhance",
            }
            return context

        # For now, handle the first document only.
        doc_meta = documents[0]
        result = self._process_single_document(doc_meta, context)

        context["reasoned_blog"] = {
            "status": "ok",
            "result": asdict(result),
        }
        return context

    # ------------------------------------------------------------------
    # Core per-document logic
    # ------------------------------------------------------------------

    def _process_single_document(
        self, doc_meta: Dict[str, Any], context: Dict[str, Any]
    ) -> ReasonedBlogResult:
        paper_id = str(doc_meta.get("document_id") or doc_meta.get("doc_id"))
        paper_title = doc_meta.get("title") or f"Paper {paper_id}"
        memcube_id = doc_meta.get("memcube_id")
        casebook_id = doc_meta.get("casebook_id")

        log.info(
            "FastEncyclopediaAgent: processing paper '%s' (id=%s, memcube=%s, casebook=%s)",
            paper_title,
            paper_id,
            memcube_id,
            casebook_id,
        )

        # Resolve sections with a simple policy
        sections = self._resolve_sections_for_paper(doc_meta, context)
        selected = self._select_sections(sections)

        reasoned_sections: List[ReasonedSection] = []
        for idx, sec in enumerate(selected):
            sec_id = str(
                sec.get("id")
                or sec.get("section_id")
                or f"s{idx}"
            )
            sec_title = (
                sec.get("title")
                or sec.get("section_name")
                or f"Section {idx + 1}"
            )
            sec_text = (
                sec.get("text")
                or sec.get("content")
                or ""
            )

            reasoned_sections.append(
                ReasonedSection(
                    section_id=sec_id,
                    title=sec_title,
                    text=sec_text,
                    candidates=[],
                    casebook_ref=None,
                    meta={
                        "paper_id": paper_id,
                        "source": "fast_encyclopedia",
                    },
                )
            )

        outline = self._build_outline(reasoned_sections)
        full_text = self._build_full_text(paper_title, doc_meta, reasoned_sections)

        meta = {
            "source_paper_id": paper_id,
            "memcube_id": memcube_id,
            "casebook_id": casebook_id,
            "agent": "FastEncyclopediaAgent",
        }

        return ReasonedBlogResult(
            outline=outline,
            sections=reasoned_sections,
            full_text=full_text,
            meta=meta,
        )

    # ------------------------------------------------------------------
    # Section resolution & selection
    # ------------------------------------------------------------------

    def _resolve_sections_for_paper(
        self,
        doc_meta: Dict[str, Any],
        context: Dict[str, Any], 
    ) -> List[Dict[str, Any]]:
        """
        Resolve a list of sections for this paper.

        Priority:
          1) document_sections table (if memory.document_sections is available)
          2) doc_meta["sections"] from ingest
          3) fallback: a single 'Overview' section from blog_markdown
        """
        doc_id = doc_meta.get("document_id") or doc_meta.get("doc_id")
        sections: List[Dict[str, Any]] = []

        # 1) document_sections from memory, if present
        if doc_id is not None:
            raw_sections = self.memory.document_sections.get_by_document(int(doc_id))
            if raw_sections:
                sections = [s.to_dict() for s in raw_sections]
        if sections:
            # Normalize keys a bit
            normalized: List[Dict[str, Any]] = []
            for idx, s in enumerate(sections):
                text = s.get("text") or s.get("content") or ""
                if len(text.strip()) < self.section_min_len:
                    continue
                normalized.append(
                    {
                        "id": s.get("id") or f"s{idx}",
                        "title": s.get("title") or s.get("section_name") or "",
                        "text": text,
                    }
                )
            if normalized:
                return normalized

        # 2) doc_meta["sections"] from ingest
        meta_sections = doc_meta.get("sections") or []
        if meta_sections:
            cleaned: List[Dict[str, Any]] = []
            for idx, s in enumerate(meta_sections):
                text = s.get("text") or s.get("content") or ""
                if len(text.strip()) < self.section_min_len:
                    continue
                cleaned.append(
                    {
                        "id": s.get("id") or s.get("section_id") or f"s{idx}",
                        "title": s.get("title") or s.get("section_name") or "",
                        "text": text,
                    }
                )
            if cleaned:
                return cleaned

        # 3) Fallback: one big 'Overview' section from blog_markdown or nothing
        blog_md = doc_meta.get("blog_markdown") or ""
        if blog_md.strip():
            return [
                {
                    "id": "s0",
                    "title": "Overview",
                    "text": blog_md.strip(),
                }
            ]

        # Absolute worst-case: empty, caller will handle it
        return []

    def _select_sections(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Simple selection: first N sections, after filtering by length.
        """
        if not sections:
            return []
        return sections[: self.max_sections]

    # ------------------------------------------------------------------
    # Outline & full text assembly
    # ------------------------------------------------------------------

    def _build_outline(self, sections: List[ReasonedSection]) -> List[Dict[str, Any]]:
        """
        Outline == ordered list of section titles.
        """
        outline: List[Dict[str, Any]] = []
        for sec in sections:
            outline.append(
                {
                    "id": sec.section_id,
                    "title": sec.title,
                    "source": "fast_encyclopedia",
                }
            )
        return outline

    def _build_full_text(
        self,
        paper_title: str,
        doc_meta: Dict[str, Any],
        sections: List[ReasonedSection],
    ) -> str:
        """
        Build a simple markdown document:

            # <paper_title>

            ## <section title>
            <section text>
        """
        chunks: List[str] = []
        # H1 title
        chunks.append(f"# {paper_title}\n")

        for sec in sections:
            title = sec.title or "Section"
            text = sec.text or ""
            chunks.append(f"## {title}\n\n{text}")

        return "\n\n".join(chunks)
