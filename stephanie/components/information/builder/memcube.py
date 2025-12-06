# stephanie/components/information/memcube_builder.py
from __future__ import annotations

from typing import Any, Dict, List

from stephanie.components.information.data import InfoSection
from stephanie.memory.casebook_store import CaseBookStore
from stephanie.memory.memcube_store import MemCubeStore
from stephanie.models.memcube import MemCubeORM
from stephanie.utils.date_utils import iso_now
from stephanie.utils.hash_utils import hash_text


class MemCubeBuilder:
    """
    Builds an 'Information MemCube' from a CaseBook and its Cases.

    Responsibilities:
      - For each Case: generate or reuse a section body (markdown)
      - Stitch sections into MemCube.content (blog-postable)
      - Populate MemCube.extra_data with structured sections etc.
    """

    def __init__(
        self,
        memcube_store: MemCubeStore,
        casebook_store: CaseBookStore,
        logger,
        llm_client,
    ) -> None:
        self.memcube_store = memcube_store
        self.casebook_store = casebook_store
        self.logger = logger
        self.llm = llm_client

    async def build_information_memcube(
        self,
        topic: str,
        target: str,
        casebook_id: int,
        case_ids: List[int],
        source_profile: str,
    ) -> MemCubeORM:
        # Load cases
        cb = self.casebook_store.get_casebook(casebook_id, include_cases=True)
        # casebook.cases will be loaded via relationship
        cases = [c for c in cb.cases if c.id in case_ids]

        # 1) Build sections
        sections: list[InfoSection] = []
        for case in cases:
            sec = await self._build_section_for_case(topic, case)
            sections.append(sec)

        # 2) Build markdown content
        content = await self._stitch_content(topic, sections)

        # 3) Build extra_data payload
        extra_data: Dict[str, Any] = {
            "topic": topic,
            "topic_slug": self._slugify(topic),
            "target": target,
            "casebook_id": casebook_id,
            "source_profile": source_profile,
            "sections": [
                {
                    "title": s.title,
                    "description": s.description,
                    "content": s.content,
                    "case_id": s.case_id,
                    "dynamic_scorable_ids": s.dynamic_scorable_ids,
                }
                for s in sections
            ],
            "references": [],  # TODO: fill from case.scorables if you want
            "related_memcubes": [],
            "tags": ["info", "memcube", "bloggable"],
            "created_by": "information_orchestrator_v1",
            "created_at_iso": iso_now(),
            "last_updated_iso": iso_now(),
        }

        # 4) Create / upsert MemCube
        data = {
            "scorable_id": self._stable_int_id(topic),
            "scorable_type": "info",
            "dimension": "topic",
            "version": "v1",
            "content": content,
            "refined_content": None,
            "original_score": None,
            "refined_score": None,
            "source": "information_orchestrator",
            "model": None,
            "priority": 5,
            "sensitivity": "public",
            "ttl": None,
            "usage_count": 0,
            "extra_data": extra_data,
        }
        cube = self.memcube_store.upsert(data, merge_extra=True)

        self.logger.log(
            "InfoMemCubeBuilt",
            {
                "topic": topic,
                "target": target,
                "memcube_id": cube.id,
                "casebook_id": casebook_id,
                "sections": len(sections),
            },
        )
        return cube

    # -------------------- helpers --------------------

    async def _build_section_for_case(self, topic: str, case) -> InfoSection:
        """
        Generate one section for a Case.

        v1: single LLM call per case; no DynamicScorable persistence wired in yet.
        You can later attach DynamicScorables here.
        """
        case_meta = case.meta or {}
        section_title = case_meta.get("section_title") or "Section"
        # You can also derive description via a small LLM call
        description = case_meta.get("description") or ""

        # Build source snippets
        snippets = []
        for cs in case.scorables:
            text = (cs.meta or {}).get("text") or cs.meta or ""
            if not text:
                continue
            snippets.append(text)

        joined = "\n\n".join(snippets[:10])

        prompt = f"""
Topic: {topic}
Section: {section_title}

You are writing a section for an AI encyclopedia / blog.
Using ONLY the following snippets as evidence, write a clear, faithful markdown section.

Snippets:
{joined}
"""
        section_body = await self.llm.generate(prompt)
        section_body = section_body.strip()

        return InfoSection(
            title=section_title,
            description=description,
            content=section_body,
            case_id=case.id,
            dynamic_scorable_ids=[],  # fill when you wire DynamicScorables
        )

    async def _stitch_content(
        self, topic: str, sections: List[InfoSection]
    ) -> str:
        """
        Assemble the full markdown content from sections.
        Also ask the LLM for a short summary at the top.
        """
        prompt = f"""
Write a 2–3 sentence abstract summary for an article titled:
"{topic}"

Do not include headings, just the abstract text.
"""
        abstract = await self.llm.generate(prompt)
        abstract = abstract.strip()

        lines = [f"# {topic}", "", "## Summary", "", abstract, ""]
        for sec in sections:
            lines.append(f"## {sec.title}")
            lines.append("")
            lines.append(sec.content)
            lines.append("")

        return "\n".join(lines)

    def _slugify(self, text: str) -> str:
        import re

        text = text.strip().lower()
        text = re.sub(r"[^a-z0-9]+", "-", text)
        text = re.sub(r"-+", "-", text).strip("-")
        return text or "topic"

    def _stable_int_id(self, text: str) -> int:
        """
        Simple stable hash → int for scorable_id.
        """

        h = hash_text(text)
        return int(h[:12], 16)
