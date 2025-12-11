from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from stephanie.components.information.data import DocumentSection, PaperReferenceGraph
from stephanie.utils.document_section_parser import DocumentSectionParser

log = logging.getLogger(__name__)

SummaryFn = Callable[[str], Awaitable[Tuple[str, str]]]
EmbedFn = Callable[[str], Awaitable[Any]]


@dataclass
class SectionBuildConfig:
    # Used both as a soft target for max block size and as fallback chunk size
    chars_per_section: int = 2000
    # Minimum chars we want per merged block
    min_chars: int = 400
    # Overlap for fallback sliding-window chunking
    overlap: int = 200


class SectionBuildTask:
    """
    Hybrid section builder:

    1. Prefer semantic sections from DocumentSectionParser (unstructured-based).
    2. Merge tiny sections forward until they reach `min_chars`.
    3. Optionally summarize + embed each block.
    4. Fallback to char-based chunking if parsing fails or yields nothing.

    - summarizer: async text -> (title, summary)
    - embedder:   async text -> embedding (whatever type your store uses)
    """

    def __init__(
        self,
        cfg: Optional[SectionBuildConfig] = None,
        memory: Any = None,
        summarizer: Optional[SummaryFn] = None,
        embedder: Optional[EmbedFn] = None,
        parser: Optional[DocumentSectionParser] = None,
    ) -> None:
        self.cfg = cfg or SectionBuildConfig()
        self.summarizer = summarizer
        self.embedder = embedder
        self.parser = parser or DocumentSectionParser()

    # ------------------------------------------------------------------ #
    async def run(
        self,
        graph: PaperReferenceGraph,
        texts: Dict[str, str],  # arxiv_id -> full text
    ) -> List[DocumentSection]:
        sections: List[DocumentSection] = []

        for arxiv_id, node in graph.nodes.items():
            text = texts.get(arxiv_id, "") or ""
            if not text.strip():
                continue

            paper_sections = await self._build_sections_for_paper(
                arxiv_id=arxiv_id,
                role=node.role,
                text=text,
            )
            sections.extend(paper_sections)

        log.info("[SectionBuildTask] Built %d sections total", len(sections))
        return sections

    # ------------------------------------------------------------------ #
    async def _build_sections_for_paper(
        self,
        *,
        arxiv_id: str,
        role: str,
        text: str,
    ) -> List[DocumentSection]:
        """
        Try parser-based sections first; if that fails, fall back to chunking.
        """
        try:
            parsed_secs, scores = self.parser.parse_with_scores(text)
        except Exception as e:
            log.warning(
                "[SectionBuildTask] parser failed for %s, falling back to chunks: %s",
                arxiv_id,
                e,
            )
            return await self._build_from_chunks(arxiv_id, role, text)

        if not parsed_secs:
            log.info(
                "[SectionBuildTask] no structured sections for %s, using chunks",
                arxiv_id,
            )
            return await self._build_from_chunks(arxiv_id, role, text)

        return await self._build_from_parsed(arxiv_id, role, text, parsed_secs, scores)

    # ------------------------------------------------------------------ #
    async def _build_from_parsed(
        self,
        arxiv_id: str,
        role: str,
        full_text: str,
        parsed_secs: Dict[str, str],
        scores: Dict[str, float],
    ) -> List[DocumentSection]:
        """
        Build DocumentSection objects from parser output.

        - `parsed_secs` is an ordered dict-like mapping: heading -> text
        - We merge adjacent small sections until we reach cfg.min_chars.
        """
        cfg = self.cfg
        sections: List[DocumentSection] = []

        # Dicts preserve insertion order in modern Python, so this keeps layout.
        items = list(parsed_secs.items())

        blocks: List[Tuple[str, str]] = []  # (combined_heading, combined_text)

        cur_headings: List[str] = []
        cur_parts: List[str] = []
        cur_len = 0

        for heading, sec_text in items:
            sec_text = (sec_text or "").strip()
            if not sec_text:
                continue

            if cur_len < cfg.min_chars:
                # Keep merging into current block
                cur_headings.append(heading)
                cur_parts.append(sec_text)
                cur_len += len(sec_text)
            else:
                # Current block is big enough – flush it and start a new one
                if cur_parts:
                    blocks.append((" / ".join(cur_headings), "\n\n".join(cur_parts)))
                cur_headings = [heading]
                cur_parts = [sec_text]
                cur_len = len(sec_text)

        # Flush last block
        if cur_parts:
            blocks.append((" / ".join(cur_headings), "\n\n".join(cur_parts)))

        # Optional: if a block is *huge*, you could further chunk it here
        # based on cfg.chars_per_section, but we can leave that as a later tweak.

        running_char = 0
        for idx, (heading, block_text) in enumerate(blocks):
            block_text = block_text.strip()
            if not block_text:
                continue

            title: Optional[str] = None
            summary: Optional[str] = None
            embedding: Any = None

            # Summarize block if possible
            if self.summarizer is not None:
                try:
                    title, summary = await self.summarizer(block_text)
                except Exception as e:
                    log.warning(
                        "Summarizer failed for %s sec %d (parsed): %s",
                        arxiv_id,
                        idx,
                        e,
                    )

            # Embed summary (preferred) or raw block text
            if self.embedder is not None:
                target = summary or block_text
                try:
                    embedding = self.embedder(target)
                except Exception as e:
                    log.warning(
                        "Embedder failed for %s sec %d (parsed): %s",
                        arxiv_id,
                        idx,
                        e,
                    )

            section_id = f"{arxiv_id}::sec-{idx}"

            # We don't have perfect char offsets from the parser, but we can
            # keep a simple running cursor for relative positions.
            start_char = running_char
            end_char = start_char + len(block_text)
            running_char = end_char

            sec = DocumentSection(
                id=section_id,
                paper_arxiv_id=arxiv_id,
                paper_role=role,
                section_index=idx,
                text=block_text,
                title=title or heading,
                summary=summary,
                start_char=start_char,
                end_char=end_char,
                embedding=embedding,
            )
            sections.append(sec)

        return sections

    # ------------------------------------------------------------------ #
    async def _build_from_chunks(
        self,
        arxiv_id: str,
        role: str,
        text: str,
    ) -> List[DocumentSection]:
        """
        Original char-based sliding window implementation kept as a fallback.
        """
        cfg = self.cfg
        length = len(text)

        if length < cfg.min_chars:
            chunks = [(0, length)]
        else:
            chunks = []
            start = 0
            while start < length:
                end = min(start + cfg.chars_per_section, length)
                chunks.append((start, end))
                if end == length:
                    break
                start = end - cfg.overlap

        sections: List[DocumentSection] = []

        for idx, (start, end) in enumerate(chunks):
            chunk_text = text[start:end]

            title: Optional[str] = None
            summary: Optional[str] = None
            embedding: Any = None

            if self.summarizer is not None:
                try:
                    title, summary = await self.summarizer(chunk_text)
                except Exception as e:
                    log.warning("Summarizer failed for %s sec %d: %s", arxiv_id, idx, e)

            if self.embedder is not None:
                target = summary or chunk_text
                try:
                    embedding = await self.embedder(target)
                except Exception as e:
                    log.warning("Embedder failed for %s sec %d: %s", arxiv_id, idx, e)

            section_id = f"{arxiv_id}::sec-{idx}"

            sec = DocumentSection(
                id=section_id,
                paper_arxiv_id=arxiv_id,
                paper_role=role,
                section_index=idx,
                text=chunk_text,
                title=title,
                summary=summary,
                start_char=start,
                end_char=end,
                embedding=embedding,
            )
            sections.append(sec)

        return sections
