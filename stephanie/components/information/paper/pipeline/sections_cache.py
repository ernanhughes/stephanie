# stephanie/components/information/paper/pipeline/sections_cache.py
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from stephanie.components.information.data import PaperSection

log = logging.getLogger(__name__)


@dataclass
class PaperSectionsCacheConfig:
    enabled: bool = True


def _sha256_text(text: str) -> str:
    norm = re.sub(r"\s+", " ", (text or "").strip())
    return hashlib.sha256(norm.encode("utf-8", "ignore")).hexdigest()


def _section_id(paper_id: str, index: int) -> str:
    """Stable, unique, source-attributed section identity.

    Convention (see ``PaperSection`` docstring): ``{paper_arxiv_id}::sec-{index}``.
    The blog generator parses this prefix to attribute a section to its source
    paper, so the prefix MUST be the section's own paper -- not the root paper.
    """
    return f"{paper_id}::sec-{index}"


class PaperSectionsCache:
    """
    Loads and persists PaperSection objects via PaperStore/ORM.

    Lifecycle rules (fixed):
      * ``maybe_load`` returns the stored sections for ONE source paper, or None.
        Callers that must regenerate pass ``force_rebuild=True`` to bypass it.
      * ``persist`` writes every section under its OWN source paper
        (``paper_arxiv_id``) with a stable unique id and the ORM's required
        fields. With ``replace=True`` the previous generation for that paper is
        removed, so a rebuild can never leave two generations active.
    """

    def __init__(self, *, cfg: Dict[str, Any], paper_store: Any, logger: Any):
        self.cfg = cfg
        self.paper_store = paper_store
        self.logger = logger

        self.enabled = bool(cfg.get("sections_cache_enabled", True))

    def maybe_load(
        self,
        arxiv_id: str,
        role: str = "reference",
        *,
        force_rebuild: bool = False,
    ) -> Optional[List[PaperSection]]:
        if not self.enabled or force_rebuild:
            return None

        arxiv_id = str(arxiv_id)
        try:
            rows = self.paper_store.get_sections_for_paper(paper_id=arxiv_id)
            if not rows:
                return None
            return self._sections_from_orm(rows)
        except Exception:
            log.warning("PaperSectionsCache: load failed for %s", arxiv_id, exc_info=True)
            return None

    def persist(
        self,
        *,
        arxiv_id: str,
        sections: List[PaperSection],
        replace: bool = False,
    ) -> int:
        """Persist sections grouped by their true source paper.

        Returns the number of rows written (0 on failure or when disabled).
        ``replace=True`` replaces the stored generation for each source paper.
        """
        if not self.enabled or not sections:
            return 0

        arxiv_id = str(arxiv_id)
        grouped = self._group_by_source(arxiv_id=arxiv_id, sections=sections)

        written = 0
        for paper_id, secs in grouped.items():
            rows = self._rows_for_paper(paper_id=paper_id, sections=secs)
            if not rows:
                continue
            try:
                if replace:
                    self.paper_store.replace_sections_for_paper(paper_id, rows)
                else:
                    for row in rows:
                        self.paper_store.upsert_section(row)
                written += len(rows)
            except Exception:
                # Keep non-fatal, but do NOT hide it: a silent failure here was
                # how the cache ended up with a stale/duplicated generation.
                log.warning(
                    "PaperSectionsCache: persist failed for paper_id=%s (replace=%s, rows=%d)",
                    paper_id,
                    replace,
                    len(rows),
                    exc_info=True,
                )
        return written

    # ----------------------- internal helpers ------------------------

    def _group_by_source(
        self, *, arxiv_id: str, sections: List[PaperSection]
    ) -> Dict[str, List[PaperSection]]:
        grouped: Dict[str, List[PaperSection]] = {}
        for sec in sections:
            pid = getattr(sec, "paper_arxiv_id", None) or getattr(sec, "paper_id", None) or arxiv_id
            grouped.setdefault(str(pid), []).append(sec)
        return grouped

    def _rows_for_paper(
        self, *, paper_id: str, sections: List[PaperSection]
    ) -> List[Dict[str, Any]]:
        ordered = sorted(
            sections, key=lambda s: int(getattr(s, "section_index", 0) or 0)
        )
        rows: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for idx, sec in enumerate(ordered):
            raw_index = getattr(sec, "section_index", None)
            index = int(raw_index) if raw_index is not None else idx
            row_id = _section_id(paper_id, index)
            if row_id in seen:
                # Same paper, same index: keep the first, never emit a
                # colliding primary key.
                log.warning(
                    "PaperSectionsCache: dropping duplicate section index %s for %s",
                    index,
                    paper_id,
                )
                continue
            seen.add(row_id)

            text = getattr(sec, "text", "") or ""
            rows.append(
                {
                    "id": row_id,
                    "paper_id": paper_id,
                    "run_id": None,
                    "parent_id": None,
                    "level": 0,
                    "path": str(index),  # NOT NULL in PaperSectionORM
                    "role": str(getattr(sec, "role", None) or getattr(sec, "paper_role", None) or "reference"),
                    "section_index": index,
                    "start_char": getattr(sec, "start_char", None),
                    "end_char": getattr(sec, "end_char", None),
                    "start_page": getattr(sec, "start_page", None),
                    "end_page": getattr(sec, "end_page", None),
                    "text": text,
                    "title": getattr(sec, "title", None),
                    "summary": getattr(sec, "summary", None),
                    "content_hash": _sha256_text(text),
                    "token_count": len(text.split()),
                    "meta": getattr(sec, "meta", None) or {},
                }
            )
        return rows

    def _sections_from_orm(self, rows: List[Any]) -> List[PaperSection]:
        sections: List[PaperSection] = []
        for idx, r in enumerate(rows):
            raw_index = getattr(r, "section_index", None)
            sections.append(
                PaperSection(
                    section_id=str(getattr(r, "id", None) or getattr(r, "section_id", "") or ""),
                    paper_arxiv_id=str(getattr(r, "paper_id", None) or getattr(r, "paper_arxiv_id", "") or ""),
                    role=str(getattr(r, "role", None) or "reference"),
                    section_index=int(raw_index) if raw_index is not None else idx,
                    text=getattr(r, "text", None) or "",
                    title=getattr(r, "title", None),
                    summary=getattr(r, "summary", None),
                    start_char=getattr(r, "start_char", None),
                    end_char=getattr(r, "end_char", None),
                    start_page=getattr(r, "start_page", None),
                    end_page=getattr(r, "end_page", None),
                    embedding=getattr(r, "embedding", None),
                    meta=getattr(r, "meta", None) or {},
                )
            )
        return sections
