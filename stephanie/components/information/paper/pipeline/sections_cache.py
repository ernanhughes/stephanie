# stephanie/components/information/paper/pipeline/sections_cache.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from stephanie.components.information.data import PaperSection

log = logging.getLogger(__name__)


@dataclass
class PaperSectionsCacheConfig:
    enabled: bool = True


class PaperSectionsCache:
    """
    Loads and persists PaperSection objects via your PaperStore/ORM.
    Mirrors the behavior from PaperPipelineAgent:
      - if cached sections exist, use them
      - else, build sections and persist back
    """

    def __init__(self, *, cfg: Dict[str, Any], paper_store: Any, logger: Any):
        self.cfg = cfg
        self.paper_store = paper_store
        self.logger = logger

        self.enabled = bool(cfg.get("sections_cache_enabled", True))

    def maybe_load(self, arxiv_id: str, role: str = "reference") -> Optional[List[PaperSection]]:
        if not self.enabled:
            return None

        arxiv_id = str(arxiv_id)
        try:
            # Your original agent used "maybe_load_sections_from_store" and
            # converted ORM rows -> PaperSection. We keep that behavior.
            rows = self.paper_store.get_sections_for_paper(paper_id=arxiv_id)
            if not rows:
                return None
            return self._sections_from_orm(rows)
        except Exception:
            log.warning("PaperSectionsCache: load failed for %s", arxiv_id, exc_info=True)
            return None

    def persist(self, *, arxiv_id: str, sections: List[PaperSection]) -> None:
        if not self.enabled:
            return
        arxiv_id = str(arxiv_id)

        try:
            self.paper_store.upsert_sections_for_arxiv(arxiv_id=arxiv_id, sections=sections)
        except Exception:
            # fallback, some stores use different method names; keep it non-fatal
            log.warning("PaperSectionsCache: persist failed for %s", arxiv_id, exc_info=True)

    # ----------------------- internal conversions ------------------------

    def _sections_from_orm(self, rows: List[Any]) -> List[PaperSection]:
        sections: List[PaperSection] = []
        for idx, r in enumerate(rows):
            # Be permissive with field names to match your ORM evolution.
            sections.append(
                PaperSection(
                    section_id=str(getattr(r, "section_id", None) or getattr(r, "id", "")),
                    section_index=int(getattr(r, "section_index", None) or idx),
                    role=str(getattr(r, "role", None) or "reference"),
                    paper_arxiv_id=str(getattr(r, "arxiv_id", None) or getattr(r, "paper_id", "")),
                    title=getattr(r, "title", None) or getattr(r, "section_title", None),
                    summary=getattr(r, "summary", None) or getattr(r, "text_summary", None),
                    text=getattr(r, "text", None) or getattr(r, "content", None) or "",
                    embedding=getattr(r, "embedding", None),
                    meta=getattr(r, "meta", None) or {},
                )
            )
        return sections
