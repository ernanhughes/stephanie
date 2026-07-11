# stephanie/components/arena/sources/paper_sections.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class PaperSectionRef:
    paper_id: str
    section_id: str
    title: str
    text: str
    start_page: Optional[int] = None
    end_page: Optional[int] = None
    score_hint: Optional[float] = None   # optional ranking hint from retrieval


class PaperSectionCandidateSource:
    """
    Single responsibility:
      sections -> arena candidate dicts
    """

    def __init__(self, *, min_text_len: int = 80, max_text_len: int = 1800):
        self.min_text_len = int(min_text_len)
        self.max_text_len = int(max_text_len)

    def build_candidates(self, sections: List[PaperSectionRef], *, max_candidates: int = 32) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for s in sections or []:
            txt = (s.text or "").strip()
            if len(txt) < self.min_text_len:
                continue
            txt = txt[: self.max_text_len]

            out.append(
                {
                    "origin": "paper_section",
                    "variant": f"{s.paper_id}#{s.section_id}",
                    "text": txt,
                    "meta": {
                        "paper_id": s.paper_id,
                        "section_id": s.section_id,
                        "title": s.title,
                        "start_page": s.start_page,
                        "end_page": s.end_page,
                        "score_hint": s.score_hint,
                    },
                }
            )

        # Optional: stable dedupe by normalized text
        seen = set()
        uniq = []
        for c in out:
            key = " ".join((c.get("text") or "").split()).lower()
            if key in seen:
                continue
            seen.add(key)
            uniq.append(c)

        return uniq[: int(max_candidates)]
