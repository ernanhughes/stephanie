# stephanie/tools/arena_citations.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
import math
import re

_logger = logging.getLogger(__name__)

@dataclass
class Citation:
    claim: str
    support_kind: str            # "sidequest" | "chat" | "kg" | "none"
    support_id: Optional[str]
    sim: float
    entity_hits: List[str]

class ArenaCitations:
    """
    Extract factual-ish claims from a winner draft and align each to the best
    supporting artifact from sidequests / chat corpus / KG entities.
    Uses memory.embedding when available; falls back to token overlap.
    """
    def __init__(self, memory: Any, container: Any, logger: Optional[logging.Logger] = None):
        self.memory = memory
        self.container = container
        self.logger = logger or _logger

    # ---------- public ----------

    def generate(
        self,
        winner_text: str,
        sidequest_items: List[Dict[str, Any]] | None = None,
        chat_items: List[Dict[str, Any]] | None = None,
        kg_entities: List[Dict[str, Any]] | None = None,
        min_support_sim: float = 0.60,
    ) -> List[Citation]:
        claims = self._extract_claims(winner_text)
        if not claims:
            return []

        sidequest_items = sidequest_items or []
        chat_items = chat_items or []
        kg_entities = kg_entities or []

        # pre-embed candidate texts for speed
        for it in sidequest_items:
            it["_text"] = it.get("text") or it.get("abstract") or it.get("body") or ""
        for it in chat_items:
            it["_text"] = it.get("assistant_text") or it.get("text") or ""

        cites: List[Citation] = []
        for c in claims:
            sq_id, sq_sim = self._best_match(c, sidequest_items)
            ch_id, ch_sim = self._best_match(c, chat_items)
            ents = self._entities_in_text(c, kg_entities)

            if sq_sim >= ch_sim and sq_sim >= min_support_sim:
                cites.append(Citation(c, "sidequest", sq_id, round(sq_sim, 3), ents[:5]))
            elif ch_sim >= min_support_sim:
                cites.append(Citation(c, "chat", ch_id, round(ch_sim, 3), ents[:5]))
            elif ents:
                # weak KG support if entities are present
                kg_id = ents[0] if isinstance(ents[0], str) else ents[0]
                cites.append(Citation(c, "kg", kg_id, 0.61, ents[:5]))
            else:
                cites.append(Citation(c, "none", None, 0.0, []))
        return cites

    def support_ratio(self, citations: List[Citation]) -> float:
        if not citations:
            return 0.0
        supported = sum(1 for ct in citations if ct.support_kind != "none" and ct.support_id)
        return supported / max(1, len(citations))

    # ---------- internals ----------

    def _extract_claims(self, text: str) -> List[str]:
        if not text:
            return []
        sents = re.split(r'(?<=[.!?])\s+', text.strip())
        out = []
        for s in sents:
            tok = s.split()
            if len(tok) < 8:
                continue
            low = s.lower()
            # heuristic triggers for factual assertions
            if any(k in low for k in [
                "we show", "we find", "results", "evidence", "improves", "reduces",
                "achieves", "outperforms", "measured", "evaluated", "leads to", "based on",
                "we observe", "we compare", "in our experiments"
            ]):
                out.append(s.strip())
        # fallback: return top-N longer sentences
        if not out:
            out = [s.strip() for s in sents if len(s.split()) >= 15][:6]
        return out[:10]

    def _best_match(self, query: str, items: List[Dict[str, Any]]) -> Tuple[Optional[str], float]:
        best_id, best_sim = None, 0.0
        for it in items:
            tid = str(it.get("id") or it.get("artifact_id") or it.get("turn_id") or it.get("uid") or "")
            txt = it.get("_text") or it.get("text") or ""
            sim = self._similarity(query, txt)
            if sim > best_sim:
                best_id, best_sim = tid or None, sim
        return best_id, best_sim

    def _entities_in_text(self, text: str, kg_entities: List[Dict[str, Any]]) -> List[str]:
        out = []
        t = (text or "").lower()
        for e in kg_entities or []:
            name = (e.get("name") or e.get("label") or "").lower()
            if name and name in t:
                out.append(e.get("id") or e.get("name"))
        return out

    def _similarity(self, a: str, b: str) -> float:
        try:
            emb = getattr(self.memory, "embedding", None)
            if emb:
                av = emb.get_or_create(a)
                bv = emb.get_or_create(b)
                num = sum(x*y for x, y in zip(av, bv))
                na = math.sqrt(sum(x*x for x in av)) or 1.0
                nb = math.sqrt(sum(y*y for y in bv)) or 1.0
                return max(0.0, min(1.0, num/(na*nb)))
        except Exception:
            pass
        # bag-of-words overlap fallback
        import re as _re
        A = set(_re.findall(r'\b\w+\b', a.lower()))
        B = set(_re.findall(r'\b\w+\b', b.lower()))
        if not A or not B:
            return 0.0
        return len(A & B) / max(1, len(A | B))
