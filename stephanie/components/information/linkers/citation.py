# stephanie/components/information/linkers/citation.py
from __future__ import annotations

import hashlib
import logging
import re
from typing import Any, Dict, List, Optional, Sequence

from stephanie.components.information.data import PaperSection
from stephanie.components.information.graph.paper_graph_abi import (GraphEdge,
                                                                    GraphNode)
from stephanie.components.information.linkers.base import BaseSectionLinker
from stephanie.scoring.scorable import Scorable
from stephanie.tools.pdf_tool import PDFConverter
from stephanie.utils.text_utils import clean_ws

log = logging.getLogger(__name__)

class CitationLinker(BaseSectionLinker):
    """
    CitationLinker v1:
      - Supports numeric citations [12] style.
      - Builds ref nodes from References section, if pdf text is available.
      - Emits edges: section -> ref node, type=CITES, weight=1.0
    """

    name = "citation"

    def __init__(
        self,
        *,
        max_pdf_pages: Optional[int] = None,
        allow_unresolved: bool = True,
    ) -> None:
        self.max_pdf_pages = max_pdf_pages
        self.allow_unresolved = allow_unresolved

    def link(
        self,
        *,
        root_arxiv_id: str,
        root_sections: Sequence[PaperSection],
        corpus_sections: Sequence[PaperSection],
        context: Dict[str, Any],
    ) -> List[GraphEdge]:
        # 1) If upstream already provided parsed structures, use them.
        #    citations_by_section: {section_id: [{"nums":[12], "marker":"[12]", "span":"..."}, ...]}
        citations_by_section = context.get("citations_by_section")
        bib_index = context.get("bib_index")  # {12: {"text": "...", "arxiv_id": "..."} } or {12:"..."}

        # 2) Otherwise, try to parse from the root PDF.
        if citations_by_section is None:
            citations_by_section = {}
            for s in root_sections:
                citations_by_section[s.id] = _extract_numeric_citations(getattr(s, "text", "") or "")

        if bib_index is None:
            bib_index = self._build_bib_index_from_pdf(context=context)

        # Keep bib info in context for report/debug
        context["bib_index"] = bib_index

        extra_nodes: List[GraphNode] = context.setdefault("paper_graph_extra_nodes", [])

        edges: List[GraphEdge] = []
        for sec in root_sections:
            sec_cites = citations_by_section.get(sec.id) or []
            if not sec_cites:
                continue

            for cite in sec_cites:
                nums = cite.get("nums") or []
                marker = cite.get("marker")
                span = cite.get("span")

                for n in nums:
                    ref_text = self._lookup_ref_text(bib_index, n)
                    if (not ref_text) and (not self.allow_unresolved):
                        continue

                    # Stable ref node id: hash of ref text or numeric placeholder
                    ref_key = ref_text or f"{root_arxiv_id}:[{n}]"
                    ref_hash = _sha1_12(ref_key)
                    ref_node_id = f"ref:{root_arxiv_id}:{ref_hash}"

                    # Add ref node once
                    if not any(nn.id == ref_node_id for nn in extra_nodes):
                        extra_nodes.append(
                            GraphNode(
                                id=ref_node_id,
                                type="ref",
                                title=(ref_text[:140] if ref_text else f"Reference [{n}]"),
                                paper_id=root_arxiv_id,
                                meta={
                                    "ref_num": int(n),
                                    "ref_text": ref_text,
                                    "source": "pdf_references" if ref_text else "unresolved",
                                },
                            )
                        )

                    edges.append(
                        GraphEdge(
                            src=f"section:{sec.id}",
                            dst=ref_node_id,
                            type="CITES",
                            weight=1.0,
                            evidence={
                                "marker": marker,
                                "ref_num": int(n),
                                "ref_text": (ref_text[:500] if ref_text else None),
                                "span": span,
                            },
                        )
                    )

        # Stats for report
        context.setdefault("paper_graph_linker_stats", {})["citations"] = {
            "sections_with_cites": sum(1 for s in root_sections if citations_by_section.get(s.id)),
            "total_edges": len(edges),
            "bib_entries": len(self._coerce_bib_num_map(bib_index)),
        }
        return edges

    def _build_bib_index_from_pdf(self, *, context: Dict[str, Any]) -> Dict[int, str]:
        # Try context first
        pdf_path = context.get("paper_pdf_path")
        paper = context.get("paper")
        if isinstance(paper, Scorable):
            meta = paper.meta or {}
            pdf_path = pdf_path or meta.get("pdf_path") or meta.get("paper_pdf_path")

        if not pdf_path:
            return {}

        try:
            full_text = PDFConverter.pdf_to_text(str(pdf_path))
            ref_block = _find_references_block(full_text)
            bib_map = _parse_numeric_bib_entries(ref_block)
            return bib_map
        except Exception:
            log.exception("CitationLinker: failed to parse bibliography from pdf_path=%s", pdf_path)
            return {}

    def _coerce_bib_num_map(self, bib_index: Any) -> Dict[int, str]:
        """
        Accept bib_index in either form:
          {12: "ref text"}
          {12: {"text": "...", ...}}
        """
        out: Dict[int, str] = {}
        if not isinstance(bib_index, dict):
            return out
        for k, v in bib_index.items():
            try:
                num = int(k)
            except Exception:
                continue
            if isinstance(v, dict):
                txt = v.get("text") or v.get("ref_text") or v.get("raw") or ""
            else:
                txt = str(v or "")
            txt = clean_ws(txt)
            if txt:
                out[num] = txt
        return out

    def _lookup_ref_text(self, bib_index: Any, n: int) -> Optional[str]:
        m = self._coerce_bib_num_map(bib_index)
        return m.get(int(n))

def _extract_numeric_citations(section_text: str) -> List[Dict[str, Any]]:
    """
    Extract bracket citations like:
      [12], [12, 13], [12–15], [12-15]
    Returns a list of dicts:
      {"marker": "[12]", "nums": [12], "span": "...context..."}
    """
    t = section_text or ""
    if not t.strip():
        return []

    # capture bracket content
    # Examples: [12], [12,13], [12, 13, 14], [12-15], [12–15]
    matches = list(re.finditer(r"\[(\s*\d{1,4}\s*(?:[,–\-]\s*\d{1,4}\s*)*)\]", t))
    out: List[Dict[str, Any]] = []

    for m in matches:
        raw = m.group(0)
        inner = m.group(1)

        # parse numbers and ranges
        nums: List[int] = []
        # split by commas first
        parts = [p.strip() for p in re.split(r"\s*,\s*", inner) if p.strip()]
        for p in parts:
            # range?
            rm = re.match(r"^(\d{1,4})\s*[–\-]\s*(\d{1,4})$", p)
            if rm:
                a = int(rm.group(1))
                b = int(rm.group(2))
                lo, hi = (a, b) if a <= b else (b, a)
                nums.extend(list(range(lo, hi + 1)))
            else:
                try:
                    nums.append(int(re.sub(r"\s+", "", p)))
                except Exception:
                    pass

        nums = sorted(set(n for n in nums if 0 < n < 10000))
        if not nums:
            continue

        # small context window for evidence (optional)
        a = max(0, m.start() - 60)
        b = min(len(t), m.end() + 60)
        span = clean_ws(t[a:b])

        out.append({"marker": raw, "nums": nums, "span": span})

    return out

def _sha1_12(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8", errors="ignore")).hexdigest()[:12]


def _find_references_block(full_text: str) -> str:
    """
    Try to locate the References/Bibliography section in extracted PDF text.
    Very heuristic, but works surprisingly often.

    Returns: substring likely containing references.
    """
    t = full_text or ""
    if not t.strip():
        return ""

    # Prefer "References" then "Bibliography"
    # We capture from that heading to end (or to "Appendix" if present).
    patterns = [
        r"\n\s*references\s*\n",
        r"\n\s*bibliography\s*\n",
    ]

    start = None
    for pat in patterns:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            start = m.start()
            break
    if start is None:
        return ""

    tail = t[start:]

    # Stop at Appendix / Acknowledgements if present (optional)
    stop_m = re.search(r"\n\s*(appendix|acknowledg(e)?ments)\s*\n", tail, flags=re.IGNORECASE)
    if stop_m:
        tail = tail[: stop_m.start()]

    return tail.strip()

def _parse_numeric_bib_entries(ref_block: str) -> Dict[int, str]:
    """
    Parse references like:
      [12] Author... Title...
    Or sometimes:
      [12] Author... (line wraps)
    Returns: {12: "Author... Title..."}
    """
    txt = clean_ws(ref_block)
    if not txt:
        return {}

    # Keep line breaks because entries are often delimited by newline + [n]
    # Normalize Windows newlines
    txt = txt.replace("\r\n", "\n").replace("\r", "\n")

    # Split on occurrences of "\n[12]" or start of string "[12]"
    # Use a capturing split so we keep the number.
    pieces = re.split(r"(?m)(?:^|\n)\s*\[(\d{1,4})\]\s*", txt)
    # pieces looks like: ["prefix", "12", "entry...", "13", "entry...", ...]
    if len(pieces) < 3:
        return {}

    out: Dict[int, str] = {}
    # skip prefix at index 0
    i = 1
    while i + 1 < len(pieces):
        num_s = pieces[i]
        body = pieces[i + 1]
        i += 2
        try:
            num = int(num_s)
        except Exception:
            continue
        # body runs until next split boundary; trim excessive whitespace
        body = clean_ws(body)
        # Avoid ridiculously huge entries
        if body:
            out[num] = body[:5000]
    return out
