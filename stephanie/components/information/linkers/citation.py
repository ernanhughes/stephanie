import re
import hashlib
from typing import Any, Dict, List, Optional, Sequence
from stephanie.components.information.data import PaperSection
from stephanie.scoring.scorable import Scorable
from .base import BaseSectionLinker

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

from stephanie.tools.arxiv_tool import extract_arxiv_references, fetch_arxiv_metadata
from stephanie.components.information.graph.paper_graph_abi import GraphNode, GraphEdge
import logging
log = logging.getLogger(__name__)


def _sha1_12(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8", errors="ignore")).hexdigest()[:12]


def _clean_ws(s: str) -> str:
    s = s or ""
    s = s.replace("\u00ad", "")  # soft hyphen
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _extract_pdf_text(pdf_path: str, max_pages: Optional[int] = None) -> str:
    if fitz is None:
        return ""
    doc = fitz.open(pdf_path)
    n = doc.page_count
    if max_pages is not None:
        n = min(n, int(max_pages))
    parts: List[str] = []
    for i in range(n):
        parts.append((doc.load_page(i).get_text("text") or ""))
    return "\n".join(parts)


def _find_references_block(full_text: str) -> str:
    t = full_text or ""
    if not t.strip():
        return ""

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
    stop_m = re.search(r"\n\s*(appendix|acknowledg(e)?ments)\s*\n", tail, flags=re.IGNORECASE)
    if stop_m:
        tail = tail[: stop_m.start()]

    return tail.strip()


def _parse_numeric_bib_entries(ref_block: str) -> Dict[int, str]:
    txt = _clean_ws(ref_block)
    if not txt:
        return {}

    txt = txt.replace("\r\n", "\n").replace("\r", "\n")
    pieces = re.split(r"(?m)(?:^|\n)\s*\[(\d{1,4})\]\s*", txt)
    if len(pieces) < 3:
        return {}

    out: Dict[int, str] = {}
    i = 1
    while i + 1 < len(pieces):
        num_s = pieces[i]
        body = pieces[i + 1]
        i += 2
        try:
            num = int(num_s)
        except Exception:
            continue
        body = _clean_ws(body)
        if body:
            out[num] = body[:8000]
    return out


def _extract_numeric_citations(section_text: str) -> List[Dict[str, Any]]:
    t = section_text or ""
    if not t.strip():
        return []

    matches = list(re.finditer(r"\[(\s*\d{1,4}\s*(?:[,–\-]\s*\d{1,4}\s*)*)\]", t))
    out: List[Dict[str, Any]] = []

    for m in matches:
        raw = m.group(0)
        inner = m.group(1)

        nums: List[int] = []
        parts = [p.strip() for p in re.split(r"\s*,\s*", inner) if p.strip()]
        for p in parts:
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

        a = max(0, m.start() - 60)
        b = min(len(t), m.end() + 60)
        span = _clean_ws(t[a:b])

        out.append({"marker": raw, "nums": nums, "span": span})

    return out


def _node_index(ctx: Dict[str, Any]) -> set[str]:
    """Global per-run node id set to avoid duplicates across linkers."""
    key = "_paper_graph_node_ids"
    s = ctx.get(key)
    if not isinstance(s, set):
        s = set()
        ctx[key] = s
    return s


def _add_node_once(ctx: Dict[str, Any], node: GraphNode) -> None:
    extra_nodes: List[GraphNode] = ctx.setdefault("paper_graph_extra_nodes", [])
    ids = _node_index(ctx)
    if node.id in ids:
        return
    extra_nodes.append(node)
    ids.add(node.id)


class CitationLinker(BaseSectionLinker):
    """
    CitationLinker v1.1 (Stephanie-enhanced):
      - Extracts numeric citations [12] style from section text.
      - Parses a numeric bibliography from PDF References section.
      - Uses arxiv_tool.extract_arxiv_references(ref_text) to resolve arXiv IDs.
      - Emits section->paper:<arxiv_id> edges when possible.
      - Falls back to section->ref:<hash> edges otherwise.
    """

    name = "citation"

    def __init__(
        self,
        *,
        max_pdf_pages: Optional[int] = None,
        validate_arxiv: bool = False,
        arxiv_timeout: float = 5.0,
        fetch_metadata: bool = False,
        allow_unresolved: bool = True,
    ) -> None:
        self.max_pdf_pages = max_pdf_pages
        self.validate_arxiv = bool(validate_arxiv)
        self.arxiv_timeout = float(arxiv_timeout)
        self.fetch_metadata = bool(fetch_metadata)
        self.allow_unresolved = bool(allow_unresolved)

    def link(
        self,
        *,
        root_arxiv_id: str,
        root_sections: Sequence[PaperSection],
        corpus_sections: Sequence[PaperSection],
        context: Dict[str, Any],
    ) -> List[GraphEdge]:
        # 1) citations_by_section optional override
        citations_by_section = context.get("citations_by_section")
        if citations_by_section is None:
            citations_by_section = {s.id: _extract_numeric_citations(getattr(s, "text", "") or "") for s in root_sections}

        # 2) bib_index optional override, else parse from PDF
        bib_index = context.get("bib_index")
        if bib_index is None:
            bib_index = self._build_bib_index_from_pdf(context=context)
        context["bib_index"] = bib_index

        edges: List[GraphEdge] = []
        bib_map = self._coerce_bib_num_map(bib_index)

        # Emit edges
        for sec in root_sections:
            sec_cites = citations_by_section.get(sec.id) or []
            if not sec_cites:
                continue

            for cite in sec_cites:
                nums = cite.get("nums") or []
                marker = cite.get("marker")
                span = cite.get("span")

                for n in nums:
                    ref_text = bib_map.get(int(n))

                    # Resolve arXiv IDs from the ref text (best case)
                    arxiv_ids: List[str] = []
                    version_map: Dict[str, Optional[str]] = {}

                    if ref_text:
                        refs = extract_arxiv_references(
                            ref_text,
                            validate=self.validate_arxiv,
                            timeout=self.arxiv_timeout,
                        )
                        for rr in refs:
                            arxiv_ids.append(rr.arxiv_id)  # normalized base id
                            version_map[rr.arxiv_id] = rr.version

                    arxiv_ids = sorted(set(arxiv_ids))

                    if arxiv_ids:
                        # Prefer paper nodes for each resolved arXiv id
                        for aid in arxiv_ids:
                            paper_node_id = f"paper:{aid}"

                            # add paper node (optionally enriched)
                            node = GraphNode(
                                id=paper_node_id,
                                type="paper",
                                title=None,
                                paper_id=aid,
                                meta={
                                    "source": "citation_arxiv_resolver",
                                    "from_root": root_arxiv_id,
                                    "ref_num": int(n),
                                    "version": version_map.get(aid),
                                    "ref_text": (ref_text[:800] if ref_text else None),
                                },
                            )

                            if self.fetch_metadata:
                                try:
                                    md = fetch_arxiv_metadata(aid) or {}
                                    node.title = md.get("title") or node.title
                                    node.meta.update(
                                        {
                                            "authors": md.get("authors"),
                                            "published": md.get("published"),
                                            "url": md.get("url"),
                                        }
                                    )
                                except Exception:
                                    # metadata is optional; failure should not break linking
                                    pass

                            _add_node_once(context, node)

                            edges.append(
                                GraphEdge(
                                    src=f"section:{sec.id}",
                                    dst=paper_node_id,
                                    type="CITES",
                                    weight=1.0,
                                    evidence={
                                        "marker": marker,
                                        "ref_num": int(n),
                                        "span": span,
                                        "resolved_arxiv_id": aid,
                                        "ref_text": (ref_text[:500] if ref_text else None),
                                    },
                                )
                            )
                        continue

                    # Fallback to ref node
                    if (not ref_text) and (not self.allow_unresolved):
                        continue

                    ref_key = ref_text or f"{root_arxiv_id}:[{n}]"
                    ref_hash = _sha1_12(ref_key)
                    ref_node_id = f"ref:{root_arxiv_id}:{ref_hash}"

                    _add_node_once(
                        context,
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
                        ),
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
                                "span": span,
                                "ref_text": (ref_text[:500] if ref_text else None),
                                "resolved_arxiv_id": None,
                            },
                        )
                    )

        # Stats for report/debug
        context.setdefault("paper_graph_linker_stats", {})["citations"] = {
            "sections_with_cites": sum(1 for s in root_sections if citations_by_section.get(s.id)),
            "total_edges": len(edges),
            "bib_entries": len(bib_map),
            "validate_arxiv": self.validate_arxiv,
            "fetch_metadata": self.fetch_metadata,
        }
        return edges

    def _build_bib_index_from_pdf(self, *, context: Dict[str, Any]) -> Dict[int, str]:
        pdf_path = context.get("paper_pdf_path")
        paper = context.get("paper")
        if isinstance(paper, Scorable):
            meta = paper.meta or {}
            pdf_path = pdf_path or meta.get("pdf_path") or meta.get("paper_pdf_path")

        if not pdf_path:
            return {}

        try:
            full_text = _extract_pdf_text(str(pdf_path), max_pages=self.max_pdf_pages)
            ref_block = _find_references_block(full_text)
            return _parse_numeric_bib_entries(ref_block)
        except Exception:
            log.exception("CitationLinker: failed to parse bibliography from pdf_path=%s", pdf_path)
            return {}

    def _coerce_bib_num_map(self, bib_index: Any) -> Dict[int, str]:
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
            txt = _clean_ws(txt)
            if txt:
                out[num] = txt
        return out
