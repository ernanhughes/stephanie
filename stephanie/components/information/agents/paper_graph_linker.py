# stephanie/components/information/agents/paper_graph_linker.py
from __future__ import annotations
import re
import hashlib
from typing import Any, Dict, List, Optional, Sequence
from pathlib import Path
from stephanie.components.information.graph.paper_graph_abi import GraphEdge, GraphNode, PaperGraphABI
from stephanie.components.information.graph.paper_graph_dumper import PaperGraphDumper
from stephanie.agents.base_agent import BaseAgent
from stephanie.components.information.tasks.section_link_task import SectionLinkTask

from stephanie.scoring.scorable import Scorable
from stephanie.components.information.data import PaperSection
import logging

from stephanie.tools.pdf_tool import PDFConverter

log = logging.getLogger(__name__)

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None


def _sha1_12(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8", errors="ignore")).hexdigest()[:12]


def _clean_ws(s: str) -> str:
    s = s or ""
    s = s.replace("\u00ad", "")  # soft hyphen
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

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
    txt = _clean_ws(ref_block)
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
        body = _clean_ws(body)
        # Avoid ridiculously huge entries
        if body:
            out[num] = body[:5000]
    return out


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
        span = _clean_ws(t[a:b])

        out.append({"marker": raw, "nums": nums, "span": span})

    return out

class BaseSectionLinker:
    name: str = "base"

    def link(
        self,
        *,
        root_arxiv_id: str,
        root_sections: Sequence[PaperSection],
        corpus_sections: Sequence[PaperSection],
        context: Dict[str, Any],
    ) -> List[GraphEdge]:
        raise NotImplementedError

class SemanticKNNLinker(BaseSectionLinker):
    """
    Wraps your existing SectionLinkTask and converts SectionMatch -> GraphEdge.
    """
    name = "semantic_knn"

    def __init__(self, *, top_k: int, min_sim: float, embed_model: Optional[str] = None) -> None:
        self.top_k = int(top_k)
        self.min_sim = float(min_sim)
        self.embed_model = embed_model

    def link(
        self,
        *,
        root_arxiv_id: str,
        root_sections: Sequence[PaperSection],
        corpus_sections: Sequence[PaperSection],
        context: Dict[str, Any],
    ) -> List[GraphEdge]:
        # SectionLinkTask expects a combined list (root + others), with embeddings prepopulated
        all_sections: List[PaperSection] = list(root_sections) + [s for s in corpus_sections if s.paper_arxiv_id != root_arxiv_id]

        task = SectionLinkTask(root_arxiv_id=root_arxiv_id, top_k=self.top_k, min_sim=self.min_sim)
        matches, clusters = task.run(all_sections)

        # expose clusters if you want them for reporting
        context["concept_clusters"] = clusters

        edges: List[GraphEdge] = []
        for m in matches:
            edges.append(
                GraphEdge(
                    src=f"section:{m.source_section_id}",
                    dst=f"section:{m.target_section_id}",
                    type="SIMILAR_SECTION",
                    weight=float(m.score),
                    evidence={
                        "rank": int(m.rank),
                        "min_sim": float(self.min_sim),
                        "top_k": int(self.top_k),
                        "embed_model": self.embed_model,
                        "reason": m.reason,
                    },
                )
            )
        return edges

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
            txt = _clean_ws(txt)
            if txt:
                out[num] = txt
        return out

    def _lookup_ref_text(self, bib_index: Any, n: int) -> Optional[str]:
        m = self._coerce_bib_num_map(bib_index)
        return m.get(int(n))


class EntityOverlapLinker(BaseSectionLinker):
    """
    Stub: cheap grounding edges via entity overlap.

    Expected future inputs:
      - context["entities_by_section"] = {section_id: ["CLIP","VLM",...]}
    """
    name = "entity_overlap"

    def __init__(self, *, min_jaccard: float = 0.2) -> None:
        self.min_jaccard = float(min_jaccard)

    def link(self, *, root_arxiv_id: str, root_sections: Sequence[PaperSection], corpus_sections: Sequence[PaperSection], context: Dict[str, Any]) -> List[GraphEdge]:
        return []


# -----------------------------
# Agent
# -----------------------------

class PaperGraphLinkerAgent(BaseAgent):
    """
    Stage: paper_graph_linker

    Inputs (context):
      - paper_sections: List[PaperSection]   (root paper semantic sections, in order)
      - section_corpus: List[PaperSection]   (optional: other paper sections with embeddings)
         OR any of: ["all_sections", "candidate_sections", "nexus_sections"]
      - paper / paper_arxiv_id: for identity
    Outputs (context):
      - paper_graph: dict (ABI)
      - section_links: list[dict] (flattened edges)
      - paper_graph_stats: dict
      - paper_graph_file: path
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg=cfg, memory=memory, container=container, logger=logger)

        self.run_dir = self.cfg.get("run_dir", f"runs/paper_blogs/{self.run_id}")
        self.filename = self.cfg.get("filename", "paper_graph.json")

        # linkers config
        sim_cfg = dict(self.cfg.get("similarity", {}) or {})
        self.sim_top_k = int(sim_cfg.get("top_k", 8))
        self.sim_min = float(sim_cfg.get("min_sim", 0.40))
        self.embed_model = sim_cfg.get("embed_model")  # optional metadata string

        self.enable_citations = bool(self.cfg.get("enable_citations", False))
        self.enable_entity_overlap = bool(self.cfg.get("enable_entity_overlap", False))

        ent_cfg = dict(self.cfg.get("entity_overlap", {}) or {})
        self.ent_min_jaccard = float(ent_cfg.get("min_jaccard", 0.2))

        self.papers_root = Path(self.cfg.get("papers_root", "data/papers"))
        self._dumper = PaperGraphDumper(run_dir=self.run_dir)

        self._linkers: List[BaseSectionLinker] = [
            SemanticKNNLinker(top_k=self.sim_top_k, min_sim=self.sim_min, embed_model=self.embed_model),
        ]
        if self.enable_citations:
            self._linkers.append(CitationLinker())
        if self.enable_entity_overlap:
            self._linkers.append(EntityOverlapLinker(min_jaccard=self.ent_min_jaccard))

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        root_arxiv_id = context.get("arxiv_id")

        paper_pdf_path = context["paper_pdf_path"] or self.papers_root / f"{root_arxiv_id}/paper.pdf"

        root_sections: List[PaperSection] = list(context.get("paper_sections") or [])
        if not root_sections:
            log.warning("PaperGraphLinkerAgent: no paper_sections in context; skipping")
            return context

        corpus_sections = self._load_section_corpus(context=context, root_arxiv_id=root_arxiv_id)

        # Build nodes
        nodes: List[GraphNode] = []
        nodes.append(GraphNode(id=f"paper:{root_arxiv_id}", type="paper", title=context.get("paper_title")))

        # root section nodes
        for s in root_sections:
            nodes.append(self._node_from_section(s, root=True))

        # corpus section nodes (only those in other papers)
        seen_section_ids = {s.id for s in root_sections}
        for s in corpus_sections:
            if s.id in seen_section_ids:
                continue
            nodes.append(self._node_from_section(s, root=False))
            seen_section_ids.add(s.id)

        # Link
        edges: List[GraphEdge] = []
        for linker in self._linkers:
            try:
                new_edges = linker.link(
                    root_arxiv_id=root_arxiv_id,
                    root_sections=root_sections,
                    corpus_sections=corpus_sections,
                    context=context,
                )
                edges.extend(new_edges)
                log.info("PaperGraphLinkerAgent: linker=%s edges=%d", linker.name, len(new_edges))
            except Exception:
                log.exception("PaperGraphLinkerAgent: linker=%s failed", linker.name)

        graph = PaperGraphABI(
            version="paper_graph_abi_v1",
            run_id=str(self.run_id),
            root_arxiv_id=root_arxiv_id,
            nodes=nodes,
            edges=edges,
            stats=self._build_stats(root_sections, corpus_sections, edges),
        )

        # Dump
        graph_file = self._dumper.dump(arxiv_id=root_arxiv_id, graph=graph, filename=self.filename)

        # Flatten edges for easy downstream consumption
        section_links = [e.__dict__ for e in edges]

        context["paper_graph"] = graph.to_dict()
        context["section_links"] = section_links
        context["paper_graph_stats"] = graph.stats
        context["paper_graph_file"] = graph_file

        log.info("PaperGraphLinkerAgent: wrote %s (nodes=%d edges=%d)", graph_file, len(nodes), len(edges))
        return context

    # -----------------------------
    # helpers
    # -----------------------------

    def _resolve_root_arxiv_id(self, context: Dict[str, Any]) -> str:
        arxiv_id = context.get("arxiv_id")
        paper = context.get("paper")
        if isinstance(paper, Scorable):
            meta = paper.meta or {}
            arxiv_id = arxiv_id or meta.get("arxiv_id") or meta.get("paper_arxiv_id")
        return str(arxiv_id or "unknown")

    def _load_section_corpus(self, *, context: Dict[str, Any], root_arxiv_id: str) -> List[PaperSection]:
        """
        Best-effort: use corpus already provided by upstream stages.
        If you want DB-backed loading, add it here (e.g. memory.paper_sections.list_for_run()).
        """
        # common keys you might already have
        for key in ("section_corpus", "all_sections", "candidate_sections", "nexus_sections"):
            val = context.get(key)
            if val:
                try:
                    sections = list(val)
                    # Ensure corpus contains non-root too (it can include root; we filter later)
                    return sections
                except Exception:
                    pass

        # fallback: at least root sections so the stage doesn’t crash
        log.warning("PaperGraphLinkerAgent: no section corpus found in context; similarity edges will be empty")
        return list(context.get("paper_sections") or [])

    def _node_from_section(self, s: PaperSection, *, root: bool) -> GraphNode:
        meta = getattr(s, "meta", None) or {}
        sp = getattr(s, "start_page", None) or meta.get("start_page")
        ep = getattr(s, "end_page", None) or meta.get("end_page")

        # NOTE: we namespace section node ids to avoid collisions across systems
        node_id = f"section:{s.id}"
        return GraphNode(
            id=node_id,
            type="section",
            title=getattr(s, "title", None),
            paper_id=getattr(s, "paper_arxiv_id", None),
            section_id=getattr(s, "id", None),
            start_page=int(sp) if sp is not None else None,
            end_page=int(ep) if ep is not None else None,
            meta={
                "root": bool(root),
                "section_index": getattr(s, "section_index", None),
            },
        )

    def _build_stats(self, root_sections: Sequence[PaperSection], corpus_sections: Sequence[PaperSection], edges: Sequence[GraphEdge]) -> Dict[str, Any]:
        by_type: Dict[str, int] = {}
        for e in edges:
            by_type[e.type] = by_type.get(e.type, 0) + 1

        return {
            "root_sections": len(root_sections),
            "corpus_sections": len(corpus_sections),
            "edges_total": len(edges),
            "edges_by_type": by_type,
            "similarity": {"top_k": self.sim_top_k, "min_sim": self.sim_min, "embed_model": self.embed_model},
        }
