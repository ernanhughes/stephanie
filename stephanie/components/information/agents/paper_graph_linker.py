# stephanie/components/information/agents/paper_graph_linker.py
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

from stephanie.agents.base_agent import BaseAgent
from stephanie.components.information.data import PaperSection
from stephanie.components.information.graph.paper_graph_abi import (
    GraphEdge,
    GraphNode,
    PaperGraphABI,
)
from stephanie.components.information.graph.paper_graph_dumper import (
    PaperGraphDumper,
)
from stephanie.components.information.linkers import (
    BaseSectionLinker,
    section_pid,
    CitationLinker,
    SemanticKNNLinker,
    EntityOverlapLinker,
)
from stephanie.components.information.utils.graph_utils import (
    node_from_section,
)

log = logging.getLogger(__name__)


_ARXIV_PREFIX_RE = re.compile(
    r"^(?P<pid>\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})$", re.IGNORECASE
)


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
        super().__init__(
            cfg=cfg, memory=memory, container=container, logger=logger
        )

        self.run_dir = self.cfg.get(
            "run_dir", f"runs/paper_blogs/{self.run_id}"
        )
        self.filename = self.cfg.get("filename", "paper_graph.json")

        # linkers config
        sim_cfg = dict(self.cfg.get("similarity", {}) or {})
        self.sim_top_k = int(sim_cfg.get("top_k", 8))
        self.sim_min = float(sim_cfg.get("min_sim", 0.40))
        self.embed_model = sim_cfg.get(
            "embed_model"
        )  # optional metadata string

        self.enable_citations = bool(self.cfg.get("enable_citations", False))
        self.enable_entity_overlap = bool(
            self.cfg.get("enable_entity_overlap", False)
        )

        ent_cfg = dict(self.cfg.get("entity_overlap", {}) or {})
        self.ent_min_jaccard = float(ent_cfg.get("min_jaccard", 0.2))

        self.papers_root = Path(self.cfg.get("papers_root", "data/papers"))
        self._dumper = PaperGraphDumper(run_dir=self.run_dir)

        self._linkers: List[BaseSectionLinker] = [
            SemanticKNNLinker(
                top_k=self.sim_top_k,
                min_sim=self.sim_min,
                embed_model=self.embed_model,
            ),
        ]
        if self.enable_citations:
            self._linkers.append(CitationLinker())
        if self.enable_entity_overlap:
            self._linkers.append(
                EntityOverlapLinker(min_jaccard=self.ent_min_jaccard)
            )

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        root_arxiv_id = _norm_pid(context.get("arxiv_id"))

        all_sections: List[PaperSection] = list(
            context.get("paper_sections") or []
        )
        if not all_sections:
            log.warning(
                "PaperGraphLinkerAgent: no paper_sections in context; skipping"
            )
            return context

        # Partition correctly (even if paper_arxiv_id was stamped wrong)
        root_sections = [
            s for s in all_sections if section_pid(s) == root_arxiv_id
        ]
        corpus_sections = [
            s for s in all_sections if section_pid(s) != root_arxiv_id
        ]

        # Debug counts (you want to see ~16 root, and many corpus)
        from collections import Counter

        c = Counter(section_pid(s) for s in all_sections)
        log.info("GraphLinker: papers=%d top=%s", len(c), c.most_common(10))
        log.info(
            "GraphLinker: root=%d corpus=%d total=%d",
            len(root_sections),
            len(corpus_sections),
            len(all_sections),
        )

        # optional: filter out spine_page sections from linking
        def _is_spine_page(s: PaperSection) -> bool:
            meta = getattr(s, "meta", None) or {}
            return meta.get("kind") == "spine_page"

        root_sections = [s for s in root_sections if not _is_spine_page(s)]
        corpus_sections = [s for s in corpus_sections if not _is_spine_page(s)]

        # Build nodes
        nodes: List[GraphNode] = []
        nodes.append(
            GraphNode(
                id=f"paper:{root_arxiv_id}",
                type="paper",
                title=context.get("paper_title"),
            )
        )

        # root section nodes
        for s in root_sections:
            nodes.append(node_from_section(s, root=True))

        # corpus section nodes (only those in other papers)
        seen_section_ids = {s.section_id for s in root_sections}
        for s in corpus_sections:
            if s.section_id in seen_section_ids:
                continue
            nodes.append(node_from_section(s, root=False))
            seen_section_ids.add(s.section_id)

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
                log.info(
                    "PaperGraphLinkerAgent: linker=%s edges=%d",
                    linker.name,
                    len(new_edges),
                )
            except Exception:
                log.exception(
                    "PaperGraphLinkerAgent: linker=%s failed", linker.name
                )

        graph = PaperGraphABI(
            version="paper_graph_abi_v1",
            run_id=str(self.run_id),
            root_arxiv_id=root_arxiv_id,
            nodes=nodes,
            edges=edges,
            stats=self._build_stats(root_sections, corpus_sections, edges),
        )

        # Dump
        graph_file = self._dumper.dump(
            arxiv_id=root_arxiv_id, graph=graph, filename=self.filename
        )

        # Flatten edges for easy downstream consumption
        section_links = [e.__dict__ for e in edges]

        context["paper_graph"] = graph.to_dict()
        context["section_links"] = section_links
        context["paper_graph_stats"] = graph.stats
        context["paper_graph_file"] = graph_file

        log.info(
            "PaperGraphLinkerAgent: wrote %s (nodes=%d edges=%d)",
            graph_file,
            len(nodes),
            len(edges),
        )
        return context

    # -----------------------------
    # helpers
    # -----------------------------

    def _load_section_corpus(
        self, *, context: Dict[str, Any], root_arxiv_id: str
    ) -> List[PaperSection]:
        """
        Best-effort: use corpus already provided by upstream stages.
        If you want DB-backed loading, add it here (e.g. memory.paper_sections.list_for_run()).
        """
        # common keys you might already have
        for key in (
            "section_corpus",
            "all_sections",
            "candidate_sections",
            "nexus_sections",
        ):
            val = context.get(key)
            if val:
                try:
                    sections = list(val)
                    # Ensure corpus contains non-root too (it can include root; we filter later)
                    return sections
                except Exception:
                    pass

        # fallback: at least root sections so the stage doesn’t crash
        log.warning(
            "PaperGraphLinkerAgent: no section corpus found in context; similarity edges will be empty"
        )
        return list(context.get("paper_sections") or [])

    def _build_stats(
        self,
        root_sections: Sequence[PaperSection],
        corpus_sections: Sequence[PaperSection],
        edges: Sequence[GraphEdge],
    ) -> Dict[str, Any]:
        by_type: Dict[str, int] = {}
        for e in edges:
            by_type[e.type] = by_type.get(e.type, 0) + 1

        return {
            "root_sections": len(root_sections),
            "corpus_sections": len(corpus_sections),
            "edges_total": len(edges),
            "edges_by_type": by_type,
            "similarity": {
                "top_k": self.sim_top_k,
                "min_sim": self.sim_min,
                "embed_model": self.embed_model,
            },
        }


def _norm_pid(x: str | None) -> str:
    x = (x or "").strip()
    return x.split("v")[0]  # drop version if present
