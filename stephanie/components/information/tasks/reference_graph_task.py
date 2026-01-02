# stephanie/components/information/tasks/reference_graph_task.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Protocol

from stephanie.components.information.data import (PaperNode,
                                                   PaperReferenceGraph,
                                                   PaperReferenceRecord,
                                                   ReferenceEdge)
from stephanie.components.information.utils.graph_utils import write_graph_json
from stephanie.tools.paper_import_tool import (PaperImportResult,
                                               PaperImportTool)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provider interfaces (very small, you can back them with arxiv/HF tools)
# ---------------------------------------------------------------------------


class ReferenceProvider(Protocol):
    def get_references_for_arxiv(self, arxiv_id: str) -> List[PaperReferenceRecord]:
        ...


class SimilarPaperProvider(Protocol):
    def get_similar_for_arxiv(self, arxiv_id: str, limit: int = 10) -> List[PaperReferenceRecord]:
        ...


# ---------------------------------------------------------------------------
# ReferenceGraphTask
# ---------------------------------------------------------------------------


class ReferenceGraphTask:
    """
    Task:
        Given a root arxiv_id,
        - import the root paper (if not already imported)
        - fetch its references via ReferenceProvider
        - (optionally) fetch similar papers via SimilarPaperProvider
        - import PDFs for references / similars
        - build a PaperReferenceGraph

    It does *not* do any sectioning / summarization; that is the next task.
    """

    def __init__(
        self,
        papers_root: Path,
        import_tool: PaperImportTool,
        ref_provider: ReferenceProvider,
        similar_provider: Optional[SimilarPaperProvider] = None,
        graph_json_name: str = "graph.json",
    ) -> None:
        self.papers_root = Path(papers_root)
        self.import_tool = import_tool
        self.ref_provider = ref_provider
        self.similar_provider = similar_provider
        self.graph_json_name = graph_json_name

    # ------------------------------------------------------------------ #
    async def run(
        self,
        root_arxiv_id: str,
        *,
        max_refs: Optional[int] = None,
        max_similar: int = 10,
    ) -> PaperReferenceGraph:
        """
        Build a reference cluster for a single root paper.
        """

        root_dir = self._ensure_root_folder(root_arxiv_id)
        log.info("[ReferenceGraphTask] Building graph for %s in %s", root_arxiv_id, root_dir)

        # 1) Import root paper
        root_import: PaperImportResult = await self.import_tool.import_paper(
            arxiv_id=root_arxiv_id,
            local_pdf_path=root_dir / "paper.pdf",
            source="arxiv",
            force_references=False,
            role="root",
        )
        root_node = root_import.node

        nodes: Dict[str, PaperNode] = {root_node.arxiv_id: root_node}
        edges: List[ReferenceEdge] = []

        # 2) Fetch references
        ref_records = self.ref_provider.get_references_for_arxiv(root_arxiv_id)
        if max_refs is not None:
            ref_records = ref_records[:max_refs]

        log.info(
            "[ReferenceGraphTask] Found %d reference(s) for %s",
            len(ref_records),
            root_arxiv_id,
        )

        # 3) Import reference PDFs and add to graph
        for rec in ref_records:
            ref_id = rec.arxiv_id
            if not ref_id:
                continue

            if ref_id in nodes:
                # already seen
                edges.append(ReferenceEdge(src=root_arxiv_id, dst=ref_id, kind="cites"))
                continue

            node: Optional[PaperNode] = None

            try:
                ref_import = await self.import_tool.import_paper(arxiv_id=ref_id, role="reference", force=False)
                node = ref_import.node
            except Exception as e:
                log.warning(
                    "Failed to import reference PDF %s; creating stub node: %s",
                    ref_id,
                    e,
                )
                # Create a lightweight stub node
                node = PaperNode(
                    arxiv_id=ref_id,
                    role="reference",
                    title=rec.title or f"arXiv:{ref_id}",
                    url=rec.url,
                    pdf_path=None,
                    text_hash=None,
                    meta={},
                )

            # Enrich metadata (both real & stub nodes)
            node.title = node.title or rec.title
            node.url = node.url or rec.url
            node.meta.setdefault("reference_raw", rec.raw)
            node.meta.setdefault("import_status", "stub" if node.pdf_path is None else "ok")

            nodes[ref_id] = node
            edges.append(ReferenceEdge(src=root_arxiv_id, dst=ref_id, kind="cites"))

        # 4) Optional: similar papers (root + references)
        if self.similar_provider is not None:
            await self._add_similar_papers(
                graph_nodes=nodes,
                edges=edges,
                root_arxiv_id=root_arxiv_id,
                max_similar=max_similar,
            )

        graph = PaperReferenceGraph(
            root_arxiv_id=root_arxiv_id,
            nodes=nodes,
            edges=edges,
        )

        # 5) Persist graph.json for inspection / reuse
        write_graph_json("graph.json", root_dir, graph)

        return graph

    # ------------------------------------------------------------------ #
    def _ensure_root_folder(self, root_arxiv_id: str) -> Path:
        root_dir = self.papers_root / root_arxiv_id
        root_dir.mkdir(parents=True, exist_ok=True)
        return root_dir

    async def _add_similar_papers(
        self,
        *,
        graph_nodes: Dict[str, PaperNode],
        edges: List[ReferenceEdge],
        root_arxiv_id: str,
        max_similar: int,
    ) -> None:
        """
        Use SimilarPaperProvider to enlarge the graph around the ROOT paper only.

        This intentionally does NOT expand similars of references.
        That behavior belongs in a separate recursive citation-crawl task.
        """

        if self.similar_provider is None:
            return

        root_similar = self.similar_provider.get_similar_for_arxiv(
            root_arxiv_id, limit=max_similar
        )

        for rec in root_similar:
            aid = rec.arxiv_id
            if not aid or aid == root_arxiv_id:
                continue

            if aid not in graph_nodes:
                try:
                    imp = await self.import_tool.import_paper(
                        arxiv_id=aid,
                        role="similar_root",
                    )
                except Exception as e:
                    log.warning(
                        "Failed to import similar(root) %s: %s",
                        aid,
                        e,
                    )
                    continue

                node = imp.node
                node.title = node.title or rec.title
                node.url = node.url or rec.url
                node.meta.setdefault("similar_raw", rec.raw)
                graph_nodes[aid] = node

            edges.append(
                ReferenceEdge(
                    src=root_arxiv_id,
                    dst=aid,
                    kind="similar_to",
                )
            )
