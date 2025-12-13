import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from stephanie.components.information_legacy.data import (PaperNode,
                                                          PaperReferenceGraph,
                                                          ReferenceEdge,
                                                          ReferenceRecord)
from stephanie.components.information_legacy.reference_provider import \
    ReferenceProvider

log = logging.getLogger(__name__)


class BuildReferenceGraphTask:
    """
    Task 2:
    Given a root arxiv_id (and an optional existing root pdf),
    - create a folder for the root paper
    - fetch its references using a ReferenceProvider
    - for each reference, try to download/import the PDF
    - hang them off the root in a folder structure
    - build a small in-memory graph + graph.json
    """

    def __init__(
        self,
        papers_root: Path,
        import_task: ImportPdfTask,
        ref_provider: ReferenceProvider,
    ) -> None:
        self.papers_root = papers_root
        self.import_task = import_task
        self.ref_provider = ref_provider

    # ---------------------------------------------------------------------    
    def run(
        self,
        root_arxiv_id: str,
        root_pdf_path: Optional[Path] = None,
        max_refs: int = 50,
    ) -> PaperReferenceGraph:
        """
        Build the reference cluster for a single root paper.

        Returns a PaperReferenceGraph that higher-level code can pass to the
        summarizer / blog-post creator.
        """
        root_dir = self._ensure_root_folder(root_arxiv_id)
        refs_dir = root_dir / "refs"
        refs_dir.mkdir(exist_ok=True, parents=True)

        # 1) Ensure we have root PDF & metadata in the root folder
        root_node = self._prepare_root_node(
            root_arxiv_id=root_arxiv_id,
            root_dir=root_dir,
            root_pdf_path=root_pdf_path,
        )

        # 2) Fetch references via provider
        ref_records = self.ref_provider.get_references_for_arxiv(root_arxiv_id)
        if max_refs is not None:
            ref_records = ref_records[:max_refs]

        log.info("Found %d reference(s) for %s", len(ref_records), root_arxiv_id)

        nodes: Dict[str, PaperNode] = {root_arxiv_id: root_node}
        edges: List[ReferenceEdge] = []

        # 3) For each reference, resolve/download/import and add to graph
        for ref in ref_records:
            node, edge = self._process_reference(
                root_arxiv_id=root_arxiv_id,
                root_dir=root_dir,
                refs_dir=refs_dir,
                ref=ref,
            )

            if node is not None:
                key = node.arxiv_id or self._make_synthetic_id(node)
                nodes[key] = node

            if edge is not None:
                edges.append(edge)

        graph = PaperReferenceGraph(
            root=root_node,
            nodes=nodes,
            edges=edges,
        )

        # 4) Persist graph.json for easy inspection / future reuse
        self._write_graph_json(root_dir, graph)

        return graph

    # ------------------------------------------------------------------    
    def _ensure_root_folder(self, root_arxiv_id: str) -> Path:
        root_dir = self.papers_root / root_arxiv_id
        root_dir.mkdir(exist_ok=True, parents=True)
        return root_dir

    def _prepare_root_node(
        self,
        root_arxiv_id: str,
        root_dir: Path,
        root_pdf_path: Optional[Path],
    ) -> PaperNode:
        """
        Make sure the root paper is imported and described as a PaperNode.
        If root_pdf_path is not given, you could optionally fetch it via arxiv.
        """
        pdf_path = None

        if root_pdf_path is not None:
            # Copy or reuse in-place; here we just reuse
            pdf_path = root_pdf_path
        else:
            # Optional: implement auto-download from arxiv here.
            # For now we leave pdf_path as None if not provided.
            log.warning("No root_pdf_path provided; pdf_path will be None.")

        metadata_path = root_dir / "metadata.json"

        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
        else:
            metadata = {
                "arxiv_id": root_arxiv_id,
                "source": "root",
            }
            metadata_path.write_text(json.dumps(metadata, indent=2))

        return PaperNode(
            arxiv_id=root_arxiv_id,
            local_dir=root_dir,
            pdf_path=pdf_path,
            metadata=metadata,
        )

    def _process_reference(
        self,
        root_arxiv_id: str,
        root_dir: Path,
        refs_dir: Path,
        ref: ReferenceRecord,
    ) -> tuple[Optional[PaperNode], Optional[ReferenceEdge]]:
        """
        Handle one reference:
        - choose a folder name
        - attempt to download/import PDF via ImportPdfTask
        - construct node + edge
        """
        # Determine a key for folder + graph
        key = ref.arxiv_id or (ref.doi or self._sanitize_title(ref.title) or "unknown_ref")
        ref_dir = refs_dir / key
        ref_dir.mkdir(exist_ok=True, parents=True)

        pdf_path: Optional[Path] = None
        metadata: Dict[str, Any] = {
            "arxiv_id": ref.arxiv_id,
            "doi": ref.doi,
            "title": ref.title,
            "year": ref.year,
            "url": ref.url,
            "raw_citation": ref.raw_citation,
        }

        # Persist metadata early
        (ref_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

        # If we have a URL (or arxiv_id we can turn into a URL), try to import the PDF
        try:
            if ref.url:
                # Reuse ImportPdfTask: it can handle URLs as input
                import_result = self.import_task.run(input_path_or_url=ref.url)
                # We expect exactly one import from a single URL
                if import_result["imports"]:
                    pdf_import = import_result["imports"][0]
                    pdf_path = pdf_import.source.path
            else:
                # Optional: if arxiv_id is available, derive the PDF URL
                # and call import_task.run(...) on it.
                pass
        except Exception as e:
            log.warning("Failed to import PDF for reference %s: %s", key, e)

        node = PaperNode(
            arxiv_id=ref.arxiv_id,
            local_dir=ref_dir,
            pdf_path=pdf_path,
            metadata=metadata,
        )

        edge = ReferenceEdge(
            source_arxiv_id=root_arxiv_id,
            target_arxiv_id=ref.arxiv_id,
            relation="cites",
        )

        return node, edge

    def _write_graph_json(self, root_dir: Path, graph: PaperReferenceGraph) -> None:
        """
        Very simple JSON serialization of the graph for inspection and later reuse.
        """
        def node_to_dict(n: PaperNode) -> Dict[str, Any]:
            return {
                "arxiv_id": n.arxiv_id,
                "local_dir": str(n.local_dir),
                "pdf_path": str(n.pdf_path) if n.pdf_path else None,
                "metadata": n.metadata,
            }

        data = {
            "root": node_to_dict(graph.root),
            "nodes": {k: node_to_dict(v) for k, v in graph.nodes.items()},
            "edges": [
                {
                    "source_arxiv_id": e.source_arxiv_id,
                    "target_arxiv_id": e.target_arxiv_id,
                    "relation": e.relation,
                }
                for e in graph.edges
            ],
        }

        (root_dir / "graph.json").write_text(json.dumps(data, indent=2))

    def _sanitize_title(self, title: Optional[str]) -> str:
        if not title:
            return "untitled"
        # crude safe-folder-name
        return "".join(c for c in title.lower().replace(" ", "_") if c.isalnum() or c in "._-")

    def _make_synthetic_id(self, node: PaperNode) -> str:
        return node.metadata.get("doi") or self._sanitize_title(node.metadata.get("title")) or "unknown"
