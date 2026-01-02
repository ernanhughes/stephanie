# stephanie/components/information/paper/pipeline/texts_loader.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from stephanie.components.information.data import PaperReferenceGraph

log = logging.getLogger(__name__)


@dataclass
class PaperTextsLoaderConfig:
    papers_root: Optional[str] = None


class PaperTextsLoader:
    """
    Loads/ensures text for all papers in a PaperReferenceGraph.
    - Ensures the root paper + referenced/similar papers exist in PaperStore.
    - Uses PaperImportTool (provided by runner) to download/ingest if missing.
    - Returns a map of arxiv_id -> text (pdf->text or cached text).
    """

    def __init__(self, *, cfg: Dict[str, Any], paper_store: Any, import_tool: Any):
        self.cfg = cfg
        self.paper_store = paper_store
        self.import_tool = import_tool

        self.papers_root = Path(
            (cfg.get("papers_root") if isinstance(cfg, dict) else None) or "data/papers"
        )

        # local cache so later stages can map arxiv_id -> DocumentORM/record
        self.doc_by_arxiv: Dict[str, Any] = {}

    async def load_texts(self, *, graph: PaperReferenceGraph) -> Dict[str, str]:
        """
        Returns:
            Dict[arxiv_id, text]
        """
        texts: Dict[str, str] = {}

        node_ids = []
        if graph is not None:
            nodes = getattr(graph, "nodes", None) or {}
            node_ids = [str(nid) for nid in nodes.keys()]

        for arxiv_id in node_ids:
            text = await self._ensure_text_for_paper(arxiv_id)
            if text:
                texts[str(arxiv_id)] = text

        return texts

    async def _ensure_text_for_paper(self, arxiv_id: str) -> str:
        """
        Ensure we have text for a paper id, possibly importing it if absent.
        """
        arxiv_id = str(arxiv_id)

        # 1) Prefer store cache if present
        try:
            doc = self.paper_store.get_by_id(arxiv_id)
            if doc is not None:
                self.doc_by_arxiv[arxiv_id] = doc
                txt = getattr(doc, "text", None) or getattr(doc, "content", None)
                if txt:
                    return str(txt)
        except Exception:
            # store may not support this call in some environments
            pass

        # 2) Try local filesystem (if your pipeline has already downloaded)
        pdf_path = self.papers_root / arxiv_id / "paper.pdf"
        if pdf_path.exists():
            # If the store didn't have it, attempt import to populate store + text
            try:
                await self.import_tool.import_paper(arxiv_id=arxiv_id, pdf_path=str(pdf_path))
            except Exception:
                # fallback: allow downstream to run even if import tool fails
                log.warning("PaperTextsLoader: import failed for %s", arxiv_id, exc_info=True)

        # 3) Last resort: ask import tool to fetch by arxiv_id (if implemented)
        try:
            await self.import_tool.import_arxiv_id(arxiv_id=arxiv_id)
        except Exception:
            pass

        # 4) Re-check store
        try:
            doc = self.paper_store.get_by_external_id(arxiv_id)
            if doc is not None:
                self.doc_by_arxiv[arxiv_id] = doc
                txt = getattr(doc, "text", None) or getattr(doc, "content", None)
                if txt:
                    return str(txt)
        except Exception:
            pass

        return ""
