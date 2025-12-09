# stephanie/components/information/agents/paper_pipeline_agent.py
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from stephanie.agents.base_agent import BaseAgent
from stephanie.components.information.data import (
    ConceptCluster,
    DocumentSection,
    PaperReferenceGraph,
    ReferenceRecord,
    SectionMatch,
)
from stephanie.components.information.tasks.paper_import_task import (
    PaperImportTask,
)
from stephanie.components.information.tasks.reference_graph_task import (
    ReferenceGraphTask,
    ReferenceProvider,
    SimilarPaperProvider,
)
from stephanie.components.information.tasks.section_build_task import (
    SectionBuildTask,
    SectionBuildConfig,
)
from stephanie.components.information.tasks.section_link_task import SectionLinkTask

# Tool + memory imports – these are real modules in your project,
# but you may need to tweak exact names / paths.
from stephanie.tools.summarization_tool import SummarizationTool
from stephanie.tools.huggingface_tool import recommend_similar_papers
from stephanie.memory.base_embedding_store import BaseEmbeddingStore
from stephanie.scoring.scorable import Scorable

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------


class DummyReferenceProvider(ReferenceProvider):
    """
    Minimal reference provider: returns no references.

    This is SAFE for local directory runs and lets you get value out of the
    pipeline immediately. Later, you can replace this with a real provider
    that uses:
      - your PDF reference extractor, or
      - an external API (OpenAlex, arXiv metadata, etc.)
    """

    def get_references_for_arxiv(self, arxiv_id: str) -> List[ReferenceRecord]:
        return []


class HFSimilarPaperProvider(SimilarPaperProvider):
    """
    Similar-paper provider using your HuggingFace Tool.

    It calls recommend_similar_papers(paper_url) and maps the results into
    ReferenceRecord objects.
    """

    def __init__(self, max_limit: int = 16) -> None:
        self.max_limit = max_limit

    def get_similar_for_arxiv(self, arxiv_id: str, limit: int = 10) -> List[ReferenceRecord]:
        limit = min(limit, self.max_limit)
        url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        try:
            hits = recommend_similar_papers(paper_url=url)
        except Exception as e:
            log.warning("HF similar papers failed for %s: %s", arxiv_id, e)
            return []

        recs: List[ReferenceRecord] = []
        for h in hits[:limit]:
            h_url = h.get("url") or h.get("paper_url") or ""
            if not h_url:
                continue

            # Try to extract an arxiv-like id from the URL.
            # Example patterns:
            #   https://arxiv.org/pdf/2505.08827.pdf
            #   https://arxiv.org/pdf/2505.08827
            m = re.search(r"/(\d{4}\.\d{4,5})(?:\.pdf)?$", h_url)
            if not m:
                # fall back to title if it looks like an id
                title = h.get("title", "")
                m2 = re.search(r"(\d{4}\.\d{4,5})", title)
                if not m2:
                    continue
                pid = m2.group(1)
            else:
                pid = m.group(1)

            recs.append(
                ReferenceRecord(
                    arxiv_id=pid,
                    title=h.get("title"),
                    url=h_url,
                    source="hf_similar",
                    raw=h,
                )
            )

        return recs


# ---------------------------------------------------------------------------
# PaperPipelineAgent
# ---------------------------------------------------------------------------


class PaperPipelineAgent(BaseAgent):
    """
    High-level agent that runs the full paper pipeline:

        arxiv_id -> graph -> sections -> cross-paper links

    Expected context input keys:
        - "arxiv_id" (or "paper_arxiv_id")
        - optional: "max_refs", "max_similar"

    It writes:
        - context["paper_graph"]
        - context["paper_sections"]
        - context["section_matches"]
        - context["concept_clusters"]

    NOTE: For now this agent is arXiv-centric. If you want to iterate over a
    directory of local PDFs, you can:
        - treat each PDF as having an "arxiv_id" equal to its stem, or
        - build a small wrapper agent that uses PaperImportTask + SectionBuildTask
          directly for non-arxiv PDFs.
    """

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # ------------------------------------------------------------------
        # 0) Resolve root identifier
        # ------------------------------------------------------------------
        arxiv_id = (
            context.get("arxiv_id")
            or context.get("paper_arxiv_id")
        )

        if not arxiv_id:
            raise ValueError("PaperPipelineAgent requires 'arxiv_id' in context")

        max_refs: Optional[int] = context.get("max_refs")
        max_similar: int = int(context.get("max_similar", 8))

        # ------------------------------------------------------------------
        # 1) Build paper graph (root + references + similar)
        # ------------------------------------------------------------------

        papers_root = Path(
            self.cfg.get("papers_root", "data/papers")
        )

        ref_provider: ReferenceProvider = self._get_reference_provider()
        sim_provider: Optional[SimilarPaperProvider] = self._get_similar_provider()

        import_task = PaperImportTask(
            papers_root=papers_root,
        )

        graph_task = ReferenceGraphTask(
            papers_root=papers_root,
            import_task=import_task,
            ref_provider=ref_provider,
            similar_provider=sim_provider,
        )

        graph: PaperReferenceGraph = await graph_task.run(
            root_arxiv_id=arxiv_id,
            max_refs=max_refs,
            max_similar=max_similar,
        )

        # ------------------------------------------------------------------
        # 2) Build sections (slice text + optional summarization/embedding)
        # ------------------------------------------------------------------

        texts = await self._load_texts_for_graph(graph, papers_root)

        section_cfg = SectionBuildConfig(
            chars_per_section=int(self.cfg.get("section_chars", 2000)),
            min_chars=int(self.cfg.get("section_min_chars", 400)),
            overlap=int(self.cfg.get("section_overlap", 200)),
        )

        section_task = SectionBuildTask(
            cfg=section_cfg,
            summarizer=self._get_summarizer(),
            embedder=self._get_embedder(),
        )

        sections: List[DocumentSection] = await section_task.run(
            graph=graph,
            texts=texts,
        )

        # ------------------------------------------------------------------
        # 3) Link sections (root vs others) + concept clusters
        # ------------------------------------------------------------------

        link_task = SectionLinkTask(
            root_arxiv_id=arxiv_id,
            top_k=int(self.cfg.get("section_top_k", 5)),
            min_sim=float(self.cfg.get("section_min_sim", 0.4)),
        )

        matches, clusters = link_task.run(sections)

        # ------------------------------------------------------------------
        # 4) Write results back to context
        # ------------------------------------------------------------------

        context["paper_graph"] = graph
        context["paper_sections"] = sections
        context["section_matches"] = matches
        context["concept_clusters"] = clusters

        return context

    # ------------------------------------------------------------------ #
    # Wiring helpers – you’ll adapt these to your actual services.
    # ------------------------------------------------------------------ #

    def _get_reference_provider(self) -> ReferenceProvider:
        """
        TODO: wire a real provider that returns ReferenceRecord objects.

        For now we return DummyReferenceProvider so the pipeline works even
        if you haven't implemented reference extraction yet.
        """
        return DummyReferenceProvider()

    def _get_similar_provider(self) -> Optional[SimilarPaperProvider]:
        """
        Returns HFSimilarPaperProvider wrapped around recommend_similar_papers.
        Set to None to disable similar-paper expansion.
        """
        if not self.cfg.get("enable_hf_similar", True):
            return None
        return HFSimilarPaperProvider(max_limit=int(self.cfg.get("hf_similar_max", 16)))

    async def _load_texts_for_graph(
        self,
        graph: PaperReferenceGraph,
        papers_root: Path,
    ) -> Dict[str, str]:
        """
        Load full text for each paper in the graph.

        For now, we simply re-use PaperImportTask to re-import each paper.
        This does a second pass, but it's simple and robust. If you prefer,
        you can:
            - cache text on disk, or
            - load from your DB using node.pdf_path / text_hash.
        """
        texts: Dict[str, str] = {}

        import_task = PaperImportTask(
            papers_root=papers_root,
        )

        for arxiv_id, node in graph.nodes.items():
            try:
                res = await import_task.run(
                    arxiv_id=arxiv_id,
                    role=node.role,
                )
                texts[arxiv_id] = res.text
            except Exception as e:
                log.warning(
                    "Failed to load text for %s (role=%s): %s",
                    arxiv_id,
                    node.role,
                    e,
                )

        return texts

    def _get_summarizer(self):
        """
        Wrap your SummarizationTool into an async (text -> (title, summary)) fn.

        Assumptions (adjust if different in your codebase):
            - SummarizationTool is a BaseTool with `async run(scorable, **kwargs)`
            - It returns a dict with keys "title" and "summary" (or similar).
        """

        cfg = self.cfg.get("summarizer", {})
        tool = SummarizationTool(
            cfg=cfg,
            memory=self.memory,
            logger=self.logger,
        )

        async def summarizer(text: str) -> Tuple[str, str]:
            sc = Scorable.from_text(text, source="paper_section")
            result: Dict[str, Any] = await tool.run(sc)

            # These keys might differ; adjust to your actual tool output.
            title = result.get("title") or result.get("headline") or ""
            summary = result.get("summary") or result.get("text") or ""

            # Fallbacks, just in case
            if not title:
                title = summary[:80] if summary else text[:80]
            if not summary:
                summary = text[:512]

            return title, summary

        return summarizer

    def _get_embedder(self):
        """
        Wrap your BaseEmbeddingStore into an async (text -> embedding) fn.

        Assumptions:
            - You have a BaseEmbeddingStore-like instance reachable from memory,
              or you can construct one here with your existing configuration.
            - It has a method like `get_or_create_embedding(text, cfg)` or similar.

        Because BaseEmbeddingStore is highly project-specific, you *will* need
        to adapt this to your real API. The skeleton below shows the pattern.
        """

        # Example: obtain an existing store from memory
        # (adjust this to match your own memory / store wiring).
        store: BaseEmbeddingStore = getattr(self.memory, "embedding_store", None)
        if store is None:
            log.warning(
                "[PaperPipelineAgent] No embedding_store on memory; skipping embeddings"
            )
            return None

        async def embedder(text: str):
            # You may need to adjust this method name:
            #   - get_or_create_embedding
            #   - get_embedding
            #   - embed_text
            # etc.
            try:
                # Example guess – change to what your store actually uses.
                embedding = store.get_or_create_embedding(text)
            except AttributeError:
                log.error(
                    "BaseEmbeddingStore has no 'get_or_create_embedding' method. "
                    "Please update _get_embedder() to use the correct API."
                )
                raise
            return embedding

        return embedder
