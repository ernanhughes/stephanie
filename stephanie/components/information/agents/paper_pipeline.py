# stephanie/components/information/agents/paper_pipeline_agent.py
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from stephanie.agents.base_agent import BaseAgent
from stephanie.components.information.data import (
    DocumentSection,
    PaperReferenceGraph,
    ReferenceRecord,
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
from stephanie.scoring.scorable import Scorable, ScorableType

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
        url = f"https://arxiv.org/abs/{arxiv_id}"

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
    def __init__(self, cfg, memory, container, logger):
        super().__init__(
            cfg=cfg, memory=memory, container=container, logger=logger
        )

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # ------------------------------------------------------------------
        # 0) Resolve root identifier
        # ------------------------------------------------------------------
        arxiv_id = (
            context.get("arxiv_id")
            or context.get("paper_arxiv_id")
        )

        if not arxiv_id:
            arxiv_id = "2511.19900"
            # raise ValueError("PaperPipelineAgent requires 'arxiv_id' in context")

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

        texts = await self._load_texts_for_graph(
            graph=graph,
            papers_root=papers_root,
            context=context,
        )

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

                # 4) Persist into Nexus + KnowledgeGraph (optional but recommended)
        try:
            await self._index_sections_into_graphs(
                arxiv_id=arxiv_id,
                sections=sections,
                context=context,
            )
        except Exception as e:
            log.warning(
                "PaperPipelineGraphIndexError arxiv_id=%s error=%s",
                arxiv_id,
                str(e),
            )


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
        context: Dict[str, Any],
    ) -> Dict[str, str]:
        """
        Load full text for each paper in the graph.

        New behavior:
        - First try to reuse an existing Document from self.memory.documents,
          keyed by a canonical arxiv PDF URL.
        - If not present (or text missing), fall back to PaperImportTask and
          store the resulting Document so subsequent runs are fast and
          consistent.
        """

        import_task = PaperImportTask(
            papers_root=papers_root,
        )

        texts: Dict[str, str] = {}
        stored_documents: List[Dict[str, Any]] = []

        # Optional: pull a goal_id from context if you want it on the Document
        goal = context.get("goal") or context.get("GOAL") or {}
        if isinstance(goal, dict):
            goal_id = goal.get("id")
        else:
            goal_id = getattr(goal, "id", None)

        for arxiv_id, node in graph.nodes.items():
            role = node.role

            # Canonical URL we will use as the key into the DocumentStore.
            # This matches what the HF similar tool uses for arxiv PDFs.
            url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

            # ------------------------------------------------------------------
            # 1) Try memory.documents first (cache hit)
            # ------------------------------------------------------------------
            doc_obj = self.memory.documents.get_by_url(url)
            if doc_obj is not None:
                doc_dict = doc_obj.to_dict()
                text = (doc_dict.get("text") or "").strip()
                if text:
                    texts[arxiv_id] = text
                    stored_documents.append(doc_dict)
                    log.debug(
                        "PaperPipelineAgent: reused cached document for %s (role=%s)",
                        arxiv_id,
                        role,
                    )
                    continue  # move on to next paper

            # ------------------------------------------------------------------
            # 2) Cache miss -> import via PaperImportTask
            # ------------------------------------------------------------------
            try:
                result = await import_task.run(arxiv_id=arxiv_id, role=role)
            except Exception as e:
                log.warning(
                    "PaperPipelineAgent: import failed for %s (role=%s): %s",
                    arxiv_id,
                    role,
                    e,
                )
                continue

            if not result or not getattr(result, "text", None):
                log.warning(
                    "PaperPipelineAgent: no text returned for %s (role=%s)",
                    arxiv_id,
                    role,
                )
                continue

            text = result.text
            texts[arxiv_id] = text

            # ------------------------------------------------------------------
            # 3) Persist into DocumentStore so we don't re-import next time
            # ------------------------------------------------------------------
            try:
                doc: Dict[str, Any] = {
                    "goal_id": goal_id,
                    "title": getattr(result, "title", None)
                    or getattr(node, "title", None)
                    or f"arXiv {arxiv_id}",
                    "external_id": arxiv_id,
                    "summary": getattr(result, "summary", None),
                    "source": "paper_pipeline",
                    "text": text,
                    "url": url,
                }

                stored = self.memory.documents.add_document(doc)
                stored_dict = stored.to_dict()
                stored_documents.append(stored_dict)

                log.debug(
                    "PaperPipelineAgent: stored document for %s (role=%s) into memory.documents",
                    arxiv_id,
                    role,
                )

                # If you want to mirror DocumentLoaderAgent behavior exactly,
                # you can also ensure a Scorable + embedding here, e.g.:
                #
                #   self._ensure_doc_scorable(stored_dict, context)
                #
                # (tiny helper method using BaseEmbeddingStore + Scorable)

            except Exception as e:
                log.warning(
                    "PaperPipelineAgent: failed to store document for %s: %s",
                    arxiv_id,
                    e,
                )

        # Expose docs back on context so later stages can use them
        context["paper_documents"] = stored_documents

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
            container=self.container,
            logger=self.logger,
        )

        async def summarizer(text: str) -> Tuple[str, str]:
            sc = Scorable(text=text, target_type=ScorableType.DOCUMENT_SECTION)
            result: Scorable = await tool.apply(sc, context={})

            summary = result.meta.get("summaries", {}).get("summarizer", {})
            # These keys might differ; adjust to your actual tool output.
            title = summary.get("title", "")
            summary = summary.get("summary", "")

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


        def embedder(text: str):
            embedding = self.memory.embedding.get_or_create(text)
            return embedding

        return embedder


    # ------------------------------------------------------------------
    # Nexus + KnowledgeGraph integration
    # ------------------------------------------------------------------

    async def _index_sections_into_graphs(
        self,
        *,
        arxiv_id: Optional[str],
        sections: List[Any],
        context: Dict[str, Any],
    ) -> None:
        """
        For each DocumentSection:
          - create/update a Nexus scorable + embedding
          - emit a knowledge_graph.index_request event so KGService
            can build entity/claim/relationship nodes.
        """
        if not sections:
            return

        nexus_store = self.memory.nexus
        bus = self.memory.bus
        embedding_store = self.memory.embedding

        enable_nexus = bool(self.cfg.get("enable_nexus_index", True) and nexus_store)
        enable_kg = bool(self.cfg.get("enable_kg_index", True) and bus)

        for idx, sec in enumerate(sections):
            try:
                scorable_id = self._make_section_scorable_id(
                    arxiv_id=arxiv_id, section=sec, index=idx
                )
                text = (
                    getattr(sec, "summary", None)
                    or getattr(sec, "text", None)
                    or ""
                )
                if not text.strip():
                    continue

                domains = list(getattr(sec, "domains", []) or [])
                title = getattr(sec, "title", None)
                section_idx = getattr(sec, "index", None) or idx

                if enable_nexus:
                    row = {
                        "id": scorable_id,
                        "chat_id": None,        # not chat-derived
                        "turn_index": None,
                        "target_type": "document_section",
                        "text": text,
                        "domains": domains,
                        "entities": None,      # KG will fill this over time
                        "meta": {
                            "kind": "paper_section",
                            "arxiv_id": arxiv_id,
                            "title": title,
                            "section_index": section_idx,
                            "paper_pipeline": True,
                        },
                    }
                    nexus_store.upsert_scorable(row)

                    # Optional: store an embedding for fast KNN / LightRAG
                    if embedding_store and self.cfg.get(
                        "index_with_embeddings", True
                    ):
                        try:
                            vec = embedding_store.get_or_create(
                                text
                            )
                            if vec is not None:
                                nexus_store.upsert_embedding(scorable_id, vec)
                        except Exception as e:
                            log.warning(
                                "PaperPipelineEmbeddingError id=%s error=%s",
                                scorable_id,
                                str(e),
                            )

                if enable_kg:
                    await self._publish_kg_index_event(
                        bus=bus,
                        scorable_id=scorable_id,
                        text=text,
                        domains=domains,
                    )

            except Exception as e:
                log.warning(
                    "PaperPipelineSectionIndexError arxiv_id=%s idx=%s error=%s",
                    arxiv_id,
                    idx,
                    str(e),
                )

    def _make_section_scorable_id(
        self,
        *,
        arxiv_id: Optional[str],
        section: Any,
        index: int,
    ) -> str:
        """
        Build a stable id for a section.

        Prefers an existing id/section_id on the DocumentSection if present,
        otherwise falls back to `paper:{arxiv_id}#sec-{index:03d}`.
        """
        # If the section already carries an id, re-use it
        sid = getattr(section, "id", None) or getattr(section, "section_id", None)
        if sid:
            return str(sid)

        prefix = (
            arxiv_id
            or getattr(section, "paper_id", None)
            or getattr(section, "source_id", None)
            or "paper"
        )
        return f"{prefix}#sec-{index:03d}"

    async def _publish_kg_index_event(
        self,
        *,
        bus: Any,
        scorable_id: str,
        text: str,
        domains: List[str],
    ) -> None:
        """
        Emit a minimal event that KnowledgeGraphService can consume.

        KG service will:
          - fetch or use `text`
          - run entity/claim extraction internally
          - upsert entity/claim/gap nodes and relationships.
        """
        envelope = {
            "event_type": "knowledge_graph.index_request",
            "payload": {
                "scorable_id": scorable_id,
                "scorable_type": "document_section",
                "domains": domains,
                "text": text,
            },
        }
        try:
            await bus.publish(
                subject="knowledge_graph.index_request",
                payload=envelope["payload"],
            )
        except Exception as e:
            log.warning(
                "KGIndexEventPublishError scorable_id=%s error=%s",
                scorable_id,
                str(e),
            )
