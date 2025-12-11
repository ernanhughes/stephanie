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

from stephanie.tools.summarization_tool import SummarizationTool
from stephanie.tools.huggingface_tool import recommend_similar_papers
from stephanie.scoring.scorable import Scorable, ScorableType
from stephanie.constants import PIPELINE_RUN_ID

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
        self.document_store = memory.documents
        self.section_store = memory.document_sections
        # populated by _load_texts_for_graph so we know which Document
        # corresponds to which arxiv_id
        self._doc_by_arxiv: Dict[str, Any] = {}


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
            context["arxiv_id"] = arxiv_id
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
        #     – but first try to re-use cached sections if we have them.
        # ------------------------------------------------------------------

        texts = await self._load_texts_for_graph(graph, papers_root)

        sections: Optional[List[DocumentSection]] = None

        # Try to reuse sections from the DB for the root paper
        if self.section_store and self.document_store:
            sections = self._maybe_load_sections_from_store(arxiv_id)

        if sections is None:
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

            sections = await section_task.run(
                graph=graph,
                texts=texts,
            )

            # Persist freshly built sections so future runs can re-use them
            self._persist_sections_to_store(
                arxiv_id=arxiv_id,
                sections=sections,
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
        # 3) Link sections (root vs others) + concept clusters
        # ------------------------------------------------------------------

        link_task = SectionLinkTask(
            root_arxiv_id=arxiv_id,
            top_k=int(self.cfg.get("section_top_k", 5)),
            min_sim=float(self.cfg.get("section_min_sim", 0.4)),
        )

        matches, clusters = link_task.run(sections)
        
        await self._dispatch_feature_jobs(sections, context)

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
    ) -> Dict[str, str]:
        """
        Load full text for each paper in the graph.

        Behaviour now:
          - If a Document already exists (by URL) and has text, reuse it.
          - Otherwise, run PaperImportTask, then create a Document row.
          - Maintain self._doc_by_arxiv[arxiv_id] for later use.
        """
        texts: Dict[str, str] = {}
        self._doc_by_arxiv = {}

        import_task = PaperImportTask(
            papers_root=papers_root,
        )

        for arxiv_id, node in graph.nodes.items():
            text = None
            doc = None
            url = self._guess_paper_url(arxiv_id, node)

            # 1) Try to reuse stored Document if available
            if self.document_store:
                try:
                    doc = self.document_store.get_by_url(url)
                except Exception as e:
                    log.warning(
                        "PaperPipeline: document lookup failed url=%s error=%s",
                        url,
                        e,
                    )
                    doc = None

                if doc is not None and getattr(doc, "text", None):
                    text = doc.text

            # 2) If no stored text, import the paper
            if text is None:
                try:
                    res = await import_task.run(
                        arxiv_id=arxiv_id,
                        role=getattr(node, "role", None),
                    )
                    text = getattr(res, "text", None)
                except Exception as e:
                    log.warning(
                        "Failed to load text for %s (role=%s): %s",
                        arxiv_id,
                        getattr(node, "role", None),
                        e,
                    )
                    continue

                # 2b) Persist a Document row if we have a store
                if self.document_store and text:
                    try:
                        if doc is None:
                            title = getattr(node, "title", None) or arxiv_id
                            source = getattr(node, "source", None) or "arxiv"

                            doc_dict = {
                                "title": title,
                                "source": source,
                                "external_id": arxiv_id,
                                "url": url,
                                "summary": None,   # let other agents fill this in
                                "text": text,
                                "goal_id": None,
                                "domains": None,
                            }
                            doc = self.document_store.add_document(doc_dict)
                    except Exception as e:
                        log.warning(
                            "PaperPipeline: failed to persist Document for %s: %s",
                            arxiv_id,
                            e,
                        )

            if text:
                texts[arxiv_id] = text
            if doc is not None:
                self._doc_by_arxiv[arxiv_id] = doc

        return texts

    def _guess_paper_url(self, arxiv_id: str, node: Any) -> str:
        """
        Best-effort reconstruction of a stable URL for this paper.
        This should line up with what DocumentLoader uses for arxiv PDFs.
        """
        url = getattr(node, "pdf_url", None) or getattr(node, "url", None)
        if not url and arxiv_id:
            # Default arxiv PDF pattern
            url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        return url

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

    # ------------------------------------------------------------------
    # Feature indexing helpers (NexusFeatureWorker)
    # ------------------------------------------------------------------

    def _build_section_scorables_for_features(
        self,
        sections: List[DocumentSection],
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Convert DocumentSection objects into Scorable dicts that
        NexusFeatureWorker / ScorableProcessor can understand.
        """
        cfg_features = self.cfg.get("features", {}) or {}
        target_type = cfg_features.get("target_type", "document_section")
        source = cfg_features.get("source", "paper_pipeline")

        arxiv_id = (
            context.get("arxiv_id")
            or context.get("paper_arxiv_id")
            or context.get("root_arxiv_id")
        )

        scorables: List[Dict[str, Any]] = []

        for idx, sec in enumerate(sections):
            text = getattr(sec, "summary", None) or getattr(sec, "text", None) or ""
            if not text or not text.strip():
                continue

            # Try to reuse an existing scorable_id if we already set one
            scorable_id = (
                getattr(sec, "scorable_id", None)
                or getattr(sec, "id", None)
            )
            # If your earlier code has a helper like _make_section_scorable_id,
            # you can fall back to that:
            if not scorable_id and hasattr(self, "_make_section_scorable_id"):
                scorable_id = self._make_section_scorable_id(sec, idx)

            if not scorable_id:
                # last resort: deterministic id from arxiv + index
                scorable_id = f"{arxiv_id or 'paper'}:sec:{idx}"
                setattr(sec, "scorable_id", scorable_id)

            domains = getattr(sec, "domains", None) or []
            title = getattr(sec, "title", None)
            external_id = getattr(sec, "external_id", None) or arxiv_id

            scorable = {
                "id": scorable_id,
                "target_type": target_type,
                "text": text,
                "title": title,
                "external_id": external_id,
                "order_index": idx,
                "domains": domains,
                "source": source,
            }
            scorables.append(scorable)

        return scorables

    async def _dispatch_feature_jobs(
        self,
        sections: List[DocumentSection],
        context: Dict[str, Any],
    ) -> None:
        """
        Send section scorables to NexusFeatureWorker so features/embeddings
        are computed and mirrored into Nexus.

        This uses the bus subject configured in cfg["features"]["subject"].
        """
        cfg_features = self.cfg.get("features", {}) or {}
        if not cfg_features.get("enabled", True):
            log.debug("PaperPipelineAgent: feature indexing disabled in config")
            return

        bus = getattr(self.memory, "bus", None)
        if bus is None:
            log.warning(
                "PaperPipelineAgent: memory.bus is missing; "
                "cannot dispatch feature jobs to NexusFeatureWorker"
            )
            return

        scorables = self._build_section_scorables_for_features(sections, context)
        if not scorables:
            log.debug(
                "PaperPipelineAgent: no non-empty sections to index for features"
            )
            return

        subject = cfg_features.get("subject", "nexus.features.index_request")

        pipeline_run_id = (
            context.get(PIPELINE_RUN_ID)
            or context.get("pipeline_run_id")
            or context.get("run_id")
        )

        payload_context = {
            PIPELINE_RUN_ID: pipeline_run_id,
            "arxiv_id": (
                context.get("arxiv_id")
                or context.get("paper_arxiv_id")
                or context.get("root_arxiv_id")
            ),
            "goal": context.get("goal"),
        }

        payload = {
            "scorables": scorables,
            "context": payload_context,
        }

        try:
            await bus.publish(subject=subject, payload=payload)
            log.info(
                "PaperPipelineAgent: dispatched %d section scorables "
                "to NexusFeatureWorker on subject=%s",
                len(scorables),
                subject,
            )
        except Exception as e:
            log.exception(
                "PaperPipelineAgent: failed to publish feature index batch: %s",
                e,
            )

    def _maybe_load_sections_from_store(
        self,
        arxiv_id: str,
    ) -> Optional[List[DocumentSection]]:
        """
        If we already have stored sections for this paper, reconstruct
        lightweight DocumentSection objects and return them.

        Returns None if:
          - there's no Document or no sections yet
          - or cfg.force_rebuild_sections is True
        """
        if not self.document_store or not self.section_store:
            return None

        doc = self._doc_by_arxiv.get(arxiv_id)
        if doc is None:
            return None

        try:
            orm_sections = self.section_store.get_by_document(doc.id)
        except Exception as e:
            log.warning(
                "PaperPipeline: failed reading sections for doc_id=%s error=%s",
                getattr(doc, "id", None),
                e,
            )
            return None

        if not orm_sections:
            return None

        if bool(self.cfg.get("force_rebuild_sections", False)):
            log.info(
                "PaperPipeline: force_rebuild_sections=True; ignoring %d cached sections for %s",
                len(orm_sections),
                arxiv_id,
            )
            return None

        sections = self._sections_from_orm(orm_sections, arxiv_id=arxiv_id)
        log.info(
            "PaperPipeline: reusing %d cached sections for %s",
            len(sections),
            arxiv_id,
        )
        return sections

    def _sections_from_orm(
        self,
        orm_sections,
        *,
        arxiv_id: str,
    ) -> List[DocumentSection]:
        """
        Convert DocumentSectionORM rows into DocumentSection objects.

        This version explicitly fills the required ctor fields:
          - paper_arxiv_id
          - paper_role
          - section_index

        and then uses signature introspection to only pass args that
        actually exist on DocumentSection in your codebase.
        """
        from inspect import signature, Parameter

        sig = signature(DocumentSection)
        param_names = [
            p.name
            for p in sig.parameters.values()
            if p.kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)
        ]

        sections: List[DocumentSection] = []

        for idx, orm in enumerate(orm_sections):
            extra = getattr(orm, "extra_data", None) or {}

            # --- required ctor fields ----------------------------------
            paper_arxiv_id = (
                extra.get("paper_arxiv_id")
                or extra.get("arxiv_id")
                or arxiv_id
            )
            paper_role = extra.get("paper_role") or "root"
            section_index = (
                extra.get("section_index")
                or extra.get("index")
                or idx
            )

            # --- common fields we may want to restore ------------------
            title = getattr(orm, "section_name", None)
            text = getattr(orm, "section_text", None)
            summary = getattr(orm, "summary", None)
            source = getattr(orm, "source", None)

            candidates = {
                # required ctor args
                "paper_arxiv_id": paper_arxiv_id,
                "paper_role": paper_role,
                "section_index": section_index,
                # ids
                "id": orm.id,
                "section_id": orm.id,
                "document_id": orm.document_id,
                # text-ish fields
                "title": title,
                "section_name": title,
                "text": text,
                "section_text": text,
                "summary": summary,
                "source": source,
                # misc / meta
                "extra_data": extra,
                "arxiv_id": arxiv_id,
                "paper_id": arxiv_id,
                "index": idx,
            }

            kwargs = {
                name: value
                for name, value in candidates.items()
                if name in param_names and value is not None
            }

            try:
                sec = DocumentSection(**kwargs)
            except TypeError as e:
                # Extremely defensive fallback: only pass the 3 required args,
                # then patch the obvious attributes onto the instance.
                self.logger.warning(
                    "PaperPipeline: DocumentSection(**kwargs) failed (%s), "
                    "falling back to minimal ctor; kwargs=%s",
                    e,
                    kwargs,
                )
                sec = DocumentSection(
                    paper_arxiv_id=paper_arxiv_id,
                    paper_role=paper_role,
                    section_index=section_index,
                )
                # best-effort attribute patching
                if title is not None and not getattr(sec, "section_name", None):
                    setattr(sec, "section_name", title)
                if text is not None and not getattr(sec, "text", None):
                    setattr(sec, "text", text)
                if summary is not None and not getattr(sec, "summary", None):
                    setattr(sec, "summary", summary)
                if source is not None and not getattr(sec, "source", None):
                    setattr(sec, "source", source)
                if extra and not getattr(sec, "extra_data", None):
                    setattr(sec, "extra_data", extra)

            sections.append(sec)

        return sections

    def _persist_sections_to_store(
        self,
        *,
        arxiv_id: str,
        sections: List[DocumentSection],
    ) -> None:
        """
        Persist freshly built sections into the DocumentSectionStore.

        - Attaches them to the Document we created/loaded earlier.
        - Writes minimal metadata into extra_data.
        - Pushes stored.id back onto the in-memory section (for stable scorable_ids).
        """
        if not self.section_store or not self.document_store:
            return
        if not sections:
            return

        doc = self._doc_by_arxiv.get(arxiv_id)
        if doc is None:
            log.info(
                "PaperPipeline: no Document found for %s; skipping section persistence",
                arxiv_id,
            )
            return

        # If sections already exist and we're not forcing, bail early
        try:
            existing = self.section_store.get_by_document(doc.id)
        except Exception:
            existing = []

        if existing and not bool(self.cfg.get("force_rebuild_sections", False)):
            log.info(
                "PaperPipeline: %d sections already stored for doc_id=%s; not overwriting",
                len(existing),
                getattr(doc, "id", None),
            )
            return

        if existing:
            self.section_store.delete_by_document(doc.id)

        for idx, sec in enumerate(sections):
            title = getattr(sec, "title", None) or getattr(sec, "section_name", None)
            if not title:
                title = (getattr(sec, "summary", "") or getattr(sec, "text", ""))[:80]

            text = getattr(sec, "text", None) or getattr(sec, "summary", None) or ""
            summary = getattr(sec, "summary", None) or ""

            extra_data = {
                "arxiv_id": arxiv_id,
                "paper_arxiv_id": arxiv_id,
                "paper_role": getattr(sec, "paper_role", None) or "root",
                "section_index": getattr(sec, "section_index", None) or idx,
                "index": getattr(sec, "index", idx),
                "domains": list(getattr(sec, "domains", []) or []),
            }
            meta = getattr(sec, "meta", None)
            if isinstance(meta, dict):
                extra_data["meta"] = meta

            section_dict = {
                "document_id": doc.id,
                "section_name": title[:255],
                "section_text": text,
                "summary": summary,
                "source": "paper_pipeline",
                "extra_data": extra_data,
            }

            try:
                stored = self.section_store.upsert(section_dict)
                # Push the DB id back onto the in-memory section so that
                # _make_section_scorable_id will now use a stable id.
                if not getattr(sec, "id", None) and not getattr(sec, "section_id", None):
                    try:
                        setattr(sec, "id", stored.id)
                    except Exception:
                        pass
            except Exception as e:
                log.warning(
                    "PaperPipeline: failed to persist section '%s' idx=%d: %s",
                    title,
                    idx,
                    e,
                )
