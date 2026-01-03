# stephanie/components/information/paper/pipeline/pipeline_runner.py
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from stephanie.components.information.data import PaperReferenceGraph, PaperSection
from stephanie.components.information.tasks.reference_graph_task import ReferenceGraphTask
from stephanie.components.information.tasks.section_build_task import SectionBuildConfig, SectionBuildTask
from stephanie.components.information.tasks.section_link_task import SectionLinkTask

from .providers import LocalJsonReferenceProvider, HFSimilarPaperProvider
from .texts_loader import PaperTextsLoader
from .sections_cache import PaperSectionsCache
from .section_features import SectionFeatureDispatcher
from .graph_indexer import PaperSectionGraphIndexer

log = logging.getLogger(__name__)


class PaperPipelineRunner:
    def __init__(self, *, cfg, memory, container, logger, import_tool):
        self.cfg = cfg
        self.memory = memory
        self.container = container
        self.logger = logger
        self.import_tool = import_tool

        self.texts_loader = PaperTextsLoader(cfg=cfg, paper_store=memory.papers, import_tool=import_tool)
        self.sections_cache = PaperSectionsCache(cfg=cfg, paper_store=memory.papers, logger=logger)
        self.feature_dispatcher = SectionFeatureDispatcher(cfg=cfg, memory=memory, logger=logger)
        self.graph_indexer = PaperSectionGraphIndexer(cfg=cfg, memory=memory, logger=logger)

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        arxiv_id = context.get("arxiv_id") or context.get("paper_arxiv_id")
        if not arxiv_id:
            arxiv_id = "2506.21734"
            context["arxiv_id"] = arxiv_id

        max_refs: Optional[int] = context.get("max_refs", 100)
        max_similar: int = int(context.get("max_similar", 8))

        papers_root = Path(self.cfg.get("papers_root", "data/papers"))
        context["papers_root"] = str(papers_root)
        context["paper_pdf_path"] = str(papers_root / f"{arxiv_id}/paper.pdf")

        # Providers
        ref_provider = LocalJsonReferenceProvider(
            papers_root=papers_root,
            max_refs=int(self.cfg.get("graph", {}).get("max_refs", 50)),
        )
        sim_provider = None
        if self.cfg.get("enable_hf_similar", True):
            sim_provider = HFSimilarPaperProvider(
                max_limit=int(self.cfg.get("hf_similar_max", 16))
            )

        # 1) Graph
        graph_task = ReferenceGraphTask(
            papers_root=papers_root,
            import_tool=self.import_tool,
            ref_provider=ref_provider,
            similar_provider=sim_provider,
        )
        graph: PaperReferenceGraph = await graph_task.run(
            root_arxiv_id=arxiv_id,
            max_refs=max_refs,
            max_similar=max_similar,
        )

        # 2) Load texts
        texts = await self.texts_loader.load_texts(graph=graph)

        # 3) Sections (cache → build)
        sections: Optional[List[PaperSection]] = self.sections_cache.maybe_load(arxiv_id, role="reference")
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
            sections = await section_task.run(graph=graph, texts=texts)

        # attach casebooks/cases for every section (cached skip makes this cheap on reruns)
        changed = self._ensure_section_casebooks(arxiv_id=arxiv_id, sections=sections, context=context)

        # persist if we built OR if we attached case metadata
        if sections is not None and (changed or context.get("_sections_built", False)):
            self.sections_cache.persist(arxiv_id=arxiv_id, sections=sections)


        # 4) Link + clusters
        link_task = SectionLinkTask(
            root_arxiv_id=arxiv_id,
            top_k=int(self.cfg.get("section_top_k", 5)),
            min_sim=float(self.cfg.get("section_min_sim", 0.4)),
        )
        matches, clusters = link_task.run(sections)

        # 5) Feature jobs
        await self.feature_dispatcher.dispatch(sections=sections, context=context)

        # 6) Index into Nexus/KG
        try:
            await self.graph_indexer.index(arxiv_id=arxiv_id, sections=sections, context=context)
        except Exception as e:
            log.warning("PaperPipelineGraphIndexError arxiv_id=%s error=%s", arxiv_id, str(e))

        # 7) context outputs
        context["paper_graph"] = graph
        context["paper_sections"] = sections
        context["section_matches"] = matches
        context["concept_clusters"] = clusters
        return context

    def _get_summarizer(self):
        tool = self.memory.tools.summarizer if hasattr(self.memory, "tools") else None
        if tool is None:
            from stephanie.tools.summarization_tool import SummarizationTool
            tool = SummarizationTool(cfg=self.cfg.get("summarizer", {}), memory=self.memory, container=self.container, logger=self.logger)

        async def summarizer(text: str) -> Tuple[str, str]:
            from stephanie.scoring.scorable import Scorable, ScorableType
            sc = Scorable(text=text, target_type=ScorableType.DOCUMENT_SECTION)
            result = await tool.apply(sc, context={})
            summary_obj = result.meta.get("summaries", {}).get("summarizer", {})
            title = summary_obj.get("title", "") or (text[:80] if text else "")
            summary = summary_obj.get("summary", "") or (text[:512] if text else "")
            return title, summary

        return summarizer

    def _get_embedder(self):
        def embedder(text: str):
            return self.memory.embedding.get_or_create(text)
        return embedder

    def _sha1(self, s: str) -> str:
        return hashlib.sha1((s or "").encode("utf-8", errors="ignore")).hexdigest()

    def _sec_id(self, sec: Any) -> str:
        return str(getattr(sec, "section_id", None) or getattr(sec, "id", None) or f"sec-{getattr(sec, 'section_index', 0)}")

    def _ensure_section_casebooks(self, *, arxiv_id: str, sections: list, context: dict) -> bool:
        """
        Ensures:
        - a CaseBook per section
        - a Case per section (deduped via section_hash)
        Writes:
        sec.meta['casebook_id'], sec.meta['case_id'], sec.meta['section_hash'], sec.meta['casebook_name']
        Returns True if anything changed (so caller can persist sections).
        """
        casebooks = self.memory.casebooks

        run_id = context.get("pipeline_run_id")
        run_id = int(run_id) if run_id is not None else 0

        goal = context.get("goal") or {}
        changed = False

        for sec in sections:
            text = getattr(sec, "text", "") or ""
            if not text.strip():
                continue

            meta = getattr(sec, "meta", None) or {}
            section_hash = self._sha1(text)

            # ✅ Hard cache: if we already attached ids for this exact text, skip DB work entirely
            if (
                meta.get("section_hash") == section_hash
                and meta.get("casebook_id")
                and meta.get("case_id")
            ):
                continue

            sec_id = self._sec_id(sec)
            cb_name = f"paper:{arxiv_id}:{sec_id}"

            cb = casebooks.ensure_casebook(
                name=cb_name,
                pipeline_run_id=run_id,
                description=f"Paper section node: {arxiv_id} / {sec_id}",
                tags=["paper", "section", str(arxiv_id)],
                meta={
                    "paper_arxiv_id": str(arxiv_id),
                    "section_id": sec_id,
                    "section_index": getattr(sec, "section_index", None),
                    "paper_role": getattr(sec, "paper_role", None) or getattr(sec, "role", None),
                    "title": getattr(sec, "title", None),
                },
            )

            # ✅ Soft dedup: if latest case already matches section_hash, reuse it
            last = casebooks.list_cases(casebook_id=cb.id, agent_name="section_build_task", limit=1)
            if last and (last[0].meta or {}).get("section_hash") == section_hash:
                case = last[0]
            else:
                scorables = [
                    {
                        "text": text,
                        "target_type": "document_section",
                        "role": "section",
                        "meta": {
                            "paper_arxiv_id": str(arxiv_id),
                            "section_id": sec_id,
                            "section_index": getattr(sec, "section_index", None),
                            "paper_role": getattr(sec, "paper_role", None) or getattr(sec, "role", None),
                        },
                    }
                ]

                summary = getattr(sec, "summary", None) or ""
                if summary.strip():
                    scorables.append(
                        {
                            "text": summary,
                            "target_type": "document_section_summary",
                            "role": "summary",
                            "meta": {
                                "paper_arxiv_id": str(arxiv_id),
                                "section_id": sec_id,
                            },
                        }
                    )

                case = casebooks.add_case(
                    casebook_id=cb.id,
                    goal_id=goal.get("id"),  # nullable is fine
                    agent_name="section_build_task",
                    name=sec_id,
                    description=getattr(sec, "title", None) or "",
                    prompt_text=json.dumps({"arxiv_id": str(arxiv_id), "section_id": sec_id}),
                    scorables=scorables,
                    meta={
                        "section_hash": section_hash,
                        "paper_arxiv_id": str(arxiv_id),
                        "section_id": sec_id,
                        "section_index": getattr(sec, "section_index", None),
                    },
                )

            meta.update(
                {
                    "section_hash": section_hash,
                    "casebook_name": cb_name,
                    "casebook_id": cb.id,
                    "case_id": case.id,
                }
            )
            sec.meta = meta
            changed = True

        return changed
