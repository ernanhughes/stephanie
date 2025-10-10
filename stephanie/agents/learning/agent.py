# stephanie/agents/learning/agent.py
from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from typing import Any, Dict, List, Optional, Set

from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.learning.attribution import AttributionTracker
from stephanie.agents.learning.corpus_retriever import CorpusRetriever
from stephanie.agents.learning.evidence import Evidence
from stephanie.agents.learning.knowledge_arena import KnowledgeArena
from stephanie.agents.learning.persistence import Persistence
from stephanie.agents.learning.progress import ProgressAdapter
from stephanie.agents.learning.scoring import Scoring
from stephanie.agents.learning.strategy_manager import StrategyManager
from stephanie.agents.learning.summarizer import Summarizer
from stephanie.services.arena_reporting_adapter import ArenaReporter
from stephanie.utils.progress import AgentProgress

_logger = logging.getLogger(__name__)


class LearningFromLearningAgent(BaseAgent):
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)

        # Progress
        self._progress_core = AgentProgress(self)
        self.progress = ProgressAdapter(self._progress_core, cfg=cfg) 

        # Services
        self.strategy = StrategyManager(cfg, memory, container, logger)
        self.corpus = CorpusRetriever(cfg, memory, container, logger)
        self.scoring = Scoring(cfg, memory, container, logger)
        self.summarizer = Summarizer(
            cfg,
            memory,
            container,
            logger,
            strategy=self.strategy,
            scoring=self.scoring,
            prompt_loader=self.prompt_loader,
            call_llm=self.call_llm,
        )
        self.arena = KnowledgeArena(cfg, memory, container, logger)
        self.arena.score_candidate = self.scoring.score_candidate
        async def _arena_improve(text, meta):
            return await self.summarizer.improve_once(
                paper=meta.get("paper"),
                section=meta.get("section"),
                current_summary=text,
                context=meta.get("context"),
            )

        self.arena.improve = _arena_improve
        self.persist = Persistence(cfg, memory, container, logger)
        self.evidence = Evidence(cfg, memory, container, logger)

        # Attribution/ablation
        self.attribution = AttributionTracker()

        # Flags
        self.use_arena = bool(cfg.get("use_arena", True))
        self.single_random_doc = bool(cfg.get("single_random_doc", False))

        self.reporter = container.get("reporting")
        self.event_service = self.container.get("event")
        
        # Start health monitoring
        self._start_health_monitor()

    def _start_health_monitor(self):
        """Start background task to monitor health metrics"""
        async def _monitor():
            while True:
                try:
                    if hasattr(self.summarizer, 'health_status'):
                        health = self.summarizer.health_status()
                        if health.get("in_flight", 0) > 0:
                            _logger.debug(f"Prompt service status: {health}")
                except Exception as e:
                    _logger.error(f"Health monitoring error: {str(e)}", exc_info=True)
                
                await asyncio.sleep(10)  # Check every 10 seconds
        # Start the monitoring task
        asyncio.create_task(_monitor())

    async def _emit(self, evt: Dict[str, Any]):
        try:
            # push minimal payload; reporter can enrich with ctx
            if self.reporter and hasattr(self.reporter, 'emit'):
                # If reporter is async
                if asyncio.iscoroutinefunction(self.reporter.emit):
                    await self.reporter.emit(
                        context={}, stage="learning", event=evt
                    )
                else:
                    # If reporter is sync, run in executor
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(
                        None, 
                        lambda: self.reporter.emit(
                            context={}, stage="learning", event=evt
                        )
                    )
        except Exception as e:
            _logger.error(f"Failed to emit event: {str(e)}", exc_info=True)



    # ---------- Canonical keys for masking ----------
    @staticmethod
    def _corpus_key(it: Dict[str, Any]) -> str:
        return f"corpus:{str(it.get('id'))}"

    @staticmethod
    def _cand_key(c: Dict[str, Any]) -> str:
        # origin+variant uniquely identify arena candidates we persist
        return f"arena:{c.get('origin', '')}#{c.get('variant', '')}"

    # ---------- Build candidates (supports mask) ----------
    def _build_candidates(
        self,
        section: Dict[str, Any],
        corpus_items: List[Dict[str, Any]],
        *,
        mask_keys: Optional[Set[str]] = None,
    ) -> List[Dict[str, Any]]:
        mask_keys = mask_keys or set()
        out: List[Dict[str, Any]] = []

        # corpus-backed candidates
        for it in (corpus_items or [])[
            : int(self.cfg.get("max_corpus_candidates", 8))
        ]:
            t = (it.get("assistant_text") or "").strip()
            if len(t) < 60:
                continue
            corpus_k = self._corpus_key(it)
            cand_k = f"arena:chat_corpus#c{it.get('id')}"
            if corpus_k in mask_keys or cand_k in mask_keys:
                continue
            out.append(
                {
                    "origin": "chat_corpus",
                    "variant": f"c{it.get('id')}",
                    "text": t,
                    "meta": {
                        "source": "corpus",
                        "corpus_id": it.get("id"),
                        "paper_id": it.get("paper_id"),
                        "section_name": it.get("section_name"),
                        "created_at": it.get("created_at"),
                    },
                }
            )

        # seed candidate from section text
        seed = (section.get("section_text") or "").strip()[:800]
        seed_key = "arena:lfl_seed#seed"
        if len(seed) >= 60 and seed_key not in mask_keys:
            out.append(
                {
                    "origin": "lfl_seed",
                    "variant": "seed",
                    "text": seed,
                    "meta": {
                        "source": "section_text",
                        "paper_id": section.get("paper_id"),
                        "section_name": section.get("section_name"),
                    },
                }
            )

        # deduplicate by normalized text
        seen = set()
        uniq: List[Dict[str, Any]] = []
        for c in out:
            key = " ".join(c["text"].split()).lower()
            if key in seen:
                continue
            seen.add(key)
            uniq.append(c)

        return uniq or (
            [
                {
                    "origin": "lfl_seed",
                    "variant": "seed",
                    "text": seed or "",
                    "meta": {
                        "source": "section_text",
                        "paper_id": section.get("paper_id"),
                        "section_name": section.get("section_name"),
                    },
                }
            ]
            if seed_key not in mask_keys
            else []
        )

    # ---------- Translate supports → mask keys ----------
    def _build_mask_keys_from_supports(
        self, supports: List[Dict[str, Any]]
    ) -> Set[str]:
        """
        supports examples:
        - {"kind":"corpus","id":"123"}
        - {"kind":"arena","origin":"chat_corpus","variant":"c123"}
        """
        out: Set[str] = set()
        for s in supports or []:
            k = s.get("kind")
            if k == "corpus" and s.get("id") is not None:
                out.add(f"corpus:{str(s['id'])}")
                out.add(
                    f"arena:chat_corpus#c{str(s['id'])}"
                )  # mirror arena candidate
            elif k == "arena":
                out.add(f"arena:{s.get('origin', '')}#{s.get('variant', '')}")
        return out

    # ---------- Collect "starred" supports (stub: adapt to your memory schema) ----------
    def _collect_star_supports(self, case_id: int) -> List[Dict[str, Any]]:
        """
        Return a list of supports used to prove applied knowledge.
        Adapt this to match your storage of votes/citations.

        Example output:
        [
          {"kind": "corpus", "id": "123"},
          {"kind": "arena", "origin": "chat_corpus", "variant": "c123"}
        ]
        """
        out: List[Dict[str, Any]] = []
        try:
            for s in self.memory.casebooks.list_scorables(case_id) or []:
                # Example: users starred a retrieved item
                if (
                    s.role == "knowledge_vote"
                    and (s.meta or {}).get("stars", 0) >= 5
                ):
                    corpus_id = (s.meta or {}).get("corpus_id")
                    if corpus_id is not None:
                        out.append({"kind": "corpus", "id": str(corpus_id)})
                # Example: auto-citations from arena
                if s.role == "arena_citations":
                    j = (
                        json.loads(s.text)
                        if isinstance(s.text, str)
                        else (s.text or {})
                    )
                    for c in (j.get("citations") or [])[:2]:
                        origin = c.get("support_origin")
                        variant = c.get("support_variant")
                        if origin and variant:
                            out.append(
                                {
                                    "kind": "arena",
                                    "origin": origin,
                                    "variant": variant,
                                }
                            )
        except Exception:
            pass
        return out

    # ---------- Run a masked re-execution (for ablation) ----------
    async def _run_with_mask(
        self,
        paper: Dict[str, Any],
        section: Dict[str, Any],
        ctx_case: Dict[str, Any],
        mask_keys: Set[str],
    ) -> Dict[str, Any]:
        # 1) corpus with mask (keep attribution wired)
        corpus_items = await self.corpus.fetch(
            section["section_text"],
            mask_keys=mask_keys,
            attribution_tracker=self.attribution,
        )
        # 2) candidates with mask
        cands = self._build_candidates(
            section, corpus_items, mask_keys=mask_keys
        )
        for c in cands:
            c.setdefault("meta", {}).update(
                {"paper": paper, "section": section, "context": ctx_case}
            )
        # 3) (arena|baseline)
        if self.use_arena:
            arena_res = self.arena.run(section["section_text"], cands)
            baseline = arena_res["winner"]["text"]
        else:
            baseline = await self.summarizer.baseline(
                paper, section, corpus_items, ctx_case
            )
        # 4) verify using current strategy
        verify = await self.summarizer.verify_and_improve(
            baseline, paper, section, ctx_case
        )
        return {"baseline": baseline, "verify": verify}

    # ---------- Main ----------
    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.time()
        documents = context.get(self.input_key, []) or []
        if self.single_random_doc:
            doc = self.memory.documents.get_random()
            if not doc:
                return {**context, "error": "no_documents"}
            documents = [
                getattr(
                    doc,
                    "to_dict",
                    lambda: {"id": doc.id, "title": getattr(doc, "title", "")},
                )()
            ]

        out: Dict[str, Any] = {}
        pipeline_run_id = context.get("pipeline_run_id")
        for paper in documents:
            casebook, goal, sections = (
                self.persist.prepare_casebook_goal_sections(paper, context)
            )
            self.progress.start_paper(paper, sections)
            case_context = {
                **context,
                "strategy_version": self.strategy.state.version,
                "verification_threshold": self.strategy.state.verification_threshold,
                "goal": goal,
            }
            results: List[Dict[str, Any]] = []
            for idx, section in enumerate(sections, start=1):
                if not self.persist.section_is_large_enough(section):
                    continue
                case = self.persist.create_section_case(
                    casebook, paper, section, goal, context
                )
                case_context["case_id"] = case.id
                # we construct a specific context for this case  
                self.progress.start_section(section, idx)

                # ----- Corpus -----
                self.progress.stage(section, "corpus:start")
                corpus_items = await self.corpus.fetch(
                    section["section_text"],
                    attribution_tracker=self.attribution,
                )
                self.progress.stage(
                    section, "corpus:done", items=len(corpus_items or [])
                )

                # (Optional) random retrieval ablation for RN metric
                ablate = random.random() < float(
                    self.cfg.get("retrieval_ablate_prob", 0.0)
                )
                if ablate:
                    self.progress.stage(section, "corpus:ablated")
                    try:
                        self.memory.casebooks.add_scorable(
                            case_id=case.id,
                            role="ablation",
                            text=json.dumps(
                                {
                                    "ablation": {
                                        "retrieval": "masked",
                                        "k": len(corpus_items),
                                    }
                                }
                            ),
                            pipeline_run_id=pipeline_run_id,
                            meta={"type": "retrieval_mask"},
                        )
                    except Exception:
                        pass
                    corpus_items_for_baseline = []
                else:
                    corpus_items_for_baseline = corpus_items

                # Attach retrieval pool for downstream attribution (optional)
                case_context["retrieval_items"] = [
                    {
                        "id": it.get("id"),
                        "text": (
                            it.get("assistant_text") or it.get("text") or ""
                        ),
                    }
                    for it in (corpus_items or [])
                ]

                # ----- Draft (arena | baseline) -----
                if self.use_arena:
                    cands = self._build_candidates(
                        section, corpus_items_for_baseline
                    )
                    for c in cands:
                        c.setdefault("meta", {}).update(
                            {
                                "paper": paper,
                                "section": section,
                                "context": case_context,
                            }
                        )

                    arena_meta = {
                        "paper_id": str(
                            paper.get("id") or paper.get("doc_id")
                        ),
                        "section_name": section.get("section_name"),
                        "case_id": case.id,
                        "agent": "LearningFromLearningAgent",
                    }
                    arena_adapter = ArenaReporter(
                        reporting_service=self.reporter,     
                        event_service=self.event_service,            
                        run_id=pipeline_run_id,
                        meta=arena_meta
                    )
                    await arena_adapter.start(context)

                    arena_res = await self.arena.run(
                        section["section_text"],
                        cands,
                        emit=arena_adapter,                      # <-- now the reporter is the emit
                        run_meta={
                            "paper_id": str(
                                paper.get("id") or paper.get("doc_id")
                            ),
                            "section_name": section.get("section_name"),
                            "case_id": case.id,
                            "agent": "LearningFromLearningAgent",
                        },
                        context=case_context
                    )
                    case_context["arena_initial_pool"] = [
                        {
                            "origin": c.get("origin"),
                            "variant": c.get("variant"),
                            "text": c.get("text", ""),
                        }
                        for c in (arena_res.get("initial_pool") or [])
                    ]
                    baseline = arena_res["winner"]["text"]
                    self.persist.persist_arena(
                        case, paper, section, arena_res, case_context
                    )
                    self.progress.stage(
                        section,
                        "arena:done",
                        winner_overall=round(
                            float(
                                arena_res["winner"]["score"].get(
                                    "overall", 0.0
                                )
                            ),
                            3,
                        ),
                    )
                else:
                    baseline = await self.summarizer.baseline(
                        paper, section, corpus_items_for_baseline, case_context
                    )
                    self.progress.stage(section, "baseline:done")

                # ----- Verify & improve -----
                self.progress.stage(section, "verify:start")
                verify = await self.summarizer.verify_and_improve(
                    baseline, paper, section, case_context
                )
                self.progress.stage(
                    section,
                    "verify:done",
                    overall=round(
                        float(verify["metrics"].get("overall", 0.0)), 3
                    ),
                )

                # ----- Persist -----
                self.progress.stage(section, "persist:start")
                saved_case = self.persist.save_section(
                    casebook,
                    paper,
                    section,
                    verify,
                    baseline,
                    goal["id"],
                    case_context,
                )
                self.progress.stage(section, "persist:done")

                # Strategy tracking & knowledge pairs
                self.strategy.track_section(saved_case, verify["iterations"], context)
                self.persist.persist_pairs(
                    saved_case.id, baseline, verify, case_context
                )

                # Progress metrics
                section_metrics = {
                    "overall": verify["metrics"].get("overall", 0.0),
                    "refinement_iterations": len(verify["iterations"]),
                }
                self.progress.end_section(saved_case, section, section_metrics)

                # A/B validation (optional)
                _ = self.strategy.validate_ab(context=case_context)

                # ----- Ablation “proof” (deterministic) -----
                if bool(self.cfg.get("run_proof", False)):
                    star_supports = self._collect_star_supports(case.id)
                    mask_keys = self._build_mask_keys_from_supports(
                        star_supports
                    )

                    with_metrics = verify["metrics"]
                    masked = await self._run_with_mask(
                        paper, section, case_context, mask_keys
                    )
                    without_metrics = masked["verify"]["metrics"]

                    delta = {
                        "overall": with_metrics["overall"]
                        - without_metrics["overall"],
                        "grounding": with_metrics["grounding"]
                        - without_metrics["grounding"],
                        "knowledge": with_metrics["knowledge_score"]
                        - without_metrics["knowledge_score"],
                    }

                    try:
                        self.memory.casebooks.add_scorable(
                            case_id=case.id,
                            role="ablation_result",
                            text=json.dumps(
                                {
                                    "mask": sorted(list(mask_keys)),
                                    "with": with_metrics,
                                    "without": without_metrics,
                                    "delta": delta,
                                }
                            ),
                            pipeline_run_id=pipeline_run_id,
                            meta={"type": "proof"},
                        )
                    except Exception:
                        pass

                # Collect result for this section
                results.append(
                    {
                        "section_name": section["section_name"],
                        "summary": verify["summary"],
                        "metrics": verify["metrics"],
                        "iterations": verify["iterations"],
                    }
                )

            # ----- Evidence / report -----
            longitudinal = self.evidence.collect_longitudinal(context=context)
            cross = self.evidence.cross_episode(context=context)
            report_md = self.evidence.report(longitudinal, cross, context=context)

            paper_out = {
                "paper_id": paper.get("id") or paper.get("doc_id"),
                "title": paper.get("title", ""),
                "results": results,
                "strategy": self.strategy.as_dict(),
                "elapsed_ms": round((time.time() - t0) * 1000, 1),
                "cross_episode": cross,
                "longitudinal_metrics": longitudinal,
                "evidence_report_md": report_md,
            }
            try:
                self.logger.log("LfL_Paper_Run_Complete", paper_out)
            except Exception:
                pass
            out.update(paper_out)

        return {**context, **out}
