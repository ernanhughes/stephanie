# stephanie/agents/learning/agent.py
from __future__ import annotations
from typing import Dict, Any, List
import time
import logging

from stephanie.agents.base_agent import BaseAgent
from .strategy_manager import StrategyManager
from .corpus_service import CorpusService
from .summarizer import Summarizer
from .arena import ArenaService   # ← fixed
from .persistence import Persistence
from .evidence import Evidence
from .progress import ProgressAdapter     # ← fixed
from .scoring import Scoring              # ← scoring for arena callbacks
from stephanie.utils.progress import AgentProgress

_logger = logging.getLogger(__name__)

class LearningFromLearningAgent(BaseAgent):
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self._progress_core = AgentProgress(self)
        self.progress = ProgressAdapter(self._progress_core, cfg=cfg)
        self.strategy = StrategyManager(cfg, memory, container, logger)
        self.corpus = CorpusService(cfg, memory, container, logger)
        # scoring service (used by summarizer & arena)
        self.scoring = Scoring(cfg, memory, container, logger)
        # summarizer uses scoring + your agent’s prompt/call hooks
        self.summarizer = Summarizer(cfg, memory, container, logger, strategy=self.strategy,
                                     scoring=self.scoring,
                                     prompt_loader=getattr(self, "prompt_loader", None),
                                     call_llm=getattr(self, "call_llm", None))
        # arena with injected callbacks
        self.arena = ArenaService(cfg, memory, container, logger)
        self.arena.score_candidate = self.scoring.score_candidate
        self.arena.improve = lambda text, meta: self.summarizer.improve_once(
            paper=meta.get("paper"),
            section=meta.get("section"),
            current_summary=text,
            context=meta.get("context")
        )
        self.persist = Persistence(cfg, memory, container, logger)
        self.evidence = Evidence(cfg, memory, container, logger)

        self.use_arena = bool(cfg.get("use_arena", True))
        self.single_random_doc = bool(cfg.get("single_random_doc", False))

    def _build_candidates(self, section: Dict[str, Any], corpus_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # a few corpus snippets + a safe seed from the section text
        out = []
        for it in (corpus_items or [])[: int(self.cfg.get("max_corpus_candidates", 8))]:
            t = (it.get("assistant_text") or "").strip()
            if len(t) >= 60:
                out.append({"origin": "chat_corpus", "variant": f"c{it.get('id')}", "text": t, "meta": {"source": "corpus"}})
        seed = (section.get("section_text") or "").strip()[:800]
        if len(seed) >= 60:
            out.append({"origin": "lfl_seed", "variant": "seed", "text": seed, "meta": {"source": "section_text"}})
        # dedupe by normalized text
        seen = set(); uniq = []
        for c in out:
            key = " ".join(c["text"].split()).lower()
            if key in seen: continue
            seen.add(key); uniq.append(c)
        return uniq or [{"origin":"lfl_seed","variant":"seed","text":seed or "", "meta":{}}]

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.time()
        documents = context.get(self.input_key, []) or []
        if self.single_random_doc:
            doc = self.memory.documents.get_random()
            if not doc:
                return {**context, "error": "no_documents"}
            documents = [getattr(doc, "to_dict", lambda: {"id": doc.id, "title": getattr(doc, "title", "")})()]

        out: Dict[str, Any] = {}
        for paper in documents:
            casebook, goal, sections = self.persist.prepare_casebook_goal_sections(paper, context)
            self.progress.start_paper(paper, sections)

            results: List[Dict[str, Any]] = []
            for idx, section in enumerate(sections, start=1):
                if not self.persist.section_is_large_enough(section):
                    continue

                case = self.persist.create_section_case(casebook, paper, section, goal, context)
                ctx_case = {
                    **context,
                    "case_id": case.id,
                    "pipeline_run_id": context.get("pipeline_run_id"),
                    # expose strategy knobs to persistence if needed
                    "strategy_version": self.strategy.state.version,
                    "verification_threshold": self.strategy.state.verification_threshold,
                }
                self.progress.start_section(section, idx)

                # Corpus
                self.progress.stage(section, "corpus:start")
                corpus_items = await self.corpus.fetch(section["section_text"])
                self.progress.stage(section, "corpus:done", items=len(corpus_items or []))

                # Draft (arena or baseline)
                if self.use_arena:
                    cands = self._build_candidates(section, corpus_items)
                    # enrich meta for improver
                    for c in cands:
                        c.setdefault("meta", {}).update({"paper": paper, "section": section, "context": ctx_case})
                    arena_res = self.arena.run(section["section_text"], cands)
                    baseline = arena_res["winner"]["text"]
                    self.persist.persist_arena(case, paper, section, arena_res, ctx_case)
                    self.progress.stage(section, "arena:done",
                        winner_overall=round(float(arena_res["winner"]["score"].get("overall", 0.0)), 3))
                else:
                    baseline = self.summarizer.baseline(paper, section, corpus_items, ctx_case)
                    self.progress.stage(section, "baseline:done")

                # Verify & improve
                self.progress.stage(section, "verify:start")
                verify = self.summarizer.verify_and_improve(baseline, paper, section, ctx_case)
                self.progress.stage(section, "verify:done", overall=round(float(verify["metrics"].get("overall", 0.0)), 3))

                # Persist section outputs
                self.progress.stage(section, "persist:start")
                saved_case = self.persist.save_section(casebook, paper, section, verify, baseline, goal["id"], ctx_case)
                self.progress.stage(section, "persist:done")

                # Track strategy + pairs
                self.strategy.track_section(saved_case, verify["iterations"])
                self.persist.persist_pairs(saved_case.id, baseline, verify, ctx_case)

                # Section metrics -> progress
                section_metrics = {
                    "overall": verify["metrics"].get("overall", 0.0),
                    "refinement_iterations": len(verify["iterations"]),
                }
                self.progress.end_section(saved_case, section, section_metrics)

                # Periodic AB validation (optional)
                _ = self.strategy.validate_ab(context=ctx_case)

                results.append({
                    "section_name": section["section_name"],
                    "summary": verify["summary"],
                    "metrics": verify["metrics"],
                    "iterations": verify["iterations"],
                })

            # Evidence / report
            longitudinal = self.evidence.collect_longitudinal()
            cross = self.evidence.cross_episode()
            report_md = self.evidence.report(longitudinal, cross)

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
            self.logger.log("LfL_Paper_Run_Complete", paper_out)
            out.update(paper_out)

        return {**context, **out}
