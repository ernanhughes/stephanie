<!-- Merged Python Code Files -->


## File: agent.py

`python
# stephanie/agents/learning/agent.py
from __future__ import annotations
from typing import Dict, Any, List, Optional, Set
import time
import logging
import json

from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.learning.attribution import AttributionTracker
from stephanie.agents.learning.strategy_manager import StrategyManager
from stephanie.agents.learning.corpus_service import CorpusService
from stephanie.agents.learning.summarizer import Summarizer
from stephanie.agents.learning.arena import ArenaService  
from stephanie.agents.learning.persistence import Persistence
from stephanie.agents.learning.evidence import Evidence
from stephanie.agents.learning.progress import ProgressAdapter    
from stephanie.agents.learning.scoring import Scoring            
from stephanie.utils.progress import AgentProgress
import random

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
        self.attribution = AttributionTracker()  
        self.use_arena = bool(cfg.get("use_arena", True))
        self.single_random_doc = bool(cfg.get("single_random_doc", False))

    def _build_candidates(self, section: Dict[str, Any], corpus_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # a few  Think yeah it's getting snippets + a safe seed from the section text
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
                corpus_items = await self.corpus.fetch(
                    section["section_text"],
                    attribution_tracker=self.attribution,  # NEW
                )
                self.progress.stage(section, "corpus:done", items=len(corpus_items or []))

                # --- ablation (for Retrieval Necessity RN) ---
                ablate = random.random() < float(self.cfg.get("retrieval_ablate_prob", 0.0))
                if ablate:
                    self.progress.stage(section, "corpus:ablated")
                    # mark in context and keep the originals in case you want to inspect later
                    ctx_case_ablation = {"ablation": {"retrieval": "masked", "k": len(corpus_items)}}
                    self.memory.casebooks.add_scorable(
                        case_id=case.id, role="ablation", text=json.dumps(ctx_case_ablation),
                        pipeline_run_id=context.get("pipeline_run_id"), meta={"type": "retrieval_mask"}
                    )
                    corpus_items_for_baseline = []
                else:
                    corpus_items_for_baseline = corpus_items

                # attach retrieval pool to context so the summarizer can attribute claims
                ctx_case["retrieval_items"] = [
                    {"id": it.get("id"), "text": (it.get("assistant_text") or it.get("text") or "")}
                    for it in (corpus_items or [])
                ]

                # if arena is on, add initial pool as another provenance source
                arena_res = None
                if self.use_arena:
                    cands = self._build_candidates(section, corpus_items_for_baseline)
                    for c in cands:
                        c.setdefault("meta", {}).update({"paper": paper, "section": section, "context": ctx_case})
                    arena_res = self.arena.run(section["section_text"], cands)
                    ctx_case["arena_initial_pool"] = [{"origin": c.get("origin"), "variant": c.get("variant"), "text": c.get("text","")} for c in (arena_res.get("initial_pool") or [])]
                    baseline = arena_res["winner"]["text"]
                    self.persist.persist_arena(case, paper, section, arena_res, ctx_case)
                    self.progress.stage(section, "arena:done", winner_overall=round(float(arena_res["winner"]["score"].get("overall", 0.0)), 3))
                else:
                    baseline = self.summarizer.baseline(paper, section, corpus_items_for_baseline, ctx_case)
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

                if bool(self.cfg.get("run_proof", False)):
                    # Example source list: ⭐️ votes or arena_citations → supports
                    star_supports = self._collect_star_supports(case.id)  # implement to fit your memory schema
                    mask_keys = self._build_mask_keys_from_supports(star_supports)

                    with_metrics = verify["metrics"]
                    masked = await self._run_with_mask(paper, section, ctx_case, mask_keys)
                    without_metrics = masked["verify"]["metrics"]

                    delta = {
                        "overall": with_metrics["overall"] - without_metrics["overall"],
                        "grounding": with_metrics["grounding"] - without_metrics["grounding"],
                        "knowledge": with_metrics["knowledge_score"] - without_metrics["knowledge_score"],
                    }

                    try:
                        self.memory.casebooks.add_scorable(
                            case_id=case.id,
                            role="ablation_result",
                            text=json.dumps({
                                "mask": sorted(list(mask_keys)),
                                "with": with_metrics,
                                "without": without_metrics,
                                "delta": delta,
                            }),
                            pipeline_run_id=ctx_case.get("pipeline_run_id"),
                            meta={"type": "proof"},
                        )
                    except Exception:
                        pass


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

    def _build_mask_keys_from_supports(
        self, 
        supports: List[Dict[str, Any]]
    ) -> Set[str]:
        """
        supports examples:
        - {"kind":"corpus","id":"123"}
        - {"kind":"arena","origin":"chat_corpus","variant":"c123"}
        Returns a set of canonical keys understood by fetch/_build_candidates.
        """
        out: Set[str] = set()
        for s in supports or []:
            k = s.get("kind")
            if k == "corpus" and s.get("id") is not None:
                out.add(f"corpus:{str(s['id'])}")
                # also mask the mirrored arena candidate variant if it exists
                out.add(f"arena:chat_corpus#c{str(s['id'])}")
            elif k == "arena":
                out.add(f"arena:{s.get('origin','')}#{s.get('variant','')}")
        return out

    async def _run_with_mask(
        self,
        paper: Dict[str, Any],
        section: Dict[str, Any],
        ctx_case: Dict[str, Any],
        mask_keys: Set[str],
    ) -> Dict[str, Any]:
        # 1) corpus with mask
        corpus_items = await self.corpus.fetch(section["section_text"], mask_keys=mask_keys)
        # 2) candidates with mask
        cands = self._build_candidates(section, corpus_items, mask_keys=mask_keys)
        for c in cands:
            c.setdefault("meta", {}).update({"paper": paper, "section": section, "context": ctx_case})
        # 3) (arena|baseline) run
        if self.use_arena:
            arena_res = self.arena.run(section["section_text"], cands)
            baseline = arena_res["winner"]["text"]
        else:
            baseline = self.summarizer.baseline(paper, section, corpus_items, ctx_case)
        # 4) verify loop w/ current strategy
        verify = self.summarizer.verify_and_improve(baseline, paper, section, ctx_case)
        return {"baseline": baseline, "verify": verify}


    @staticmethod
    def _corpus_key(it: Dict[str, Any]) -> str:
        return f"corpus:{str(it.get('id'))}"

    @staticmethod
    def _cand_key(c: Dict[str, Any]) -> str:
        # origin+variant uniquely identify arena candidates we persist
        return f"arena:{c.get('origin','')}#{c.get('variant','')}"

    def _build_candidates(
        self,
        section: Dict[str, Any],
        corpus_items: List[Dict[str, Any]],
        *,
        mask_keys: Optional[Set[str]] = None,   # ← NEW (non-breaking)
    ) -> List[Dict[str, Any]]:
        mask_keys = mask_keys or set()

        out: List[Dict[str, Any]] = []
        # corpus-backed candidates (carry a variant tied to the corpus id)
        for it in (corpus_items or [])[: int(self.cfg.get("max_corpus_candidates", 8))]:
            t = (it.get("assistant_text") or "").strip()
            if len(t) < 60:
                continue
            corpus_k = self._corpus_key(it)
            cand_k   = f"arena:chat_corpus#c{it.get('id')}"
            if corpus_k in mask_keys or cand_k in mask_keys:
                continue  # ← respect masks
            out.append({
                "origin": "chat_corpus",
                "variant": f"c{it.get('id')}",
                "text": t,
                "meta": {"source": "corpus"},
            })

        # seed candidate from the section text
        seed = (section.get("section_text") or "").strip()[:800]
        seed_key = "arena:lfl_seed#seed"
        if len(seed) >= 60 and seed_key not in mask_keys:
            out.append({
                "origin": "lfl_seed",
                "variant": "seed",
                "text": seed,
                "meta": {"source": "section_text"},
            })

        # dedupe by normalized text (unchanged)
        seen = set(); uniq: List[Dict[str, Any]] = []
        for c in out:
            key = " ".join(c["text"].split()).lower()
            if key in seen: 
                continue
            seen.add(key); uniq.append(c)

        return uniq or ([{
            "origin":"lfl_seed","variant":"seed","text":seed or "", "meta":{}
        }] if seed_key not in mask_keys else [])


    @staticmethod
    def _corpus_key(it: Dict[str, Any]) -> str:
        return f"corpus:{str(it.get('id'))}"

    @staticmethod
    def _cand_key(c: Dict[str, Any]) -> str:
        return f"arena:{c.get('origin','')}#{c.get('variant','')}"

    def _build_candidates(
        self,
        section: Dict[str, Any],
        corpus_items: List[Dict[str, Any]],
        *,
        mask_keys: Optional[Set[str]] = None,
    ) -> List[Dict[str, Any]]:
        mask_keys = mask_keys or set()
        out: List[Dict[str, Any]] = []

        # corpus-backed candidates (variant ties to corpus id)
        for it in (corpus_items or [])[: int(self.cfg.get("max_corpus_candidates", 8))]:
            t = (it.get("assistant_text") or "").strip()
            if len(t) < 60: continue
            corpus_k = self._corpus_key(it)
            cand_k = f"arena:chat_corpus#c{it.get('id')}"
            if corpus_k in mask_keys or cand_k in mask_keys: continue
            out.append({"origin":"chat_corpus","variant":f"c{it.get('id')}","text":t,"meta":{"source":"corpus"}})

        seed = (section.get("section_text") or "").strip()[:800]
        if len(seed) >= 60 and "arena:lfl_seed#seed" not in mask_keys:
            out.append({"origin":"lfl_seed","variant":"seed","text":seed,"meta":{"source":"section_text"}})

        seen = set(); uniq: List[Dict[str, Any]] = []
        for c in out:
            key = " ".join(c["text"].split()).lower()
            if key in seen: continue
            seen.add(key); uniq.append(c)
        return uniq or [{"origin":"lfl_seed","variant":"seed","text":seed or "", "meta":{}}]

    def _build_mask_keys_from_supports(self, supports: List[Dict[str, Any]]) -> Set[str]:
        out: Set[str] = set()
        for s in supports or []:
            k = s.get("kind")
            if k == "corpus" and s.get("id") is not None:
                out.add(f"corpus:{str(s['id'])}")
                out.add(f"arena:chat_corpus#c{str(s['id'])}")
            elif k == "arena":
                out.add(f"arena:{s.get('origin','')}#{s.get('variant','')}")
        return out

    async def _run_with_mask(self, paper, section, ctx_case, mask_keys: Set[str]):
        corpus_items = await self.corpus.fetch(
            section["section_text"],
            mask_keys=mask_keys,
            attribution_tracker=self.attribution,
        )
        cands = self._build_candidates(section, corpus_items, mask_keys=mask_keys)
        for c in cands:
            c.setdefault("meta", {}).update({"paper": paper, "section": section, "context": ctx_case})

        if self.use_arena:
            arena_res = self.arena.run(section["section_text"], cands)
            baseline = arena_res["winner"]["text"]
        else:
            baseline = self.summarizer.baseline(paper, section, corpus_items, ctx_case)

        verify = self.summarizer.verify_and_improve(baseline, paper, section, ctx_case)
        return {"baseline": baseline, "verify": verify}
``n

## File: arena.py

`python
# stephanie/agents/learning/arena_service.py
from __future__ import annotations
from typing import Any, Dict, List
import logging

_logger = logging.getLogger(__name__)


class ArenaService:
    """
    Self-play tournament that takes candidates, iteratively improves them,
    and returns winner + telemetry. No DB side effects; caller persists.
    """

    def __init__(self, cfg, memory, container, logger):
        self.cfg = cfg
        self.memory = memory
        self.container = container
        self.logger = logger

    # ---- external scoring hooks provided by caller ----
    def score_candidate(
        self, text: str, section_text: str
    ) -> Dict[str, float]:
        raise NotImplementedError  # caller sets via injectors or subclass

    def improve(self, cand_text: str, improve_context: Dict[str, Any]) -> str:
        raise NotImplementedError  # caller sets via injectors or subclass

    # ---- main API ----
    def run(
        self, section_text: str, initial_candidates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        # score pool
        scored = []
        for c in initial_candidates:
            s = self.score_candidate(c["text"], section_text)
            scored.append({**c, "score": s})
        scored.sort(
            key=lambda x: (
                x.get("score", {}).get("verified", False),
                x.get("score", {}).get("overall", 0.0),
            ),
            reverse=True,
        )

        beam_w = int(self.cfg.get("beam_width", 5))
        max_rounds = int(self.cfg.get("self_play_rounds", 2))
        plateau_eps = float(self.cfg.get("self_play_plateau_eps", 0.005))
        min_marg = float(self.cfg.get("min_marginal_reward_per_ktok", 0.05))

        beam = scored[:beam_w]
        iters: List[List[Dict[str, Any]]] = []

        def _est_tokens(t: str) -> int:
            return max(1, int(len(t or "") / 4))

        def _marginal(
            prev_best: float, curr_best: float, prev_toks: int, curr_toks: int
        ) -> float:
            dr, dt = (curr_best - prev_best), max(1, curr_toks - prev_toks)
            return (dr / dt) * 1000.0

        best_hist: List[float] = []
        prev_best = beam[0]["score"]["overall"] if beam else 0.0
        prev_toks = _est_tokens(beam[0]["text"]) if beam else 1

        for r in range(max_rounds):
            new_beam = []
            for cand in beam:
                improved = self.improve(
                    cand["text"], {**cand.get("meta", {}), "round": r}
                )
                s = self.score_candidate(improved, section_text)
                new_beam.append(
                    {
                        **cand,
                        "variant": f"{cand.get('variant', 'v')}+r{r + 1}",
                        "text": improved,
                        "score": s,
                    }
                )

            new_beam.sort(
                key=lambda x: (
                    x["score"].get("verified", False),
                    x["score"]["overall"],
                ),
                reverse=True,
            )

            # Diversity guard (optional)
            origins = {b.get("origin") for b in new_beam}
            if len(origins) == 1:
                alt = next(
                    (c for c in scored if c.get("origin") not in origins), None
                )
                if alt:
                    new_beam[-1] = alt

            curr_best = (
                new_beam[0]["score"]["overall"] if new_beam else prev_best
            )
            curr_toks = (
                _est_tokens(new_beam[0]["text"]) if new_beam else prev_toks
            )
            marginal = _marginal(prev_best, curr_best, prev_toks, curr_toks)

            iters.append(
                [
                    {
                        "variant": b["variant"],
                        "overall": b["score"]["overall"],
                        "k": b["score"].get("k", 0.0),
                    }
                    for b in new_beam
                ]
            )

            if marginal < min_marg:
                break
            best_hist.append(curr_best)
            if (
                len(best_hist) >= 2
                and (best_hist[-1] - best_hist[-2]) < plateau_eps
            ):
                break

            beam, prev_best, prev_toks = (
                new_beam[:beam_w],
                curr_best,
                curr_toks,
            )

        winner = (beam or scored or [{"text": "", "score": {}}])[0]
        return {
            "winner": winner,
            "beam": beam,
            "initial_pool": scored,
            "iterations": iters,
        }
``n

## File: attribution.py

`python
# stephanie/agents/learning/attribution.py
from __future__ import annotations
from typing import Dict, Any, List
import time


class AttributionTracker:
    """Tracks knowledge contributions and their ablation impact."""

    def __init__(self):
        self.contributions: Dict[str, Dict[str, Any]] = {}
        self.ablation_results: List[Dict[str, Any]] = []

    def record_contribution(self, key: str, data: Dict[str, Any]):
        self.contributions.setdefault(
            key,
            {"key": key, "data": data, "used_in": [], "ablation_impact": None},
        )

    def mark_used(self, key: str, section_id: str, metrics: Dict[str, float]):
        c = self.contributions.get(key)
        if not c:
            return
        c["used_in"].append(
            {"section_id": section_id, "metrics": metrics, "ts": time.time()}
        )

    def record_ablation(
        self,
        key: str,
        with_metrics: Dict[str, float],
        without_metrics: Dict[str, float],
    ):
        c = self.contributions.get(key)
        if not c:
            return
        delta = {
            "overall": with_metrics["overall"] - without_metrics["overall"],
            "knowledge_score": with_metrics["knowledge_score"]
            - without_metrics["knowledge_score"],
            "grounding": with_metrics["grounding"]
            - without_metrics["grounding"],
        }
        impact = {
            "with": with_metrics,
            "without": without_metrics,
            "delta": delta,
            "ts": time.time(),
        }
        c["ablation_impact"] = impact
        self.ablation_results.append(
            {"key": key, "contribution": c, "delta": delta}
        )

    def get_significant_contributions(
        self, min_impact: float = 0.03
    ) -> List[Dict[str, Any]]:
        return [
            r
            for r in self.ablation_results
            if r["delta"]["overall"] >= min_impact
        ]

    def evidence_md(self) -> str:
        sig = self.get_significant_contributions()
        if not sig:
            return "**No significant ablation impacts yet.** Try enabling ablations on high-impact supports."
        avg = sum(r["delta"]["overall"] for r in sig) / max(1, len(sig))
        lines = [
            "## 🔬 Ablation Evidence (Applied Knowledge)",
            f"- Significant contributions: **{len(sig)}**",
            f"- Avg Δ verification score (with − without): **{avg:+.3f}**",
            "",
        ]
        for i, r in enumerate(
            sorted(sig, key=lambda x: x["delta"]["overall"], reverse=True)[:3],
            1,
        ):
            d = r["delta"]
            data = r["contribution"]["data"]
            lines += [
                f"### Example #{i}",
                f"- Source: `{data.get('source')}` (id={data.get('id')})",
                f"- Context: {data.get('retrieval_context', '')}",
                f"- Excerpt: “{(data.get('section_text') or '')[:180]}…”",
                f"- Impact: overall {d['overall']:+.3f}, knowledge {d['knowledge_score']:+.3f}, grounding {d['grounding']:+.3f}",
                "",
            ]
        return "\n".join(lines)
``n

## File: corpus_service.py

`python
# stephanie/agents/learning/corpus_service.py
from typing import List, Dict, Any, Optional, Set
from .attribution import AttributionTracker  # NEW
import logging
_logger = logging.getLogger(__name__)

class CorpusService:
    @staticmethod
    def _corpus_key(it: Dict[str, Any]) -> str:
        return f"corpus:{str(it.get('id'))}"

    async def fetch(
        self,
        section_text: str,
        *,
        mask_keys: Optional[Set[str]] = None,
        allow_keys: Optional[Set[str]] = None,
        attribution_tracker: Optional[AttributionTracker] = None,  # NEW
    ) -> List[Dict[str, Any]]:
        try:
            res = self.chat_corpus(
                section_text,
                k=self.cfg.get("chat_corpus_k", 60),
                weights={"semantic": 0.6, "entity": 0.25, "domain": 0.15},
                include_text=True,
            )
            items = res.get("items", []) or []

            # NEW: allowlist/mask
            if allow_keys is not None:
                items = [it for it in items if self._corpus_key(it) in allow_keys]
            if mask_keys:
                mk = set(mask_keys)
                items = [it for it in items if self._corpus_key(it) not in mk]

            # NEW: attribution breadcrumbs
            if attribution_tracker:
                for it in items:
                    k = self._corpus_key(it)
                    it["attribution_id"] = k
                    attribution_tracker.record_contribution(k, {
                        "source": "corpus",
                        "id": it.get("id"),
                        "score": float((it.get("score") or 0.0)),
                        "section_text": section_text[:240],
                        "retrieval_context": "section processing",
                    })

            # (annotate/analyze) unchanged
            try:
                if self.annotate: await self.annotate.run(context={"scorables": items})
                if self.analyze:  await self.analyze.run(context={"chats": items})
            except Exception as e:
                _logger.warning(f"Corpus annotate/analyze skipped: {e}")
            return items
        except Exception as e:
            _logger.warning(f"Chat corpus retrieval failed: {e}")
            return []
``n

## File: evidence.py

`python
# stephanie/agents/learning/evidence.py
from __future__ import annotations
from typing import Dict, Any, Optional, Tuple
import json
import time

class Evidence:
    def __init__(self, cfg, memory, container, logger):
        self.cfg, self.memory, self.container, self.logger = (
            cfg, memory, container, logger
        )
        self.casebook_tag = cfg.get("casebook_action", "blog")
        self._last: Dict[str, Any] = {}   # NEW: last snapshot for delta reporting

    # -------------- small reporting bus (NEW) --------------
    def _emit(self, event: str, **fields):
        payload = {"event": event, **fields}
        # try container.report
        try:
            rep = getattr(self.container, "report", None)
            if callable(rep):
                rep(payload)
                return
        except Exception:
            pass
        # try a dedicated reporter service
        try:
            get_service = getattr(self.container, "get_service", None)
            if callable(get_service):
                reporter = get_service("report")
                if reporter and callable(getattr(reporter, "report", None)):
                    reporter.report(payload)
                    return
        except Exception:
            pass
        # fallback to logger
        try:
            self.logger.log(event, payload)
        except Exception:
            pass

    # -------------- util --------------
    @staticmethod
    def _percent_change(start: float, end: float) -> float:
        try:
            if start is None or end is None or start == 0:
                return 0.0
            return (end - start) / abs(start) * 100.0
        except Exception:
            return 0.0

    # -------------- longitudinal (unchanged logic; added emit) --------------
    def collect_longitudinal(self) -> Dict[str, Any]:
        out = {
            "total_papers": 0,
            "verification_scores": [],
            "iteration_counts": [],
            "avg_verification_score": 0.0,
            "avg_iterations": 0.0,
            "score_improvement_pct": 0.0,
            "iteration_reduction_pct": 0.0,
            "strategy_versions": [],
            "strategy_evolution_rate": 0.0,
        }
        try:
            casebooks = self.memory.casebooks.get_casebooks_by_tag(self.casebook_tag) or []
            for cb in casebooks:
                cases = self.memory.casebooks.get_cases_for_casebook(cb.id) or []
                for case in cases:
                    for s in self.memory.casebooks.list_scorables(case.id) or []:
                        if s.role != "metrics":
                            continue
                        try:
                            payload = json.loads(s.text) if isinstance(s.text, str) else (s.text or {})
                            final_scores = (payload or {}).get("final_scores") or {}
                            overall = final_scores.get("overall")
                            iters = payload.get("refinement_iterations")
                            if overall is not None:
                                out["verification_scores"].append(float(overall))
                            if iters is not None:
                                out["iteration_counts"].append(int(iters))
                        except Exception:
                            continue

            vs, it = out["verification_scores"], out["iteration_counts"]
            out["total_papers"] = len(vs)
            if vs:
                out["avg_verification_score"] = sum(vs) / len(vs)
            if it:
                out["avg_iterations"] = sum(it) / len(it)
            if len(vs) >= 2:
                out["score_improvement_pct"] = self._percent_change(vs[0], vs[-1])
            if len(it) >= 2:
                out["iteration_reduction_pct"] = -1.0 * self._percent_change(it[0], it[-1])

            # EMIT longitudinal snapshot + deltas (NEW)
            self._emit("evidence.longitudinal",
                       at=time.time(),
                       total_papers=out["total_papers"],
                       avg_score=out["avg_verification_score"],
                       avg_iters=out["avg_iterations"],
                       score_trend_pct=out["score_improvement_pct"],
                       iter_trend_pct=out["iteration_reduction_pct"],
                       delta=self._delta_section("longitudinal", out))
        except Exception as e:
            try:
                self.logger.log("LfL_Longitudinal_Failed", {"err": str(e)})
            except Exception:
                pass
        return out

    # -------------- helpers used below (NEW) --------------
    def _delta_section(self, key: str, current: Dict[str, Any]) -> Dict[str, Tuple[Any, Any]]:
        """Return {field: (prev, curr)} for changed fields and store snapshot."""
        prev = self._last.get(key, {})
        changed = {}
        for k, v in current.items():
            if isinstance(v, (int, float, str)) and prev.get(k) != v:
                changed[k] = (prev.get(k), v)
        self._last[key] = {**prev, **{k: current[k] for k in current}}
        return changed

    # -------------- cross-episode (added AR/AKL/RN/TR + emit) --------------
    def _get_paper_performance(self, casebook) -> Dict[str, float]:
        scores, iters = [], []
        for case in (self.memory.casebooks.get_cases_for_casebook(casebook.id) or []):
            for s in self.memory.casebooks.list_scorables(case.id) or []:
                if s.role == "metrics":
                    try:
                        rec = json.loads(s.text) if isinstance(s.text, str) else (s.text or {})
                        final = (rec or {}).get("final_scores") or {}
                        if "overall" in final: scores.append(float(final["overall"]))
                        itv = rec.get("refinement_iterations")
                        if itv is not None: iters.append(int(itv))
                    except Exception:
                        pass
        return {
            "mean_overall": (sum(scores)/len(scores)) if scores else 0.0,
            "mean_iters": (sum(iters)/len(iters)) if iters else 0.0,
            "n": len(scores),
        }

    def _extract_winner_origins(self, casebook) -> Dict[str, int]:
        tally = {}
        for case in (self.memory.casebooks.get_cases_for_casebook(casebook.id) or []):
            for s in self.memory.casebooks.list_scorables(case.id) or []:
                if s.role == "arena_winner":
                    origin = (s.meta or {}).get("origin")
                    if origin: tally[origin] = tally.get(origin, 0) + 1
                    break
        return tally

    def _calculate_knowledge_transfer(self) -> Dict[str, Any]:
        cbs = self.memory.casebooks.get_casebooks_by_tag(self.casebook_tag) or []
        if len(cbs) < 2:
            return {"rate": 0.0, "examples": []}
        transfer, examples = 0, []
        for i in range(1, len(cbs)):
            prev_cb, curr_cb = cbs[i-1], cbs[i]
            prev = self._extract_winner_origins(prev_cb)
            curr = self._extract_winner_origins(curr_cb)
            reused = [k for k in prev.keys() if curr.get(k, 0) > 0]
            if reused:
                transfer += 1
                perf = self._get_paper_performance(curr_cb)
                examples.append({
                    "from_paper": getattr(prev_cb, "name", str(prev_cb.id)),
                    "to_paper": getattr(curr_cb, "name", str(curr_cb.id)),
                    "patterns_used": [{"name": k, "description": f"Winner origin reused: {k}"} for k in reused],
                    "performance_impact": perf["mean_overall"],
                })
        return {"rate": transfer / max(1, len(cbs)-1), "examples": examples}

    def _calculate_domain_learning(self) -> Dict[str, Any]:
        cbs = self.memory.casebooks.get_casebooks_by_tag(self.casebook_tag) or []
        vals = []
        for cb in cbs:
            perf = self._get_paper_performance(cb)
            vals.append(perf["mean_overall"])
        return {"all_mean": (sum(vals)/len(vals)) if vals else 0.0, "samples": len(vals)}

    def _calculate_meta_patterns(self) -> Dict[str, Any]:
        cbs = self.memory.casebooks.get_casebooks_by_tag(self.casebook_tag) or []
        rounds = 0; sections = 0
        for cb in cbs:
            for case in self.memory.casebooks.get_cases_for_casebook(cb.id) or []:
                sections += 1
                for s in self.memory.casebooks.list_scorables(case.id) or []:
                    if s.role == "arena_round_metrics":
                        try:
                            j = json.loads(s.text) if isinstance(s.text, str) else (s.text or {})
                            rounds += len((j or {}).get("rounds", []))
                        except Exception:
                            pass
        return {"avg_rounds": (rounds/sections) if sections else 0.0, "sections": sections}

    def _calculate_adaptation_rate(self) -> float:
        return 0.0

    def _collect_improve_attributions(self):
        total_sections, supported_sections = 0, 0
        lifts = []
        ablation_pairs = []  # (score_normal, score_ablated)
        casebooks = self.memory.casebooks.get_casebooks_by_tag(self.casebook_tag) or []
        for cb in casebooks:
            for case in (self.memory.casebooks.get_cases_for_casebook(cb.id) or []):
                total_sections += 1
                has_supported = False
                final_overall = None
                ablated = False
                for s in self.memory.casebooks.list_scorables(case.id) or []:
                    if s.role == "metrics":
                        try:
                            rec = json.loads(s.text) if isinstance(s.text, str) else (s.text or {})
                            final_overall = ((rec or {}).get("final_scores") or {}).get("overall")
                            # accept either path: directly in metrics, or nested in final_scores (your patch)
                            k_lift = (rec or {}).get("knowledge_applied_lift", None)
                            if k_lift is None:
                                k_lift = ((rec or {}).get("final_scores") or {}).get("knowledge_applied_lift", 0.0)
                            if k_lift:
                                lifts.append(float(k_lift))
                        except Exception:
                            pass
                    elif s.role == "improve_attribution":
                        has_supported = True
                    elif s.role == "ablation":
                        ablated = True
                if has_supported:
                    supported_sections += 1
                # naive pairing: compare ablated case to any non-ablated peer in same casebook
                if ablated and final_overall is not None:
                    peer = None
                    for c2 in (self.memory.casebooks.get_cases_for_casebook(cb.id) or []):
                        if c2.id == case.id:
                            continue
                        found_ablate = False
                        peer_final = None
                        for s2 in (self.memory.casebooks.list_scorables(c2.id) or []):
                            if s2.role == "ablation":
                                found_ablate = True
                            elif s2.role == "metrics":
                                try:
                                    r2 = json.loads(s2.text) if isinstance(s2.text, str) else (s2.text or {})
                                    peer_final = ((r2 or {}).get("final_scores") or {}).get("overall")
                                except Exception:
                                    pass
                        if not found_ablate and peer_final is not None:
                            peer = peer_final
                            break
                    if peer is not None:
                        ablation_pairs.append((peer, final_overall))
        AR = supported_sections / max(1, total_sections)
        AKL = (sum(lifts) / len(lifts)) if lifts else 0.0
        RN = None
        if ablation_pairs:
            diffs = [a - b for (a, b) in ablation_pairs]
            RN = sum(diffs) / len(diffs)
        return {
            "attribution_rate": AR,
            "applied_knowledge_lift": AKL,
            "retrieval_ablation_delta": RN,
        }

    def _strict_transfer_rate(self):
        cbs = self.memory.casebooks.get_casebooks_by_tag(self.casebook_tag) or []
        if len(cbs) < 2:
            return 0.0
        reused = 0; denom = 0
        prev_citations = set()
        for i, cb in enumerate(cbs):
            cites_here = set()
            for case in (self.memory.casebooks.get_cases_for_casebook(cb.id) or []):
                for s in self.memory.casebooks.list_scorables(case.id) or []:
                    if s.role == "arena_citations":
                        try:
                            j = json.loads(s.text) if isinstance(s.text, str) else (s.text or {})
                            for c in (j or {}).get("citations", []):
                                key = (c.get("support_origin"), c.get("support_variant"))
                                if key[0] or key[1]:
                                    cites_here.add(key)
                        except Exception:
                            pass
            if i > 0:
                denom += 1
                if prev_citations & cites_here:
                    reused += 1
            prev_citations = cites_here
        return reused / max(1, denom)

    def cross_episode(self) -> Dict[str, Any]:
        kt = self._calculate_knowledge_transfer()
        dom = self._calculate_domain_learning()
        meta = self._calculate_meta_patterns()
        adapt_rate = self._calculate_adaptation_rate()
        ak = self._collect_improve_attributions()
        strict_tr = self._strict_transfer_rate()
        out = {
            "knowledge_transfer_rate": kt["rate"],
            "knowledge_transfer_examples": kt["examples"][:3],
            "domain_learning_patterns": dom,
            "meta_pattern_recognition": meta,
            "strategy_adaptation_rate": adapt_rate,
            "cross_episode_evidence_strength": self._calculate_evidence_strength(kt, dom, meta, adapt_rate),
            "attribution_rate": ak["attribution_rate"],
            "applied_knowledge_lift": ak["applied_knowledge_lift"],
            "retrieval_ablation_delta": ak["retrieval_ablation_delta"],
            "strict_transfer_rate": strict_tr,
        }
        # EMIT cross-episode snapshot + deltas (NEW)
        self._emit("evidence.cross_episode",
                   at=time.time(),
                   knowledge_transfer_rate=out["knowledge_transfer_rate"],
                   attribution_rate=out["attribution_rate"],
                   applied_knowledge_lift=out["applied_knowledge_lift"],
                   retrieval_ablation_delta=out["retrieval_ablation_delta"],
                   strict_transfer_rate=out["strict_transfer_rate"],
                   evidence_strength=out["cross_episode_evidence_strength"],
                   delta=self._delta_section("cross_episode", out))
        # one-line headline (NEW)
        self._emit("evidence.summary",
                   at=time.time(),
                   msg=("AR={:.0%} | AKL={:+.3f} | RNΔ={}".format(
                        out["attribution_rate"],
                        out["applied_knowledge_lift"],
                        "n/a" if out["retrieval_ablation_delta"] is None else f"{out['retrieval_ablation_delta']:+.3f}"
                   )))
        return out

    def _calculate_evidence_strength(self, kt: Dict[str, Any], dom: Dict[str, Any], meta: Dict[str, Any], adapt_rate: float) -> float:
        return max(0.0, min(1.0, 0.35*kt.get("rate",0.0) + 0.25*dom.get("all_mean",0.0) + 0.20*(meta.get("avg_rounds",0.0)/5.0) + 0.20*adapt_rate))

    def report(self, longitudinal: Dict[str, Any], cross: Dict[str, Any]) -> str:
        if not longitudinal or longitudinal.get("total_papers", 0) < 3:
            return ""
        score_trend = longitudinal.get("score_improvement_pct", 0.0)
        iter_trend  = longitudinal.get("iteration_reduction_pct", 0.0)
        arrow_score = "↑" if score_trend > 0 else "↓"
        arrow_iter  = "↓" if iter_trend > 0 else "↑"

        lines = []
        lines.append("## Learning from Learning: Evidence Report")
        lines.append("")
        lines.append(f"- **Total papers processed**: {longitudinal.get('total_papers', 0)}")
        lines.append(f"- **Verification score trend**: {score_trend:.1f}% {arrow_score}")
        lines.append(f"- **Average iterations trend**: {iter_trend:.1f}% {arrow_iter}")
        lines.append(f"- **Knowledge transfer rate**: {cross.get('knowledge_transfer_rate', 0.0):.0%}")
        if cross.get("strategy_ab_validation"):
            ab = cross["strategy_ab_validation"]
            lines.append(f"- **A/B delta (B−A)**: {ab.get('delta_B_minus_A', 0.0):+.3f} (A n={ab.get('samples_A',0)}, B n={ab.get('samples_B',0)})")
        lines.append(f"- **Cross-episode evidence strength**: {cross.get('cross_episode_evidence_strength', 0.0):.0%}")
        lines.append("")
        vs = longitudinal.get("verification_scores", [])
        it = longitudinal.get("iteration_counts", [])
        if vs and it:
            lines.append("### Snapshot")
            lines.append(f"- First: score={vs[0]:.2f}, iterations={it[0]}")
            lines.append(f"- Latest: score={vs[-1]:.2f}, iterations={it[-1]}")
        if cross.get("knowledge_transfer_examples"):
            ex = cross["knowledge_transfer_examples"][0]
            lines.append("")
            lines.append("### Cross-Episode Example")
            lines.append(f"- *{ex['from_paper']}* → *{ex['to_paper']}* reused patterns:")
            for p in ex.get("patterns_used", [])[:3]:
                lines.append(f" • {p['name']} – {p['description']}")
            lines.append(f"- Mean overall after transfer: {ex.get('performance_impact', 0.0):.3f}")

        lines.append(f"- **Attribution rate**: {cross.get('attribution_rate', 0.0):.0%}")
        lines.append(f"- **Applied-knowledge lift**: {cross.get('applied_knowledge_lift', 0.0):+.3f}")
        if cross.get("retrieval_ablation_delta") is not None:
            lines.append(f"- **Retrieval necessity (ablation Δ)**: {cross['retrieval_ablation_delta']:+.3f}")
        lines.append(f"- **Strict transfer rate**: {cross.get('strict_transfer_rate', 0.0):.0%}")

        md = "\n".join(lines)

        # also emit the final markdown block once per call (NEW)
        self._emit("evidence.report_md", markdown=md, at=time.time())
        return md
``n

## File: persistence.py

`python
# stephanie/agents/learning/persistence.py
from __future__ import annotations
from typing import Dict, Any, List
import time
import json
from stephanie.models.casebook import CaseBookORM, CaseORM
from stephanie.scoring.scorable import ScorableType
from stephanie.utils.casebook_utils import generate_casebook_name
from stephanie.utils.json_sanitize import dumps_safe
from stephanie.utils.paper_utils import build_paper_goal_text, build_paper_goal_meta
import logging

_logger = logging.getLogger(__name__)

AGENT_NAME = "LearningFromLearningAgent"  # stable label

class Persistence:
    def __init__(self, cfg, memory, container, logger):
        self.cfg = cfg
        self.memory = memory
        self.container = container
        self.logger = logger
        self.casebook_action = cfg.get("casebook_action","blog")
        self.min_section_length = int(cfg.get("min_section_length", 100))

    def prepare_casebook_goal_sections(self, paper: Dict[str, Any], context: Dict[str, Any]):
        title = paper.get("title",""); doc_id = paper.get("id") or paper.get("doc_id")
        name = generate_casebook_name(self.casebook_action, title)
        cb = self.memory.casebooks.ensure_casebook(name=name, description=f"LfL agent runs for paper {title}", tag=self.casebook_action)
        goal = self.memory.goals.get_or_create({
            "goal_text": build_paper_goal_text(title),
            "description": "LfL: verify & improve per section",
            "meta": build_paper_goal_meta(title, doc_id, domains=self.cfg.get("domains", [])),
        }).to_dict()
        sections = self._resolve_sections(paper)
        return cb, goal, sections

    def _resolve_sections(self, paper: Dict[str, Any]) -> List[Dict[str, Any]]:
        doc_id = paper.get("id") or paper.get("doc_id")
        sections = self.memory.document_sections.get_by_document(doc_id) or []
        if sections:
            return [{
                "section_name": s.section_name, "section_text": s.section_text or "",
                "section_id": s.id, "order_index": getattr(s, "order_index", None),
                "attributes": {"paper_id": str(doc_id), "section_name": s.section_name, "section_index": getattr(s,"order_index",0), "case_kind": "summary"},
            } for s in sections]
        return [{
            "section_name": "Abstract",
            "section_text": f"{paper.get('title','').strip()}\n\n{paper.get('abstract','').strip()}",
            "section_id": None, "order_index": 0,
            "attributes": {"paper_id": str(doc_id), "section_name": "Abstract", "section_index": 0, "case_kind": "summary"},
        }]

    def section_is_large_enough(self, section: Dict[str, Any]) -> bool:
        return bool((section.get("section_text") or "").strip()) and len(section.get("section_text","")) >= self.min_section_length

    def create_section_case(self, casebook: CaseBookORM, paper: Dict[str, Any], section: Dict[str, Any], goal: Dict[str, Any], context: Dict[str, Any]) -> CaseORM:
        case = self.memory.casebooks.add_case(
            casebook_id=casebook.id, goal_id=goal["id"], prompt_text=f"Section: {section['section_name']}",
            agent_name=AGENT_NAME, 
            meta={"type":"section_case"},
        )
        self.memory.casebooks.set_case_attr(case.id, "paper_id", value_text=str(paper.get("id") or paper.get("doc_id")))
        self.memory.casebooks.set_case_attr(case.id, "section_name", value_text=str(section["section_name"]))
        if section.get("section_id") is not None:
            self.memory.casebooks.set_case_attr(case.id, "section_id", value_text=str(section["section_id"]))
        if section.get("order_index") is not None:
            self.memory.casebooks.set_case_attr(case.id, "section_index", value_num=float(section.get("order_index") or 0))
        self.memory.casebooks.set_case_attr(case.id, "case_kind", value_text="summary")
        self.memory.casebooks.set_case_attr(case.id, "scorable_id", value_text=str(section.get("section_id") or ""))
        self.memory.casebooks.set_case_attr(case.id, "scorable_type", value_text=str(ScorableType.DOCUMENT_SECTION))


        return case

    def save_section(self, casebook, paper, section, verify, baseline, goal_id, context) -> CaseORM:
        threshold = float(context.get("verification_threshold", 0.85))
        return self._save_section_to_casebook(
            casebook=casebook, goal_id=goal_id,
            doc_id=str(paper.get("id") or paper.get("doc_id")),
            section_name=section["section_name"], section_text=section["section_text"],
            result={
                "initial_draft":{"title": section["section_name"], "body": baseline},
                "refined_draft":{"title": section["section_name"], "body": verify["summary"]},
                "verification_report":{"scores": verify["metrics"], "iterations": verify["iterations"]},
                "final_validation":{"scores": verify["metrics"], "passed": verify["metrics"]["overall"] >= threshold},
                "passed": verify["metrics"]["overall"] >= threshold,
                "refinement_iterations": len(verify["iterations"])
            },
            context={
                **context, "paper_title": paper.get("title",""),
                "paper_id": paper.get("id") or paper.get("doc_id"),
                "section_order_index": section.get("order_index")
            }
        )

    
    def _save_section_to_casebook(
        self,
        casebook: CaseBookORM,
        goal_id: int,
        doc_id: str,
        section_name: str,
        section_text: str,
        result: Dict[str, Any],
        context: Dict[str, Any],
    ) -> CaseORM:
        paper_title = context.get("paper_title")
        order_index = context.get("section_order_index")
        pipeline_run_id = context.get("pipeline_run_id")

        case_prompt = {
            "paper_id": doc_id,
            "paper_title": paper_title,
            "section_name": section_name,
            "section_order_index": order_index,
        }
        case_meta = {
            "type": "draft_trajectory",
            "paper_id": doc_id,
            "paper_title": paper_title,
            "section_name": section_name,
            "section_order_index": order_index,
            "timestamp": time.time(),
            "source": "lfl.agent",
        }

        case = self.memory.casebooks.add_case(
            casebook_id=casebook.id,
            goal_id=goal_id,
            prompt_text=dumps_safe(case_prompt),
            agent_name=AGENT_NAME,
            meta=case_meta,
        )

        def _smeta(extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
            base = {
                "paper_id": doc_id,
                "paper_title": paper_title,
                "section_name": section_name,
                "section_order_index": order_index,
            }
            if extra:
                base.update(extra)
            return base

        self.memory.casebooks.add_scorable(
            case_id=case.id,
            pipeline_run_id=pipeline_run_id,
            text=section_text,
            role="section_text",
            meta=_smeta(),
        )
        self.memory.casebooks.add_scorable(
            case_id=case.id,
            pipeline_run_id=pipeline_run_id,
            text=dumps_safe(result.get("initial_draft", {})),
            role="initial_draft",
            meta=_smeta(),
        )
        self.memory.casebooks.add_scorable(
            case_id=case.id,
            pipeline_run_id=pipeline_run_id,
            text=dumps_safe(result.get("refined_draft", {})),
            role="refined_draft",
            meta=_smeta(
                {
                    "refinement_iterations": result.get(
                        "refinement_iterations", 0
                    )
                }
            ),
        )
        self.memory.casebooks.add_scorable(
            case_id=case.id,
            pipeline_run_id=pipeline_run_id,
            text=dumps_safe(result.get("verification_report", {})),
            role="verification_report",
            meta=_smeta(),
        )
        self.memory.casebooks.add_scorable(
            case_id=case.id,
            pipeline_run_id=pipeline_run_id,
            text=dumps_safe(result.get("final_validation", {})),
            role="final_validation",
            meta=_smeta(),
        )
        final_scores = (result.get("final_validation", {}) or {}).get("scores", {}) or {}
        metrics_payload = {
            "passed": result.get("passed", False),
            "refinement_iterations": result.get("refinement_iterations", 0),
            "final_scores": final_scores,
            "knowledge_applied_iters": final_scores.get("knowledge_applied_iters", 0),
            "knowledge_applied_lift": final_scores.get("knowledge_applied_lift", 0.0),
            "ablation": context.get("ablation"),
        }

        self.memory.casebooks.add_scorable(
            case_id=case.id,
            pipeline_run_id=pipeline_run_id,
            text=dumps_safe(metrics_payload),
            role="metrics",
            meta=_smeta(),
        )
        try:
            self.memory.casebooks.set_case_attr(
                case.id, "knowledge_applied_iters",
                value_num=float(metrics_payload.get("knowledge_applied_iters", 0))
            )
            self.memory.casebooks.set_case_attr(
                case.id, "knowledge_applied_lift",
                value_num=float(metrics_payload.get("knowledge_applied_lift", 0.0))
            )
        except Exception:
            pass


        return case
    
    def persist_pairs(self, case_id: int, baseline: str, verify: Dict[str, Any], context: Dict[str, Any]):
        metrics = verify["metrics"]
        improved = verify["summary"]
        try:
            self.memory.casebooks.add_scorable(
                case_id=case_id, role="knowledge_pair_positive", text=improved,
                pipeline_run_id=context.get("pipeline_run_id"),
                meta={"verification_score": metrics.get("overall",0.0),
                      "knowledge_score": metrics.get("knowledge_score",0.0),
                      "strategy_version": context.get("strategy_version")},
            )
            if metrics.get("overall",0.0) >= 0.85:
                self.memory.casebooks.add_scorable(
                    case_id=case_id, role="knowledge_pair_negative", text=baseline,
                    pipeline_run_id=context.get("pipeline_run_id"),
                    meta={"verification_score": max(0.0, metrics.get("overall",0.0)-0.15),
                          "knowledge_score": max(0.0, metrics.get("knowledge_score",0.0)*0.7),
                          "strategy_version": context.get("strategy_version")},
                )
        except Exception:
            pass

    def persist_arena(self, case, paper, section, arena, context):
        """
        Persist arena artifacts for audit & reuse, plus compact telemetry.

        Writes:
        - arena_candidate (initial_pool, capped)
        - arena_beam      (final beam, capped)
        - arena_winner
        - arena_round_metrics (JSON: per-round summary w/o large texts)

        And attributes on the case:
        - arena_beam_width (num)
        - arena_rounds     (num)
        - arena_winner_origin (text)
        - arena_winner_overall/k/c/g (num each)
        - arena_winner_sidequest_id (text, if present)
        """
        def _meta(base=None, **kw):
            m = {
                "paper_id": str(paper.get("id") or paper.get("doc_id")),
                "section_name": section.get("section_name"),
                "type": "arena",
            }
            if base:
                m.update(base)
            m.update(kw)
            return m

        try:
            pipeline_run_id = context.get("pipeline_run_id")
            pool_cap = int(self.cfg.get("persist_pool_cap", 30))
            beam_cap = int(self.cfg.get("persist_beam_cap", 10))

            # -------- Initial pool (scored candidates) --------
            for c in (arena.get("initial_pool") or [])[:pool_cap]:
                self.memory.casebooks.add_scorable(
                    case_id=case.id,
                    role="arena_candidate",
                    pipeline_run_id=pipeline_run_id,
                    text=c.get("text", "") or "",
                    meta=_meta(
                        origin=c.get("origin"),
                        variant=c.get("variant"),
                        score=c.get("score", {}),
                        source=(c.get("meta") or {}).get("source"),
                    ),
                )

            # -------- Final beam (last round’s top-K) --------
            for b in (arena.get("beam") or [])[:beam_cap]:
                self.memory.casebooks.add_scorable(
                    case_id=case.id,
                    role="arena_beam",
                    pipeline_run_id=pipeline_run_id,
                    text=b.get("text", "") or "",
                    meta=_meta(
                        origin=b.get("origin"),
                        variant=b.get("variant"),
                        score=b.get("score", {}),
                    ),
                )

            # -------- Winner --------
            w = arena.get("winner") or {}

            # Auto-citation (uses embedding service if available)
            try:
                winner_txt = w["text"] or ""
                claims = self._extract_claim_sentences(winner_txt)
                if claims:
                    emb = self.memory.embedding
                    support_pool = (arena.get("initial_pool") or [])[:20]  # cap
                    citations = []
                    if emb:
                        cvecs = [(c, emb.get_or_create((c["text"] or "")[:2000])) for c in support_pool]
                        for s in claims:
                            svec = emb.get_or_create(s)
                            # best supporting source
                            best = None
                            best_sim = 0.0
                            for cand, v in cvecs:
                                sim = self._cos_sim(svec, v)
                                if sim > best_sim:
                                    best_sim, best = sim, cand
                            citations.append({
                                "claim": s,
                                "support_origin": best.get("origin") if best else None,
                                "support_variant": best.get("variant") if best else None,
                                "similarity": round(best_sim, 3),
                            })
                    if citations:
                        self.memory.casebooks.add_scorable(
                            case_id=case.id,
                            role="arena_citations",
                            text=dumps_safe({"citations": citations}),
                            pipeline_run_id=context.get("pipeline_run_id"),
                            meta=_meta(origin=w["origin"], variant=w["variant"])
                        )
            except Exception as e:
                _logger.warning(f"auto-citation skipped: {e}")

            w_score = w.get("score") or {}
            self.memory.casebooks.add_scorable(
                case_id=case.id,
                role="arena_winner",
                pipeline_run_id=pipeline_run_id,
                text=w.get("text", "") or "",
                meta=_meta(
                    origin=w.get("origin"),
                    variant=w.get("variant"),
                    score=w_score,
                    sidequest_id=(w.get("meta") or {}).get("sidequest_id"),
                    artifact_kind=(w.get("meta") or {}).get("artifact_kind"),
                ),
            )

            # -------- Round metrics (compact JSON) --------
            rounds_summary = []
            for ri, round_beam in enumerate(arena.get("iterations") or []):
                # each entry in iterations is already a small dict per beam item (variant/overall/k)
                rounds_summary.append({"round": ri, "beam": round_beam})

            if rounds_summary:
                self.memory.casebooks.add_scorable(
                    case_id=case.id,
                    role="arena_round_metrics",
                    pipeline_run_id=pipeline_run_id,
                    text=json.dumps({"rounds": rounds_summary}, ensure_ascii=False),
                    meta=_meta(),
                )

            # -------- Case attributes (typed; one value_* each) --------
            # numeric attrs
            try:
                self.memory.casebooks.set_case_attr(
                    case.id, "arena_beam_width", value_num=float(self.cfg.get("beam_width", 5))
                )
                self.memory.casebooks.set_case_attr(
                    case.id, "arena_rounds", value_num=float(len(arena.get("iterations") or []))
                )
                if "overall" in w_score:
                    self.memory.casebooks.set_case_attr(
                        case.id, "arena_winner_overall", value_num=float(w_score["overall"])
                    )
                if "k" in w_score:
                    self.memory.casebooks.set_case_attr(
                        case.id, "arena_winner_k", value_num=float(w_score["k"])
                    )
                if "c" in w_score:
                    self.memory.casebooks.set_case_attr(
                        case.id, "arena_winner_c", value_num=float(w_score["c"])
                    )
                if "g" in w_score:
                    self.memory.casebooks.set_case_attr(
                        case.id, "arena_winner_g", value_num=float(w_score["g"])
                    )
            except Exception:
                # do not fail the run on attribute write issues
                pass

            # text attrs
            try:
                if w.get("origin"):
                    self.memory.casebooks.set_case_attr(
                        case.id, "arena_winner_origin", value_text=str(w["origin"])
                    )
                sid = (w.get("meta") or {}).get("sidequest_id")
                if sid:
                    self.memory.casebooks.set_case_attr(
                        case.id, "arena_winner_sidequest_id", value_text=str(sid)
                    )
            except Exception:
                pass

            # -------- Optional SIS card (non-blocking) --------
            try:
                sis = getattr(self.container, "get_service", lambda *_: None)("sis")
                if sis:
                    cards = [
                        {"type": "metric", "title": "Winner Overall", "value": round(float(w_score.get("overall", 0.0)), 3)},
                        {"type": "bar", "title": "K/C/G", "series": [
                            {"name": "K", "value": round(float(w_score.get("k", 0.0)), 3)},
                            {"name": "C", "value": round(float(w_score.get("c", 0.0)), 3)},
                            {"name": "G", "value": round(float(w_score.get("g", 0.0)), 3)},
                        ]},
                        {"type": "list", "title": "Winner", "items": [
                            f"Origin: {w.get('origin')}",
                            f"Variant: {w.get('variant')}",
                            f"Rounds: {len(arena.get('iterations') or [])}",
                        ]},
                    ]
                    sis.publish_cards({
                        "scope": "arena",
                        "key": f"paper:{paper.get('id') or paper.get('doc_id')}|sec:{section.get('section_name')}",
                        "title": "Arena Winner",
                        "cards": cards,
                        "meta": {
                            "case_id": case.id,
                            "paper_id": str(paper.get("id") or paper.get("doc_id")),
                            "section_name": section.get("section_name"),
                        },
                    })
            except Exception:
                pass

        except Exception as e:
            # Never let persistence explode the agent; log and continue
            try:
                self.logger.log("ArenaPersistError", {
                    "err": str(e),
                    "paper_id": str(paper.get("id") or paper.get("doc_id")),
                    "section": section.get("section_name"),
                })
            except Exception:
                _logger.warning(f"_persist_arena failed: {e}")


    @staticmethod
    def _extract_claim_sentences(text: str) -> List[str]:
        import re
        sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', (text or '').strip()) if s.strip()]
        keep = [s for s in sents if any(w in s.lower() for w in ["show", "demonstrate", "evidence", "we", "results", "prove", "suggest"])]
        return keep[:8]

    @staticmethod
    def _cos_sim(a: List[float], b: List[float]) -> float:
        num = sum(x*y for x,y in zip(a,b))
        na = (sum(x*x for x in a))**0.5 or 1.0
        nb = (sum(x*x for x in b))**0.5 or 1.0
        return max(0.0, min(1.0, num/(na*nb)))
``n

## File: progress.py

`python
# stephanie/agents/learning/progress.py
class ProgressAdapter:
    def __init__(self, agent_progress, cfg):
        self.core = agent_progress  # reuse AgentProgress you already create
        self.cfg = cfg or {}

    def start_paper(self, paper, sections):
        self.core.start_paper(
            paper.get("id") or paper.get("doc_id"),
            paper.get("title", ""),
            len(sections),
        )

    def start_section(self, section, idx):
        self.core.start_section(section.get("section_name", "unknown"), idx)

    def stage(self, section, stage, **kv):
        self.core.stage(stage, section.get("section_name", "unknown"), **kv)

    def end_section(self, case, section, metrics):
        self.core.end_section(section.get("section_name", "unknown"), metrics)
``n

## File: proof.py

`python
# stephanie/agents/learning/proof.py
from dataclasses import dataclass
from typing import List, Dict, Any
import json


@dataclass
class AblationConfig:
    support_ids: List[str]  # ids from arena_candidate/corpus items
    seeds: int = 3  # repeat runs to smooth nondeterminism
    max_iterations: int = 3


class ProofOfAppliedKnowledge:
    def __init__(self, cfg, memory, container, logger, scorer):
        self.cfg, self.memory, self.container, self.logger, self.scorer = (
            cfg,
            memory,
            container,
            logger,
            scorer,
        )

    def _mask_corpus(self, corpus_items, mask_ids: List[str]):
        return [
            it
            for it in (corpus_items or [])
            if str(it.get("id")) not in set(map(str, mask_ids))
        ]

    def run_ablation(
        self,
        paper: Dict[str, Any],
        section: Dict[str, Any],
        baseline_ctx: Dict[str, Any],
        supports_to_mask: List[str],
        fetch_corpus_fn,
        build_candidates_fn,
        arena_run_fn,
        verify_improve_fn,
    ) -> Dict[str, Any]:
        """Re-run the section with supports masked out; return before/after metrics."""
        # 1) Reuse original corpus; if not available, re-fetch once.
        corpus = fetch_corpus_fn(section["section_text"])
        with_mask = self._mask_corpus(corpus, supports_to_mask)

        def _score_with(items):
            # Build candidates, run (arena|baseline) + verify loop once
            cands = build_candidates_fn(section, items)
            arena_res = arena_run_fn(
                section["section_text"], cands
            )  # respects cfg.use_arena
            baseline = arena_res["winner"]["text"] if arena_res else ""
            verify = verify_improve_fn(baseline, paper, section, baseline_ctx)
            return {
                "metrics": verify["metrics"],
                "iters": verify["iterations"],
            }

        # 2) Repeat to reduce variance (temperature 0 helps too)
        R = self.cfg.get("proof_repeats", 3)
        with_runs, without_runs = [], []
        for _ in range(R):
            with_runs.append(_score_with(corpus))
            without_runs.append(_score_with(with_mask))

        def agg(rs):  # simple mean
            m = lambda k: sum(r["metrics"][k] for r in rs) / len(rs)
            return {
                "overall": m("overall"),
                "knowledge": m("knowledge_score"),
                "grounding": m("grounding"),
                "clarity": m("clarity"),
                "iters": sum(len(r["iters"]) for r in rs) / len(rs),
            }

        with_m, without_m = agg(with_runs), agg(without_runs)
        delta = {
            k: with_m[k] - without_m[k]
            for k in ("overall", "knowledge", "grounding", "clarity")
        }
        out = {"with": with_m, "without": without_m, "delta": delta, "runs": R}

        # 3) Persist an ablation_result scorable
        try:
            self.memory.casebooks.add_scorable(
                case_id=baseline_ctx.get("case_id"),
                role="ablation_result",
                text=json.dumps(
                    {"mask": list(map(str, supports_to_mask)), **out}
                ),
                pipeline_run_id=baseline_ctx.get("pipeline_run_id"),
                meta={"type": "proof"},
            )
        except Exception:
            pass
        return out
``n

## File: scoring.py

`python
# stephanie/agents/learning/scoring.py
from __future__ import annotations
from typing import Dict, Any, Tuple, List
import re

class Scoring:
    def __init__(self, cfg, memory, container, logger):
        self.cfg, self.memory, self.container, self.logger = cfg, memory, container, logger
        try:
            from stephanie.scoring.scorer.knowledge_scorer import KnowledgeScorer
            self.knowledge = KnowledgeScorer(cfg.get("knowledge_scorer", {}), memory, container, logger)
        except Exception:
            self.knowledge = None

    def rubric_dims(self, text: str, ref: str) -> Tuple[float, float]:
        sents = [s for s in re.split(r"[.!?]\s+", (text or "").strip()) if s]
        avg_len = sum(len(s.split()) for s in sents) / max(1, len(sents))
        clarity = max(0.0, min(1.0, 1.1 - (abs(avg_len - 22) / 22)))
        def toks(t): return set(re.findall(r"\b\w+\b", (t or "").lower()))
        inter = len(toks(text) & toks(ref))
        grounding = max(0.0, min(1.0, inter / max(30, len(toks(ref)) or 1)))
        return clarity, grounding

    def weaknesses(self, summary: str, ref: str) -> List[str]:
        out = []
        if len(summary or "") < 400: out.append("too short / thin detail")
        if "we propose" in (ref or "").lower() and "we propose" not in (summary or "").lower():
            out.append("misses core claim language")
        if (summary or "").count("(") != (summary or "").count(")"):
            out.append("formatting/parens issues")
        return out

    def score_summary(self, text: str, paper: Dict[str, Any], section: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        clarity, grounding = self.rubric_dims(text, section.get("section_text", ""))
        if self.knowledge:
            goal_text = f"{paper.get('title','')}\n\n{paper.get('abstract','')}"
            meta = {"text_len_norm": min(1.0, len(text)/2000.0)}
            # Support either .predict or .model.predict
            if hasattr(self.knowledge, "predict"):
                p, comps = self.knowledge.predict(goal_text, text, meta=meta, return_components=True)
            elif hasattr(self.knowledge, "model") and hasattr(self.knowledge.model, "predict"):
                p, comps = self.knowledge.model.predict(goal_text, text, meta=meta, return_components=True)
            else:
                p, comps = 0.5, {"probability": 0.5}
            knowledge = float((comps or {}).get("probability", p))
        else:
            knowledge = 0.5*clarity + 0.5*grounding
        overall = 0.6*knowledge + 0.25*clarity + 0.15*grounding
        return {
            "overall": overall,
            "knowledge_score": knowledge,
            "clarity": clarity,
            "grounding": grounding,
            "weaknesses": self.weaknesses(text, section.get("section_text","")),
        }

    def score_candidate(self, text: str, section_text: str) -> Dict[str, float]:
        dims = self.score_summary(text, {"title": "", "abstract": ""}, {"section_text": section_text}, {})
        k, c, g = dims["knowledge_score"], dims["clarity"], dims["grounding"]
        overall = 0.6*k + 0.25*c + 0.15*g
        verified = (g >= 0.45) and (len(text) >= self.cfg.get("min_verified_len", 250))
        return {"k": k, "c": c, "g": g, "overall": overall, "verified": bool(verified)}
``n

## File: strategy_manager.py

`python
# stephanie/agents/learning/strategy_manager.py
from __future__ import annotations
from dataclasses import dataclass, asdict, replace
from typing import Any, Dict, List, Optional, Tuple
import json
import random
import time
import math
import logging

from stephanie.utils.json_sanitize import dumps_safe

_logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class Strategy:
    """
    Tunable knobs for verification. Immutable so we can reason about versions.
    Use StrategyManager.set_state(...) to change.
    """
    verification_threshold: float = 0.85
    skeptic_weight: float = 0.34
    editor_weight: float = 0.33
    risk_weight: float = 0.33
    version: int = 1

class StrategyManager:
    """
    Owns strategy knobs, runs lightweight A/B enrollment and validation,
    and records evolution breadcrumbs into casebooks as scorables.
    """

    def __init__(self, cfg, memory, container, logger):
        self.cfg, self.memory, self.container, self.logger = cfg, memory, container, logger

        self.casebook_tag: str = cfg.get("casebook_action", "blog")
        self.min_gain: float = float(cfg.get("min_gain", 0.01))
        self.high_gain: float = float(cfg.get("high_gain", 0.03))

        # A/B validation knobs
        self.history_limit: int = int(cfg.get("strategy_test_history", 20))
        self.min_samples: int = int(cfg.get("ab_min_samples", 10))
        self.window_seconds: Optional[int] = (
            int(cfg["ab_window_seconds"]) if "ab_window_seconds" in cfg else None
        )

        # Deterministic, seedable randomness (or you can hash case_id)
        seed = cfg.get("rng_seed")
        self.rng = random.Random(seed) if seed is not None else random.Random()

        # Current state (immutable dataclass)
        self.state: Strategy = Strategy(
            verification_threshold=float(cfg.get("verification_threshold", 0.85)),
            skeptic_weight=float(cfg.get("skeptic_weight", 0.34)),
            editor_weight=float(cfg.get("editor_weight", 0.33)),
            risk_weight=float(cfg.get("risk_weight", 0.33)),
            version=1,
        )

        self._evolution_log: List[Dict[str, Any]] = []

    # ---------- Public helpers ----------
    def get_state(self) -> Strategy:
        return self.state

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self.state)

    def set_state(self, **fields) -> None:
        """Safely replace fields (keeps immutability)."""
        self.state = replace(self.state, **fields)

    def bump_version(self) -> None:
        self.set_state(version=self.state.version + 1)

    # ---------- Recording ----------
    def record_state(self, context: Optional[Dict[str, Any]], tag: str = "pre_change"):
        payload = {**self.as_dict(), "tag": tag, "timestamp": time.time()}
        try:
            case_id = (context or {}).get("case_id")
            if case_id:
                self.memory.casebooks.add_scorable(
                    case_id=case_id,
                    role="strategy_state",
                    text=dumps_safe(payload),
                    pipeline_run_id=(context or {}).get("pipeline_run_id"),
                    meta={"tag": tag},
                )
        except Exception:
            # Never block on telemetry
            pass

    # ---------- Evolution logic ----------
    def propose(self, avg_gain: float) -> Strategy:
        """
        Create a proposed strategy (copy) based on observed avg_gain.
        """
        s = self.state
        if avg_gain < self.min_gain:
            change = 0.06 if avg_gain < 0.005 else 0.03
            return replace(
                s,
                skeptic_weight=min(0.60, s.skeptic_weight + change),
                editor_weight=max(0.20, s.editor_weight - change / 2),
                risk_weight=max(0.20, s.risk_weight - change / 2),
            )
        elif avg_gain > self.high_gain:
            return replace(s, verification_threshold=max(0.80, s.verification_threshold - 0.01))
        return s  # no change

    def _pick_group(self, case_id: Optional[int]) -> str:
        """
        Deterministic-ish group selection.
        If rng_seed provided, uses seeded RNG; else hash case_id fallback.
        """
        if case_id is not None and self.cfg.get("ab_hash_assign", True) and self.cfg.get("rng_seed") is None:
            # Stable across runs on same case_id
            g = "B" if (hash((case_id, self.casebook_tag)) & 1) else "A"
            return g
        return "B" if self.rng.random() < 0.5 else "A"

    def evolve(self, iterations: List[Dict[str, Any]], context: Optional[Dict[str, Any]]):
        """
        Called at the end of a section’s refinement loop.
        Computes avg gain, proposes changes, enrolls next unit in A or B,
        records breadcrumb, and (if B) updates current state.
        """
        if len(iterations) < 2:
            self.record_state(context, "pre_change")
            return

        gains = [
            (iterations[i]["score"] - iterations[i - 1]["score"])
            for i in range(1, len(iterations))
            if "score" in iterations[i] and "score" in iterations[i - 1]
        ]
        avg_gain = (sum(gains) / len(gains)) if gains else 0.0

        old = self.as_dict()
        self.record_state(context, "pre_change")
        proposed = self.propose(avg_gain)

        case_id = (context or {}).get("case_id")
        group = self._pick_group(case_id)

        if group == "B" and proposed != self.state:
            # adopt proposed knobs for the NEXT unit
            self.state = proposed
            self.bump_version()

        self._record_ab(context, old, proposed, group, avg_gain)

        if group == "B":
            self._evolution_log.append(
                {"avg_gain": round(avg_gain, 4), "old": old, "new": self.as_dict(), "timestamp": time.time()}
            )

    def _record_ab(self, context, old, new: Strategy, group: str, avg_gain: float):
        payload = {
            "test_group": group,
            "avg_gain": float(avg_gain),
            "old_strategy": old,
            "new_strategy": asdict(new),
            "timestamp": time.time(),
        }
        try:
            case_id = (context or {}).get("case_id")
            if case_id:
                self.memory.casebooks.add_scorable(
                    case_id=case_id,
                    role="strategy_ab_enroll",
                    text=dumps_safe(payload),
                    pipeline_run_id=(context or {}).get("pipeline_run_id"),
                    meta={"group": group},
                )
        except Exception:
            pass

    def track_section(self, case, iterations):
        """
        Optional compact attribute (avg_gain, iters, version) for dashboards/queries.
        """
        try:
            if not iterations:
                return
            gains = [
                (iterations[i]["score"] - iterations[i - 1]["score"])
                for i in range(1, len(iterations))
                if "score" in iterations[i] and "score" in iterations[i - 1]
            ]
            avg_gain = (sum(gains) / len(gains)) if gains else 0.0
            payload = {
                "avg_gain": round(avg_gain, 6),
                "iteration_count": len(iterations),
                "strategy": self.as_dict(),
                "timestamp": time.time(),
            }
            self.memory.casebooks.set_case_attr(case.id, "strategy_evolution", value_json=payload)
        except Exception:
            pass

    # ---------- A/B effectiveness ----------
    def _ab_results(self) -> List[Dict[str, Any]]:
        """
        Pull recent A/B enrollments + final scores from scorables.
        Applies history/window filters to avoid stale leakage.
        """
        now = time.time()
        results: List[Dict[str, Any]] = []
        casebooks = self.memory.casebooks.get_casebooks_by_tag(self.casebook_tag) or []

        for cb in reversed(casebooks):  # bias toward latest casebooks
            for case in reversed(self.memory.casebooks.get_cases_for_casebook(cb.id) or []):
                group, ts, perf = None, 0.0, None
                for s in self.memory.casebooks.list_scorables(case.id) or []:
                    if s.role == "strategy_ab_enroll":
                        try:
                            rec = json.loads(s.text) if isinstance(s.text, str) else (s.text or {})
                            group = rec.get("test_group")
                            ts = float(rec.get("timestamp", 0.0))
                        except Exception:
                            pass
                    elif s.role == "metrics":
                        try:
                            rec = json.loads(s.text) if isinstance(s.text, str) else (s.text or {})
                            final = (rec or {}).get("final_scores") or {}
                            if "overall" in final:
                                perf = float(final["overall"])
                        except Exception:
                            pass

                if group in ("A", "B") and isinstance(perf, (int, float)):
                    if self.window_seconds is not None and (now - ts) > self.window_seconds:
                        continue
                    results.append({"group": group, "performance": perf, "timestamp": ts, "case_id": case.id})
                if len(results) >= self.history_limit:
                    break
            if len(results) >= self.history_limit:
                break

        results.sort(key=lambda r: r["timestamp"], reverse=True)
        return results[: self.history_limit]

    @staticmethod
    def _mean_stdev(xs: List[float]) -> Tuple[float, float]:
        if not xs:
            return 0.0, 0.0
        m = sum(xs) / len(xs)
        if len(xs) == 1:
            return m, 0.0
        v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
        return m, math.sqrt(max(v, 0.0))

    @staticmethod
    def _cohens_d(a: List[float], b: List[float]) -> float:
        if len(a) < 2 or len(b) < 2:
            return 0.0
        ma, sa = StrategyManager._mean_stdev(a)
        mb, sb = StrategyManager._mean_stdev(b)
        # pooled SD (unbiased)
        sp_num = ((len(a) - 1) * (sa ** 2) + (len(b) - 1) * (sb ** 2))
        sp_den = (len(a) + len(b) - 2)
        if sp_den <= 0 or sp_num <= 0:
            return 0.0
        sp = math.sqrt(sp_num / sp_den)
        if sp == 0:
            return 0.0
        return (mb - ma) / sp

    @staticmethod
    def _welch_ttest(a: List[float], b: List[float]) -> Tuple[float, float]:
        """
        Returns (t_stat, p_two_sided) using Welch's t-test (approx, no deps).
        """
        na, nb = len(a), len(b)
        if na < 2 or nb < 2:
            return 0.0, 1.0
        ma, sa = StrategyManager._mean_stdev(a)
        mb, sb = StrategyManager._mean_stdev(b)
        sa2, sb2 = sa ** 2, sb ** 2
        denom = math.sqrt((sa2 / na) + (sb2 / nb)) or 1e-9
        t = (mb - ma) / denom
        # Welch–Satterthwaite dof
        num = (sa2 / na + sb2 / nb) ** 2
        den = ((sa2 / na) ** 2) / (na - 1) + ((sb2 / nb) ** 2) / (nb - 1)
        dof = max(num / den, 1.0) if den > 0 else 1.0

        # two-sided p via survival of Student's t (approx using normal fallback)
        # For small samples this is rough; good enough for telemetry.
        # Convert to normal as an approximation:
        # p ≈ 2 * (1 - Φ(|t|)), Φ normal CDF
        p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t) / math.sqrt(2))))
        return t, p

    def validate_ab(self, context: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Aggregate recent A/B samples and return a small stats bundle.
        Does not mutate state; caller can decide to commit/revert elsewhere.
        """
        tests = self._ab_results()
        if not tests or len(tests) < self.min_samples:
            return None

        perf_a = [r["performance"] for r in tests if r["group"] == "A"]
        perf_b = [r["performance"] for r in tests if r["group"] == "B"]
        if not perf_a or not perf_b:
            return None

        mean_a, sd_a = self._mean_stdev(perf_a)
        mean_b, sd_b = self._mean_stdev(perf_b)
        delta = mean_b - mean_a
        d = self._cohens_d(perf_a, perf_b)
        t, p = self._welch_ttest(perf_a, perf_b)

        out = {
            "samples_A": len(perf_a),
            "samples_B": len(perf_b),
            "mean_A": mean_a,
            "mean_B": mean_b,
            "sd_A": sd_a,
            "sd_B": sd_b,
            "delta_B_minus_A": delta,
            "cohens_d": d,
            "welch_t": t,
            "p_value_two_sided": p,
            "timestamp": time.time(),
        }
        try:
            self.logger.log("StrategyAB_Validation", out)
        except Exception:
            pass
        return out

    def _evolve_strategy(self, iters: List[Dict[str, Any]], context: Optional[Dict[str, Any]]):
        if len(iters) < 2:
            # still record a point so we can analyze later
            self._record_strategy_state(context, tag="pre_change")
            return

        gains = [iters[i]["score"] - iters[i-1]["score"] for i in range(1, len(iters))]
        avg_gain = sum(gains) / len(gains) if gains else 0.0

        old_strategy = {
            "verification_threshold": self.strategy.verification_threshold,
            "skeptic_weight": self.strategy.skeptic_weight,
            "editor_weight": self.strategy.editor_weight,
            "risk_weight": self.strategy.risk_weight,
            "version": self.strategy.version,
        }

        # always record the current state (for later comparison)
        self._record_strategy_state(context, tag="pre_change")

        # propose (don’t apply yet)
        proposed = self._propose_strategy_changes(avg_gain)

        # A/B enroll for *next* work unit
        #  - A keeps current knobs
        #  - B uses proposed knobs
        if random.random() < 0.5:
            # switch to proposed for the next work unit
            self.strategy = proposed
            group = "B"
            # bump version only when actually switching
            self.strategy.version += 1
        else:
            group = "A"

        # record the assignment
        self._record_strategy_test(context, old_strategy=old_strategy, new_strategy=proposed, test_group=group, avg_gain=avg_gain)

        # keep the in-memory evolution event so your longitudinal view can see changes
        if group == "B":
            new_strategy = {
                "verification_threshold": self.strategy.verification_threshold,
                "skeptic_weight": self.strategy.skeptic_weight,
                "editor_weight": self.strategy.editor_weight,
                "risk_weight": self.strategy.risk_weight,
                "version": self.strategy.version,
            }
            event = {
                "avg_gain": round(avg_gain, 4),
                "change_amount": None,  # already in deltas above
                "old": old_strategy,
                "new": new_strategy,
                "iteration_count": len(iters),
                "timestamp": time.time(),
            }
            self._evolution_log.append(event)
            _logger.info(f"LfL_Strategy_Evolved(AB): {event}")
            if context is not None:
                context.setdefault("strategy_evolution", []).append(event)

    def _propose_strategy_changes(self, avg_gain: float) -> Strategy:
        """Return a *copy* of the current strategy with proposed adjustments applied."""
        proposed = Strategy(
            verification_threshold=self.strategy.verification_threshold,
            skeptic_weight=self.strategy.skeptic_weight,
            editor_weight=self.strategy.editor_weight,
            risk_weight=self.strategy.risk_weight,
            version=self.strategy.version,
        )
        change_amount = 0.06 if avg_gain < self.cfg.get("min_gain", 0.01) and avg_gain < 0.005 else 0.03
        if avg_gain < self.cfg.get("min_gain", 0.01):
            proposed.skeptic_weight = min(0.60, proposed.skeptic_weight + change_amount)
            proposed.editor_weight  = max(0.20, proposed.editor_weight - change_amount / 2)
            proposed.risk_weight    = max(0.20, proposed.risk_weight   - change_amount / 2)
        elif avg_gain > self.cfg.get("high_gain", 0.03):
            proposed.verification_threshold = max(0.80, proposed.verification_threshold - 0.01)
        return proposed

    def _record_strategy_test(self, context: Optional[Dict[str, Any]], old_strategy: Dict[str, Any],
                            new_strategy: Strategy, test_group: str, avg_gain: float) -> None:
        """Record that we entered A or B for upcoming work."""
        payload = {
            "test_group": test_group,
            "avg_gain": avg_gain,
            "old": old_strategy,
            "new": {
                "verification_threshold": new_strategy.verification_threshold,
                "skeptic_weight": new_strategy.skeptic_weight,
                "editor_weight": new_strategy.editor_weight,
                "risk_weight": new_strategy.risk_weight,
                "version": new_strategy.version,
            },
            "timestamp": time.time(),
        }
        try:
            case_id = (context or {}).get("case_id")
            if case_id:
                self.memory.casebooks.add_scorable(
                    case_id=case_id, role="strategy_ab_enroll",
                    text=dumps_safe(payload),
                    pipeline_run_id=(context or {}).get("pipeline_run_id"),
                    meta={"group": test_group}
                )
        except Exception:
            pass


    def _record_strategy_state(self, context: Optional[Dict[str, Any]], tag: str = "pre_change") -> None:
        """Record current strategy knobs so we can compare later."""
        payload = {
            "tag": tag,
            "verification_threshold": self.strategy.verification_threshold,
            "skeptic_weight": self.strategy.skeptic_weight,
            "editor_weight": self.strategy.editor_weight,
            "risk_weight": self.strategy.risk_weight,
            "version": self.strategy.version,
            "timestamp": time.time(),
        }
        try:
            # If a current case_id is available in context, attach it; otherwise, store on the pipeline run
            case_id = (context or {}).get("case_id")
            if case_id:
                self.memory.casebooks.add_scorable(
                    case_id=case_id, role="strategy_state", text=dumps_safe(payload),
                    pipeline_run_id=(context or {}).get("pipeline_run_id"),
                    meta={"tag": tag}
                )
        except Exception:
            pass
``n

## File: summarizer.py

`python
# stephanie/agents/learning/summarizer.py
from __future__ import annotations
from typing import Dict, Any, List
import json

class Summarizer:
    def __init__(self, cfg, memory, container, logger, strategy, scoring=None, prompt_loader=None, call_llm=None):
        self.cfg, self.memory, self.container, self.logger = cfg, memory, container, logger
        self.strategy = strategy
        # fallbacks
        self.scoring = scoring or (container.get("scoring") if hasattr(container, "get") else None)
        if self.scoring is None:
            from .scoring import Scoring
            self.scoring = Scoring(cfg, memory, container, logger)
        self.prompt_loader = prompt_loader or (container.get("prompt_loader") if hasattr(container, "get") else None)
        self.call_llm = call_llm or (container.get("call_llm") if hasattr(container, "get") else None)

    def baseline(self, paper: Dict[str, Any], section: Dict[str, Any], critical_msgs: List[Dict[str, Any]], context: Dict[str, Any]) -> str:
        merged = {
            "title": paper.get("title", ""),
            "abstract": paper.get("abstract", ""),
            "focus_section": section.get("section_name"),
            "focus_text": section.get("section_text", "")[:5000],
            "hints": "\n".join((m.get("assistant_text") or m.get("text") or "") 
                for m in (critical_msgs[:6] if critical_msgs else [])),
            **context,
        }
        prompt = self.prompt_loader.from_file("baseline_summary", self.cfg, merged)
        return self.call_llm(prompt, merged)

    def improve_once(self, paper: Dict[str, Any], section: Dict[str, Any], current_summary: str, context: Dict[str, Any], return_attribution: bool=False):
        metrics = self.scoring.score_summary(current_summary, paper, section, context)
        merged = {
            "title": paper.get("title", ""),
            "section_name": section.get("section_name"),
            "section_text": section.get("section_text", "")[:6000],
            "current_summary": current_summary,
            "skeptic_weight": self.strategy.state.skeptic_weight,
            "editor_weight": self.strategy.state.editor_weight,
            "risk_weight": self.strategy.state.risk_weight,
            "weaknesses": json.dumps(metrics.get("weaknesses", []), ensure_ascii=False),
            **context,
        }
        prompt = self.prompt_loader.from_file("improve_summary", self.cfg, merged)
        improved = self.call_llm(prompt, context)

        if not return_attribution:
            return improved

        # --- attribution (AR/AKL) ---
        claims = self._extract_claim_sentences(improved)
        rpool = (context.get("retrieval_items") or []) + (context.get("arena_initial_pool") or [])
        th = float(self.cfg.get("applied_knowledge",{}).get("attr_sim_threshold", 0.75))
        matches = self._attribute_claims(claims, rpool, th)

        # store a compact scorable for this improve step (if a case is present)
        try:
            case_id = context.get("case_id")
            if case_id and matches:
                payload = {"claims": matches, "threshold": th, "timestamp": time.time()}
                self.memory.casebooks.add_scorable(
                    case_id=case_id, role="improve_attribution",
                    text=json.dumps(payload, ensure_ascii=False),
                    pipeline_run_id=context.get("pipeline_run_id"),
                    meta={"iteration": context.get("iteration")}
                )
        except Exception:
            pass

        return {"text": improved, "attribution": matches}

    def verify_and_improve(self, baseline: str, paper: Dict[str, Any], section: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        current = baseline
        iters: List[Dict[str, Any]] = []
        max_iter = int(self.cfg.get("max_iterations", 3))
        last_attr_supported = False
        first_score_with_knowledge = None
        metrics = {"overall": 0.0}  # default

        for i in range(1, max_iter + 1):
            metrics = self.scoring.score_summary(current, paper, section, context)
            # annotate whether this iteration's text was influenced by knowledge (from prior improvement step)
            it = {"iteration": i, "score": metrics["overall"], "metrics": metrics, "knowledge_applied": bool(last_attr_supported)}
            iters.append(it)

            if last_attr_supported and first_score_with_knowledge is None:
                first_score_with_knowledge = metrics["overall"]

            if metrics["overall"] >= self.strategy.state.verification_threshold:
                break

            # run one improvement with attribution
            improved = self.improve_once(paper, section, current, {**context, "iteration": i}, return_attribution=True)
            if isinstance(improved, dict):
                current = improved["text"]
                last_attr_supported = bool(improved.get("attribution"))
            else:
                current = improved
                last_attr_supported = False

        # rollups that Persistence can store
        k_applied_iters = sum(1 for it in iters if it.get("knowledge_applied"))
        k_applied_lift = (metrics["overall"] - first_score_with_knowledge) if first_score_with_knowledge is not None else 0.0

        self.strategy.evolve(iters, context)
        return {"summary": current, "metrics": {**metrics, "knowledge_applied_iters": k_applied_iters, "knowledge_applied_lift": k_applied_lift}, "iterations": iters}

    def _extract_claim_sentences(self, text: str) -> List[str]:
        import re
        sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', (text or '').strip()) if s.strip()]
        triggers = ("show", "demonstrate", "evidence", "result", "increase", "decrease", "improve", "we propose", "we find")
        return [s for s in sents if any(t in s.lower() for t in triggers)][: int(self.cfg.get("applied_knowledge",{}).get("max_claims", 8))]

    def _cos_sim(self, a, b) -> float:
        num = sum(x*y for x,y in zip(a,b))
        na = (sum(x*x for x in a))**0.5 or 1.0
        nb = (sum(x*x for x in b))**0.5 or 1.0
        return max(0.0, min(1.0, num/(na*nb)))

    def _attribute_claims(self, claims: List[str], sources: List[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
        emb = getattr(self.memory, "embedding", None) if self.cfg.get("applied_knowledge",{}).get("use_embeddings", True) else None
        out = []
        # pre-embed sources (cap for speed)
        S = [{"meta": s, "v": (emb.get_or_create((s.get("text") or "")[:2000]) if emb else None)} for s in (sources[:50] if sources else [])]
        for c in claims:
            if not c: 
                continue
            best, best_sim = None, 0.0
            if emb and S:
                cv = emb.get_or_create(c)
                for s in S:
                    sim = self._cos_sim(cv, s["v"])
                    if sim > best_sim:
                        best_sim, best = sim, s["meta"]
            else:
                # fallback: token overlap
                import re
                ctoks = set(re.findall(r"\b\w+\b", c.lower()))
                for s in S or sources:
                    stoks = set(re.findall(r"\b\w+\b", (s.get("text","").lower())))
                    sim = len(ctoks & stoks) / float(max(1, len(ctoks)|len(stoks)))
                    if sim > best_sim:
                        best_sim, best = sim, s
            if best and best_sim >= threshold:
                out.append({"claim": c, "support": {"text": best.get("text","")[:220], "origin": best.get("origin"), "variant": best.get("variant")}, "similarity": round(best_sim, 3)})
        return out
``n
