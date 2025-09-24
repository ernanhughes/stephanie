# stephanie/services/cbr_service.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from stephanie.constants import INCLUDE_MARS
from stephanie.scoring.calculations.mars_calculator import MARSCalculator
from stephanie.scoring.scorer.scorable_ranker import ScorableRanker
from stephanie.services.service_protocol import Service


# ---- Public view of a case (normalized) ----
@dataclass
class CaseView:
    id: Any
    goal_text: str
    summary: str
    lessons: str
    winner_rationale: str
    score: Optional[float] = None
    casebook_name: Optional[str] = None


LLMFn = Callable[[str, Dict[str, Any] | None], str]


class CBRService(Service):
    """
    Case-Book Based Reasoning:
      R1) RETRIEVE  cases from the paper/blog casebook (fallback: global embedding search)
      R2) REUSE    (adapt) best cases to the new goal using the LLM
      R3) REVISE   score + iterate (simple loop now; MCTS hook included)
      R4) RETAIN   persist the new best case back into the casebook
    """

    def __init__(self, cfg, memory, container, logger):
        self.cfg = cfg or {}
        self.memory = memory
        self.container = container
        self.logger = logger
        self._initialized = False

        c = self.cfg.get("cbr", {}) or {}
        self.pool_k = int(c.get("pool_k", 64))
        self.min_text_len = int(c.get("min_text_len", 8))
        self.adapt_iters = int(c.get("adapt_iters", 1))   # keep small; agent already iterates
        self.top_k_default = int(c.get("top_k", 5))

        # LLM adapter
        self.llm = container.get("llm") 

        # scoring
        self.scorer = container.get("scoring") 

        self.mars = MARSCalculator(cfg, memory, self.container, logger) if cfg.get(INCLUDE_MARS, True) else None

    # --- Service protocol ---
    def initialize(self, **kwargs) -> None:
        self._initialized = True
        if self.logger:
            self.logger.log("CBRv2Init", {"pool_k": self.pool_k, "adapt_iters": self.adapt_iters})

    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy" if self._initialized else "uninitialized",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "dependencies": {
                "embedding": self.memory.embedding.name,
            },
        }

    def shutdown(self) -> None:
        self._initialized = False

    @property
    def name(self) -> str:
        return "cbr-service-v2"

    # ============================================================
    # ==============  PUBLIC, SIMPLE ENTRY POINTS  ===============
    # ============================================================

    def retrieve(self, *, goal_text: str, top_k: int | None = None,
                 casebook_name: Optional[str] = None) -> List[CaseView]:
        """
        Prefer cases in the given casebook (blog::<paper_id>::<slug>), else fallback to global search.
        """
        if not goal_text or not goal_text.strip():
            return []

        # 1) Casebook-scoped retrieval if we can
        cases: List[CaseView] = []
        if casebook_name and hasattr(self.memory, "casebooks"):
            try:
                # try a native casebook search if you have it; else filter all cases by casebook_name
                rows = self.memory.casebooks.search(
                    query_text=goal_text,
                    casebook_name=casebook_name,
                    top_k=self.pool_k,
                )
                for r in rows or []:
                    meta = getattr(r, "meta", {}) or {}
                    cases.append(self._from_casebook_row(r, casebook_name, meta))
            except Exception as e:
                if self.logger:
                    self.logger.log("CBRCasebookSearchWarn", {"error": str(e)})

        # 2) Fallback to embedding search across all “case” scorables
        try:
            hits = self.memory.embedding.search_related_scorables(
                goal_text,
                top_k=self.pool_k,
                include_ner=False,
                target_type="case",
            )
            for h in hits or []:
                cv = self._from_embedding_hit(h)
                # keep scoping if a casebook_name was requested
                if not casebook_name or cv.casebook_name == casebook_name:
                    cases.append(cv)
        except Exception as e:
            if self.logger:
                self.logger.log("CBREmbeddingSearchWarn", {"error": str(e)})

        # length filter + sort by score desc (None -> worst)
        filtered = [c for c in cases if max(len(c.goal_text), len(c.summary)) >= self.min_text_len]
        filtered.sort(key=lambda c: (c.score is not None, c.score), reverse=True)
        k = int(top_k or self.top_k_default)
        return filtered[:k]

    def retrieve_pack(self, *, goal_text: str, k: int = 3, casebook_name: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Shape the cases into the exact pack your Track-C agent expects.
        Returns: [{"title", "why_it_won", "patch", "summary"}, ...]
        """
        out = []
        for c in self.retrieve(goal_text=goal_text, top_k=k, casebook_name=casebook_name):
            out.append({
                "title": (c.goal_text or "")[:160],
                "why_it_won": (c.winner_rationale or "")[:240],
                "patch": (c.lessons or "")[:240],
                "summary": (c.summary or "")[:400],
            })
        return out

    def adapt(self, *, goal_text: str, documents_text: str,
              base: CaseView, context: Optional[Dict[str, Any]] = None) -> str:
        """
        LLM-based adaptation that reuses lessons and rationale from a prior case.
        """
        if not self.llm:
            # minimal fallback: return original summary
            return base.summary

        prompt = f"""You are a Case-Based Reasoning adapter.

NEW GOAL:
{goal_text}

CURRENT CONTEXT (documents excerpt):
\"\"\"{documents_text[:2000]}\"\"\"

PAST CASE:
- Prior goal: {base.goal_text}
- Why it won: {base.winner_rationale}
- Lessons: {base.lessons}
- Best summary/solution:
\"\"\"{base.summary[:1800]}\"\"\"

Task:
Adapt the past case to the NEW GOAL, using its lessons. Be concrete and faithful to the provided context.
Return only the adapted draft (no commentary). Keep it concise and structured.
"""
        try:
            return (self.llm(prompt, context or {}) or "").strip()
        except Exception as e:
            if self.logger:
                self.logger.log("CBRAdaptLLMError", {"error": str(e)})
            return base.summary

    def assess(self, *, goal_text: str, draft_text: str, context: Optional[Dict[str, Any]] = None) -> float:
        """
        Score a draft (0..1). Uses MARS if available, else ranker probability, else a neutral 0.5.
        """
        try:
            if self.mars:
                s = self.mars.score(goal_text=goal_text, text=draft_text, context=context or {})
                return float(max(0.0, min(1.0, s)))
        except Exception:
            pass
        try:
            # ScorableRanker may provide a calibrated 'preference' or probability
            p = self.scorer.score_pairwise(query_text=goal_text, pos_text=draft_text, neg_text="", context=context or {})
            return float(max(0.0, min(1.0, p)))
        except Exception:
            return 0.5

    def revise(self, *, goal_text: str, documents_text: str,
               seeds: List[CaseView], iters: int | None = None,
               context: Optional[Dict[str, Any]] = None) -> Tuple[str, float, CaseView]:
        """
        Simple hill-climb: adapt each seed once per iter; keep best draft.
        Returns (best_text, best_score, best_seed_case).
        """
        iters = int(iters or self.adapt_iters)
        best_text, best_score, best_seed = "", -1.0, None

        for _ in range(max(1, iters)):
            for s in seeds:
                draft = self.adapt(goal_text=goal_text, documents_text=documents_text, base=s, context=context)
                score = self.assess(goal_text=goal_text, draft_text=draft, context=context)
                if score > best_score:
                    best_text, best_score, best_seed = draft, score, s
        return best_text, best_score, best_seed

    def retain(self, *, casebook_name: str, goal_text: str, draft_text: str,
               seed: Optional[CaseView], score: float,
               extra_meta: Optional[Dict[str, Any]] = None) -> Any:
        """
        Persist new/updated case back into the casebook.
        """
        meta = {
            "goal_text": goal_text,
            "summary": draft_text,
            "lessons": (seed.lessons if seed else ""),
            "winner_rationale": (seed.winner_rationale if seed else "derived_by_adaptation"),
            "score": float(score),
            "origin": "cbr_v2",
        }
        if extra_meta:
            meta.update(extra_meta)

        try:
            if hasattr(self.memory, "casebooks") and hasattr(self.memory.casebooks, "add"):
                case_id = self.memory.casebooks.add(
                    casebook_name=casebook_name,
                    role="case",
                    text=draft_text,
                    meta=meta,
                )
                if self.logger:
                    self.logger.log("CBRRetainSaved", {"casebook": casebook_name, "case_id": getattr(case_id, "id", case_id)})
                return case_id
        except Exception as e:
            if self.logger:
                self.logger.log("CBRRetainError", {"error": str(e), "casebook": casebook_name})
        return None

    # ---- optional: a tiny MCTS-like search (breadth-limited) ----
    def mcts_search(self, *, goal_text: str, documents_text: str, seeds: List[CaseView],
                    depth: int = 2, width: int = 3, context: Optional[Dict[str, Any]] = None) -> Tuple[str, float, List[CaseView]]:
        """
        Very small beam-style search (width) for depth steps. Good enough to start.
        Returns (best_text, best_score, path_seed_cases)
        """
        frontier: List[Tuple[str, float, List[CaseView]]] = []
        # seed layer
        for s in seeds[:width]:
            draft = self.adapt(goal_text=goal_text, documents_text=documents_text, base=s, context=context)
            score = self.assess(goal_text=goal_text, draft_text=draft, context=context)
            frontier.append((draft, score, [s]))

        best_draft, best_score, best_path = ("", -1.0, [])
        for _ in range(max(1, depth - 1)):
            # expand current frontier with top seeds again (reuse top-k as “moves”)
            frontier.sort(key=lambda t: t[1], reverse=True)
            new_frontier: List[Tuple[str, float, List[CaseView]]] = []
            for draft, _, path in frontier[:width]:
                for s in seeds[:width]:
                    ndraft = self.adapt(goal_text=goal_text, documents_text=documents_text, base=s, context=context)
                    nscore = self.assess(goal_text=goal_text, draft_text=ndraft, context=context)
                    new_frontier.append((ndraft, nscore, path + [s]))
                    if nscore > best_score:
                        best_draft, best_score, best_path = ndraft, nscore, path + [s]
            frontier = new_frontier

        if best_score < 0 and frontier:
            best_draft, best_score, best_path = max(frontier, key=lambda t: t[1])
        return best_draft, best_score, best_path

    def _from_casebook_row(self, row: Any, casebook_name: str, meta: Dict[str, Any]) -> CaseView:
        goal_text = (meta.get("goal_text") or meta.get("title") or meta.get("goal") or "").strip()
        summary = (meta.get("summary") or getattr(row, "text", "") or "").strip()
        lessons = (meta.get("lessons") or "").strip()
        winner_rationale = (meta.get("winner_rationale") or meta.get("scores", {}).get("winner_rationale") or "").strip()
        score = meta.get("score")
        try:
            score = float(score) if score is not None else None
        except Exception:
            score = None
        return CaseView(
            id=getattr(row, "id", None),
            goal_text=goal_text,
            summary=summary,
            lessons=lessons,
            winner_rationale=winner_rationale,
            score=score,
            casebook_name=casebook_name,
        )

    def _from_embedding_hit(self, hit: Dict[str, Any]) -> CaseView:
        meta = hit.get("metadata", {}) or hit.get("meta", {}) or {}
        scores = meta.get("scores", {}) or {}
        return CaseView(
            id=hit.get("id", meta.get("id")),
            goal_text=meta.get("goal_text", meta.get("title", "")) or "",
            summary=meta.get("summary", meta.get("text", "")) or "",
            lessons=meta.get("lessons", "") or "",
            winner_rationale=scores.get("winner_rationale", meta.get("winner_rationale", "")) or "",
            score=hit.get("score", meta.get("score")),
            casebook_name=meta.get("casebook_name"),
        )
