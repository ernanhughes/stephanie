# stephanie/services/cbr_service.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from stephanie.services.service_protocol import Service


# ---------- small utils ----------
def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def _safe_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    try:
        return str(x)
    except Exception:
        return ""


@dataclass
class _CaseView:
    """Uniform view over different case/scorable shapes."""
    id: Any
    goal_text: str
    summary: str
    lessons: str
    winner_rationale: str
    raw: Any
    score: Optional[float] = None   # similarity score if provided by retriever


class CBRService(Service):
    """
    Case-based retrieval that prefers the *dense retriever* path:

        memory.embedding.search_related_scorables(
            goal_text, top_k=pool_k, include_ner=..., target_type=...
        )

    Falls back to:
      1) store-native search_* methods if present
      2) full scan + cosine similarity

    Output items are normalized so your agent’s `_retrieve_case_pack` can do:

        for c in cbr.retrieve(...):
            pack.append({
               "title": c["goal_text"],
               "why_it_won": c["scores"]["winner_rationale"],
               "patch": c["lessons"],
            })
    """

    def __init__(self, cfg, memory, container, logger):
        self.cfg = cfg or {}
        self.memory = memory
        self.container = container
        self.logger = logger
        self._initialized = False

        # deps
        self.cb = getattr(memory, "casebooks", None)
        self.embed = getattr(memory, "embedding", None)

        # knobs
        c = self.cfg.get("cbr", {}) or {}
        self.pool_k = int(c.get("pool_k", 64))
        self.include_ner = bool(c.get("include_ner", False))
        self.target_type = c.get("target_type", "case")  # str | list[str] | None
        self.max_pool = int(c.get("max_pool", 10000))
        self.min_text_len = int(c.get("min_text_len", 8))

    # --- Service protocol ---
    def initialize(self, **kwargs) -> None:
        self._initialized = True
        try:
            self.logger and self.logger.log("CBRServiceInit", {"status": "ok", "mode": "dense-first"})
        except Exception:
            pass

    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy" if self._initialized else "uninitialized",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "metrics": {},
            "dependencies": {
                "casebooks": bool(self.cb),
                "embedding": bool(self.embed),
                "dense_search": bool(self._has_dense_api()),
            },
        }

    def shutdown(self) -> None:
        self._initialized = False

    @property
    def name(self) -> str:
        return "cbr-service-v2-dense"

    # --- Public API used by your agent ---
    def retrieve(self, *, goal_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        goal_text = (_safe_text(goal_text) or "").strip()
        if not goal_text:
            return []

        # 1) DENSE FIRST: embedding.search_related_scorables
        if self._has_dense_api():
            try:
                dense = self._dense_search(goal_text, pool_k=max(top_k, self.pool_k))
                if dense:
                    return self._pack_for_agent(self._truncate(dense, top_k))
            except Exception as e:
                self._log("DenseSearchError", {"error": str(e)})

        # 2) store-native search_* if exposed
        native = self._find_cases_native(goal_text)
        if native:
            ranked = self._rank_by_similarity(goal_text, native)
            return self._pack_for_agent(self._truncate(ranked, top_k))

        # 3) fallback: full scan + cosine
        scanned = self._find_cases_fallback(goal_text)
        ranked = self._rank_by_similarity(goal_text, scanned)
        return self._pack_for_agent(self._truncate(ranked, top_k))

    # ---------- dense retriever path ----------
    def _has_dense_api(self) -> bool:
        return bool(self.embed and hasattr(self.embed, "search_related_scorables"))

    def _dense_search(self, goal_text: str, pool_k: int) -> List[_CaseView]:
        """
        Calls memory.embedding.search_related_scorables and normalizes results.
        We accept any scorable target_type; restrict via cfg["cbr"]["target_type"] if desired.
        """
        hits = self.embed.search_related_scorables(
            goal_text,
            top_k=int(pool_k),
            include_ner=self.include_ner,
            target_type=self.target_type,  # may be None, str, or list[str] in your implementation
        ) or []

        views: List[_CaseView] = []
        for h in hits:
            cv = self._from_scorable(h)
            if len(cv.goal_text) >= self.min_text_len or len(cv.summary) >= self.min_text_len:
                views.append(cv)
        # If retriever supplied "score", preserve ordering; otherwise we’ll re-rank later if needed
        return views

    def _from_scorable(self, s: Any) -> _CaseView:
        """Normalize a scorable or dict from the dense retriever into a _CaseView."""
        if isinstance(s, dict):
            meta = s.get("metadata") or s.get("meta") or {}
            score = s.get("score") or meta.get("score")
            return _CaseView(
                id=s.get("id"),
                goal_text=_safe_text(meta.get("goal_text") or meta.get("title") or s.get("title")),
                summary=_safe_text(s.get("text") or meta.get("text") or meta.get("summary")),
                lessons=_safe_text(meta.get("lessons")),
                winner_rationale=_safe_text((meta.get("scores") or {}).get("winner_rationale") or meta.get("winner_rationale")),
                raw=s,
                score=float(score) if score is not None else None,
            )

        # object-like
        meta = getattr(s, "metadata", None) or getattr(s, "meta", None) or {}
        score = getattr(s, "score", None) or meta.get("score")
        return _CaseView(
            id=getattr(s, "id", None),
            goal_text=_safe_text(meta.get("goal_text") or meta.get("title") or getattr(s, "title", None)),
            summary=_safe_text(getattr(s, "text", None) or meta.get("text") or meta.get("summary")),
            lessons=_safe_text(meta.get("lessons")),
            winner_rationale=_safe_text((meta.get("scores") or {}).get("winner_rationale") or meta.get("winner_rationale")),
            raw=s,
            score=float(score) if score is not None else None,
        )

    # ---------- native store path ----------
    def _find_cases_native(self, goal_text: str) -> List[_CaseView]:
        if not self.cb:
            return []
        for name in ("search_cases_for_goal_text", "search_cases_by_goal", "search_cases", "find_cases_by_text", "query_cases"):
            fn = getattr(self.cb, name, None)
            if callable(fn):
                try:
                    raw = fn(goal_text=goal_text, limit=self.max_pool)
                except TypeError:
                    try:
                        raw = fn(goal_text)  # positional
                    except Exception:
                        continue
                return [self._to_view(c) for c in self._as_iter(raw)]
        return []

    # ---------- fallback full-scan path ----------
    def _find_cases_fallback(self, goal_text: str) -> List[_CaseView]:
        if not self.cb:
            return []
        views: List[_CaseView] = []
        for case in self._iter_all_cases(self.cb):
            cv = self._to_view(case)
            if len(cv.goal_text) >= self.min_text_len:
                views.append(cv)
            if len(views) >= self.max_pool:
                break
        return views

    # ---------- ranking ----------
    def _rank_by_similarity(self, goal_text: str, cases: List[_CaseView]) -> List[_CaseView]:
        if not cases:
            return []
        # If dense path gave us scores, use them
        if any(cv.score is not None for cv in cases):
            return sorted(cases, key=lambda cv: (cv.score is None, -(cv.score or 0.0)))
        # Else use embeddings for cosine
        if self.embed:
            try:
                qv = np.asarray(self.embed.get_or_create(goal_text), dtype=float)
                scored: List[Tuple[float, _CaseView]] = []
                for cv in cases:
                    base = cv.goal_text or cv.summary
                    vv = np.asarray(self.embed.get_or_create(base), dtype=float)
                    scored.append((_cos(qv, vv), cv))
                scored.sort(key=lambda t: t[0], reverse=True)
                return [cv for s, cv in scored]
            except Exception:
                pass
        # Fallback trivial
        return sorted(cases, key=lambda cv: len(cv.goal_text + " " + cv.summary), reverse=True)

    # ---------- normalization helpers ----------
    def _to_view(self, case_obj: Any) -> _CaseView:
        if isinstance(case_obj, dict):
            meta = case_obj.get("meta", {}) or {}
            scores = case_obj.get("scores", {}) or meta.get("scores", {}) or {}
            return _CaseView(
                id=case_obj.get("id"),
                goal_text=_safe_text(case_obj.get("goal_text") or meta.get("goal_text")),
                summary=_safe_text(meta.get("summary") or meta.get("text") or case_obj.get("text")),
                lessons=_safe_text(case_obj.get("lessons") or meta.get("lessons")),
                winner_rationale=_safe_text(scores.get("winner_rationale") or meta.get("winner_rationale")),
                raw=case_obj,
            )
        meta = getattr(case_obj, "meta", None) or {}
        scores = getattr(case_obj, "scores", None) or meta.get("scores", {}) or {}
        return _CaseView(
            id=getattr(case_obj, "id", None),
            goal_text=_safe_text(getattr(case_obj, "goal_text", None) or meta.get("goal_text")),
            summary=_safe_text(meta.get("summary") or meta.get("text") or getattr(case_obj, "text", None)),
            lessons=_safe_text(getattr(case_obj, "lessons", None) or meta.get("lessons")),
            winner_rationale=_safe_text(getattr(scores, "winner_rationale", None) if hasattr(scores, "winner_rationale") else scores.get("winner_rationale")),
            raw=case_obj,
        )

    def _pack_for_agent(self, views: List[_CaseView]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for cv in views:
            out.append({
                "id": cv.id,
                "goal_text": cv.goal_text or cv.summary,  # ensure something usable
                "scores": {"winner_rationale": cv.winner_rationale},
                "lessons": cv.lessons,
                "text": cv.summary,
            })
        return out

    # generic iteration helpers
    def _as_iter(self, maybe_iter: Any) -> Iterable[Any]:
        if maybe_iter is None:
            return []
        if isinstance(maybe_iter, (list, tuple, set)):
            return maybe_iter
        try:
            return list(maybe_iter)
        except Exception:
            return [maybe_iter]

    def _iter_all_cases(self, store: Any) -> Iterable[Any]:
        for book in self._iter_casebooks(store):
            yield from self._iter_cases_in_book(book)
        for m in ("list_cases", "all_cases", "iter_cases", "get_all_cases"):
            fn = getattr(store, m, None)
            if callable(fn):
                try:
                    for c in self._as_iter(fn()):
                        yield c
                    return
                except Exception:
                    continue

    def _iter_casebooks(self, store: Any) -> Iterable[Any]:
        for m in ("list_casebooks", "all_casebooks", "iter_casebooks", "get_all_casebooks"):
            fn = getattr(store, m, None)
            if callable(fn):
                try:
                    yield from self._as_iter(fn())
                    return
                except Exception:
                    continue
        for attr in ("casebooks", "books", "all", "items"):
            coll = getattr(store, attr, None)
            if coll is not None:
                try:
                    for b in coll:
                        yield b
                    return
                except Exception:
                    continue

    def _iter_cases_in_book(self, book: Any) -> Iterable[Any]:
        for m in ("list_cases", "all_cases", "iter_cases", "get_cases"):
            fn = getattr(book, m, None)
            if callable(fn):
                try:
                    yield from self._as_iter(fn())
                    return
                except Exception:
                    continue
        for attr in ("cases", "items"):
            coll = getattr(book, attr, None)
            if coll is not None:
                try:
                    for c in coll:
                        yield c
                    return
                except Exception:
                    continue

    @staticmethod
    def _truncate(items: List[_CaseView], k: int) -> List[_CaseView]:
        k = max(1, int(k))
        return items[:k]

    def _log(self, event: str, payload: Dict[str, Any]):
        try:
            self.logger and self.logger.log(event, payload)
        except Exception:
            pass
