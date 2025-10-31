# stephanie/components/arena/knowledge_arena.py
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable, Set

from stephanie.components.arena.scoring.aggregate import WeightedAggregator
from stephanie.components.arena.plugins.registry import list_scorers


EmitFn = Callable[[Dict[str, Any]], Any]  # accepts event dict; may be sync or async


@dataclass
class ArenaConfig:
    """Light wrapper over cfg for the arena bits you care about."""
    improve_iters: int = 0                 # optional local refine loop
    topk_to_improve: int = 1               # improve top-K candidates (usually 1)
    enabled_scorers: Optional[List[str]] = None
    scorer_weights: Optional[Dict[str, float]] = None
    min_text_len: int = 60                 # drop tiny candidates
    max_candidates: int = 32               # guardrail
    use_plugin_scorers: bool = False       # if True, use plugin scorers + aggregator


def _is_coro(x):  # tiny utility to unify sync/async callables
    return asyncio.iscoroutinefunction(x)


async def _maybe_await(fn: Callable, *a, **kw):
    if _is_coro(fn):
        return await fn(*a, **kw)
    return fn(*a, **kw)


async def _emit(emit: Optional[Any], name: str, payload: Dict[str, Any]):
    if not emit:
        return
    evt = {"event": name, **payload}
    # Accept: callable, object with .emit(), or ArenaReporter instance
    if callable(emit):
        return await _maybe_await(emit, evt)
    if hasattr(emit, "emit"):
        return await _maybe_await(emit.emit, evt)
    # worst case: ignore silently


class KnowledgeArena:
    """
    General-purpose candidate arena that mirrors your agent flow:
    - scores a pool using injected score_candidate
    - (optionally) runs a tiny local improve loop via injected improve
    - emits events compatible with ArenaReporter
    """

    def __init__(self, cfg: Dict[str, Any], memory, container, logger):
        self.cfg_raw = dict(cfg or {})
        self.memory = memory
        self.container = container
        self.log = logger

        # External hooks (wired by the agent)
        self.score_candidate: Optional[
            Callable[[str, str, Dict[str, Any]], Any]
        ] = None  # async/ sync accepted
        self.improve: Optional[
            Callable[[str, Dict[str, Any]], Any]
        ] = None  # async/ sync accepted; signature(text, meta) -> improved_text

        # Parsed config
        self.cfg = ArenaConfig(
            improve_iters=int(self.cfg_raw.get("arena_improve_iters", 0)),
            topk_to_improve=int(self.cfg_raw.get("arena_topk_to_improve", 1)),
            enabled_scorers=list(self.cfg_raw.get("arena_scorers", []) or []),
            scorer_weights=dict(self.cfg_raw.get("arena_scorer_weights", {}) or {}),
            min_text_len=int(self.cfg_raw.get("arena_min_text_len", 60)),
            max_candidates=int(self.cfg_raw.get("arena_max_candidates", 32)),
            use_plugin_scorers=bool(self.cfg_raw.get("arena_use_plugin_scorers", False)),
        )

        # Optional plugin scorer pipeline
        self._scorers = {}
        self._aggregator: Optional[WeightedAggregator] = None
        if self.cfg.use_plugin_scorers and self.cfg.enabled_scorers:
            all_sc = list_scorers()
            self._scorers = {n: all_sc[n] for n in self.cfg.enabled_scorers if n in all_sc}
            self._aggregator = WeightedAggregator(self.cfg.scorer_weights or {})

    # ---------------------------------------------------------------------
    # Public: prepare candidates (so you can remove _build_candidates from agent)
    # ---------------------------------------------------------------------
    def prepare_candidates(
        self,
        section: Dict[str, Any],
        corpus_items: List[Dict[str, Any]],
        *,
        mask_keys: Optional[Set[str]] = None,
        seed_len: int = 800,
        max_corpus_candidates: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        mask_keys = mask_keys or set()
        max_k = int(max_corpus_candidates or self.cfg_raw.get("max_corpus_candidates", 8))

        out: List[Dict[str, Any]] = []

        # corpus-backed
        for it in (corpus_items or [])[:max_k]:
            t = (it.get("assistant_text") or "").strip()
            if len(t) < self.cfg.min_text_len:
                continue
            corpus_k = f"corpus:{str(it.get('id'))}"
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

        # seed from section
        seed = (section.get("section_text") or "").strip()[:seed_len]
        if len(seed) >= self.cfg.min_text_len and "arena:lfl_seed#seed" not in mask_keys:
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

        # dedupe by normalized text
        seen = set()
        uniq = []
        for c in out:
            key = " ".join((c.get("text") or "").split()).lower()
            if key in seen:
                continue
            seen.add(key)
            uniq.append(c)

        return uniq[: self.cfg.max_candidates]

    # ---------------------------------------------------------------------
    # Internal: scoring helpers
    # ---------------------------------------------------------------------
    async def _score_one(
        self,
        problem_text: str,
        cand: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, float]:
        if not self.score_candidate:
            raise RuntimeError("KnowledgeArena.score_candidate is not set")
        # Expect your Scoring.score_candidate(...) to return metrics including 'overall'
        metrics = await _maybe_await(self.score_candidate, problem_text, cand["text"], context)
        # Optionally route metrics through plugin scorers -> aggregator
        if self._scorers and self._aggregator:
            per = {name: sc.score(context, metrics) for name, sc in self._scorers.items()}
            reward = float(self._aggregator.aggregate(per))
            metrics = dict(metrics or {})
            metrics["overall"] = reward  # align to agent's usage of .get("overall")
        return {k: float(v) for k, v in (metrics or {}).items()}

    @staticmethod
    def _best_idx(scores: List[float]) -> int:
        if not scores:
            return -1
        best, best_i = -1e9, -1
        for i, s in enumerate(scores):
            if s > best:
                best, best_i = s, i
        return best_i

    # ---------------------------------------------------------------------
    # Public: main run
    # ---------------------------------------------------------------------
    async def run(
        self,
        problem_text: str,
        candidates: List[Dict[str, Any]],
        *,
        emit: Optional[Any] = None,
        run_meta: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Returns:
          {
            "winner": {
                "text": ...,
                "origin": ..., "variant": ...,
                "score": {"overall": x, ...},
            },
            "initial_pool": [cands...],
            "scored_pool": [{"...","score":{...}}, ...],
            "iterations": [{"improved_text": "...", "score": {...}}, ...]  # if improve used
          }
        """
        ctx = dict(context or {})
        await _emit(emit, "arena:start", {"meta": run_meta or {}, "pool_size": len(candidates)})

        # 1) score initial pool
        pool = []
        for idx, c in enumerate(candidates):
            t = (c.get("text") or "").strip()
            if len(t) < self.cfg.min_text_len:
                continue
            m = await self._score_one(problem_text, c, ctx)
            item = {**c, "score": m}
            pool.append(item)
            await _emit(emit, "arena:candidate_scored", {"idx": idx, "score": m, "origin": c.get("origin"), "variant": c.get("variant")})

        if not pool:
            await _emit(emit, "arena:empty_pool", {"meta": run_meta or {}})
            return {"winner": {"text": "", "origin": "none", "variant": "", "score": {"overall": 0.0}}, "initial_pool": list(candidates or []), "scored_pool": []}

        scores = [float(p.get("score", {}).get("overall", 0.0)) for p in pool]
        best_i = self._best_idx(scores)
        winner = dict(pool[best_i])
        await _emit(emit, "arena:winner_initial", {"score": winner["score"], "origin": winner.get("origin"), "variant": winner.get("variant")})

        # 2) optional local improve loop (uses injected self.improve)
        iterations: List[Dict[str, Any]] = []
        if self.improve and self.cfg.improve_iters > 0:
            # choose top-K to refine (default 1: the winner)
            topk = sorted(range(len(pool)), key=lambda i: float(pool[i]["score"].get("overall", 0.0)), reverse=True)[: max(1, self.cfg.topk_to_improve)]
            for step in range(self.cfg.improve_iters):
                for i in topk:
                    base = pool[i]
                    meta = (base.get("meta") or {}) | {"support_origin": base.get("origin"), "support_variant": base.get("variant")}
                    improved_text = await _maybe_await(self.improve, base.get("text", ""), meta)
                    # Re-score improved text
                    tmp_c = {"origin": "improve", "variant": f"i{step}_k{i}", "text": improved_text, "meta": meta}
                    m2 = await self._score_one(problem_text, tmp_c, ctx)
                    iterations.append({"improved_text": improved_text, "score": m2})
                    await _emit(emit, "arena:improve_scored", {"step": step, "k_idx": i, "score": m2})

                    # keep improvement if better
                    if m2.get("overall", -1e9) > float(base["score"].get("overall", -1e9)):
                        pool[i] = {**tmp_c, "score": m2}
                # refresh winner
                scores = [float(p.get("score", {}).get("overall", 0.0)) for p in pool]
                best_i = self._best_idx(scores)
                winner = dict(pool[best_i])
                await _emit(emit, "arena:winner_update", {"step": step, "score": winner["score"]})

        await _emit(emit, "arena:done", {"winner_score": winner["score"]})
        return {
            "winner": {"text": winner.get("text", ""), "origin": winner.get("origin"), "variant": winner.get("variant"), "score": dict(winner.get("score") or {})},
            "initial_pool": list(candidates or []),
            "scored_pool": pool,
            "iterations": iterations,
        }
