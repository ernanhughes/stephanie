# stephanie/agents/learning/knowledge_arena.py
from __future__ import annotations

import asyncio
import logging
import math
import time
import uuid
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from stephanie.utils.emit_utils import prepare_emit

Logger = logging.Logger
Score = Dict[str, float]
Candidate = Dict[str, Any]
EmitFn = Optional[Callable[[Dict[str, Any]], Awaitable[None] | None]]

_logger = logging.getLogger(__name__)

def _is_coro_fn(fn: Callable) -> bool:
    return asyncio.iscoroutinefunction(fn)

def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        if hasattr(v, "item"):  # numpy scalar
            v = v.item()
        f = float(v)
        return 0.0 if math.isnan(f) or math.isinf(f) else f
    except Exception:
        return default

def _to_bool(v: Any) -> bool:
    try:
        if hasattr(v, "item"):
            v = v.item()
        return bool(v)
    except Exception:
        return False

def _sanitize_payload(d: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        if hasattr(v, "item"):
            v = v.item()
        if isinstance(v, float):
            if math.isnan(v) or math.isinf(v):
                v = None
        out[k] = v
    return out

def _norm_score(s: Optional[Score]) -> Score:
    s = s or {}
    return {
        "k": _to_float(s.get("k")),
        "c": _to_float(s.get("c")),
        "g": _to_float(s.get("g")),
        "overall": _to_float(s.get("overall")),
        "verified": _to_bool(s.get("verified")),
    }

class KnowledgeArena:
    """
    “to the best of my knowledge” — run self-play rounds to select the best candidate.

    Responsibilities:
      - Score an initial pool -> keep top-K (beam)
      - Iteratively improve and re-score candidates
      - Early-stop on plateau or low marginal reward per kTok
      - Emit structured lifecycle events (caller persists as needed)

    Injected hooks (sync or async):
      - score_candidate(text: str, section_text: str) -> Score
      - improve(text: str, improve_ctx: Dict[str, Any]) -> str
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        memory: Any,
        container: Any,
        logger: Optional[Logger],
        *,
        token_estimator: Optional[Callable[[str], int]] = None,
    ):
        self.cfg = cfg
        self.memory = memory
        self.container = container
        self.logger = logger or _logger

        # config with sensible defaults
        self._beam_w = max(1, int(cfg.get("beam_width", 5)))
        self._max_rounds = max(0, int(cfg.get("self_play_rounds", 2)))
        self._plateau_eps = max(0.0, float(cfg.get("self_play_plateau_eps", 0.005)))
        self._min_marg = max(0.0, float(cfg.get("min_marginal_reward_per_ktok", 0.05)))
        self._enable_diversity_guard = bool(cfg.get("enable_diversity_guard", True))
        # how to pick a diversity replacement: "last" (replace worst), "closest" (replace closest duplicate), "none"
        self._diversity_mode = str(cfg.get("diversity_mode", "last")).lower()
        # parallelism for scoring/improving
        self._max_parallel = max(1, int(cfg.get("max_parallel", 8)))
        self._sem = asyncio.Semaphore(self._max_parallel)

        self._tok = token_estimator or (lambda t: max(1, int(len(t or "") / 4)))

        # hook signatures (caller must override these)
        # sync or async are both supported — we detect and await if needed.
        self.score_candidate: Callable[[str, str], Score] = self._must_override_score
        self.improve: Callable[[str, Dict[str, Any]], str] = self._must_override_improve

    # ---- abstract defaults (raise helpful errors if not set) ----
    def _must_override_score(self, *_a, **_k) -> Score:
        raise NotImplementedError("KnowledgeArena.score_candidate must be provided by the caller.")

    def _must_override_improve(self, *_a, **_k) -> str:
        raise NotImplementedError("KnowledgeArena.improve must be provided by the caller.")

    # ---- unified call wrappers (sync/async transparent) ----
    async def _call_score(self, text: str, section_text: str) -> Score:
        try:
            if _is_coro_fn(self.score_candidate):
                s = await self.score_candidate(text, section_text)
            else:
                s = self.score_candidate(text, section_text)
            return _norm_score(s)
        except Exception as e:
            self.logger.warning("Arena.score_candidate failed; zeroing score: %s", e)
            return _norm_score(None)

    async def _call_improve(self, text: str, improve_ctx: Dict[str, Any]) -> str:
        try:
            if _is_coro_fn(self.improve):
                out = await self.improve(text, improve_ctx)
            else:
                out = self.improve(text, improve_ctx)
            return out if isinstance(out, str) and out else text
        except Exception as e:
            self.logger.warning("Arena.improve failed; keeping original: %s", e)
            return text

    # ---- emitter that supports fn or events object ----
    async def _emit(self, emit: EmitFn | Any, payload: Dict[str, Any], *, method: Optional[str] = None) -> None:
        if not emit:
            return
        try:
            safe = _sanitize_payload(payload)
            # If an events object was provided (with named method), call it; else call emit(safe).
            if method and hasattr(emit, method):
                fn = getattr(emit, method)
                if asyncio.iscoroutinefunction(fn):
                    await fn(safe)
                else:
                    fn(safe)
            else:
                if asyncio.iscoroutinefunction(emit):
                    await emit(safe)
                else:
                    emit(safe)
        except Exception as e:
            # never fail the arena for telemetry issues
            self.logger.debug("Arena emit skipped: %s", e)

    # ---- helpers ----
    def _marginal_per_ktok(self, prev_best: float, curr_best: float, prev_toks: int, curr_toks: int) -> float:
        dr, dt = (curr_best - prev_best), max(1, curr_toks - prev_toks)
        return (dr / dt) * 1000.0

    def _cfg_snapshot(self) -> Dict[str, Any]:
        return {
            "beam_width": self._beam_w,
            "self_play_rounds": self._max_rounds,
            "self_play_plateau_eps": self._plateau_eps,
            "min_marginal_reward_per_ktok": self._min_marg,
            "enable_diversity_guard": self._enable_diversity_guard,
            "diversity_mode": self._diversity_mode,
            "max_parallel": self._max_parallel,
        }

    # ---- diversity guard ----
    def _apply_diversity_guard(self, new_beam: List[Candidate], scored_pool: List[Candidate]) -> Tuple[List[Candidate], bool]:
        if not self._enable_diversity_guard or not new_beam:
            return new_beam, False

        origins = [b.get("origin") for b in new_beam]
        unique = set(o for o in origins if o is not None)
        if len(unique) > 1:
            return new_beam, False

        # find an alternative candidate with a different origin
        alt = next((c for c in scored_pool if c.get("origin") not in unique), None)
        if not alt:
            return new_beam, False

        replaced = False
        if self._diversity_mode == "closest":
            # replace the item whose score is closest to the leader (preserves tail diversity)
            lead = _to_float(new_beam[0].get("score", {}).get("overall"))
            idx, _ = min(
                enumerate(new_beam),
                key=lambda kv: abs(_to_float(kv[1].get("score", {}).get("overall")) - lead),
            )
            new_beam[idx] = alt
            replaced = True
        else:
            # default: replace the last item (worst)
            new_beam[-1] = alt
            replaced = True

        return new_beam, replaced

    # ---- concurrent map utility ----
    async def _bounded_gather(self, coros: List[Callable[[], Awaitable[Any]]]) -> List[Any]:
        async def _run(coro_factory: Callable[[], Awaitable[Any]]):
            async with self._sem:
                return await coro_factory()
        return await asyncio.gather(*[_run(cf) for cf in coros], return_exceptions=False)

    # ---- main API ----
    async def run(
        self,
        section_text: str,
        initial_candidates: List[Candidate],
        context: Optional[Dict[str, Any]],
        *,
        emit: EmitFn | Any = None,  # callable OR events object with .started/.round_start/.round_end/.done
        run_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        run_id = context.get("pipeline_run_id")
        started_at = time.time()
        cfg_snap = self._cfg_snapshot()

        await self._emit(
            emit,
            prepare_emit("arena_start", {"run_id": run_id, "t": started_at, **(run_meta or {}), **cfg_snap}),
            method="started",
        )

        # ---- empty guard ----
        if not initial_candidates:
            empty = {"text": "", "score": _norm_score(None), "origin": "empty", "variant": "v0"}
            out = {
                "winner": empty, "beam": [empty], "initial_pool": [],
                "iterations": [], "rounds_run": 0,
                "best_history": [], "marginal_history": [],
                "stop_reason": "no_candidates",
                "arena_run_id": run_id, "started_at": started_at, "ended_at": time.time(),
                "summary": {"winner_overall": 0.0, "rounds_run": 0, "reason": "no_candidates"},
            }
            await self._emit(emit, {"event": "arena_stop", "run_id": run_id, "reason": "no_candidates", "winner_overall": 0.0, "rounds_run": 0}, method="round_end")
            await self._emit(emit, {"event": "arena_done", "run_id": run_id, "winner_overall": 0.0}, method="done")
            return out

        # ---- initial scoring (parallel) ----
        score_jobs = [
            (c, lambda txt=c.get("text", ""): self._call_score(txt, section_text))
            for c in initial_candidates
        ]
        scored: List[Candidate] = []
        results = await self._bounded_gather([job for _, job in score_jobs])
        for (c, _), s in zip(score_jobs, results):
            scored.append({**c, "score": s})
        scored.sort(
            key=lambda x: (
                _to_bool(x.get("score", {}).get("verified")),
                _to_float(x.get("score", {}).get("overall")),
                len(x.get("text", "") or ""),
            ),
            reverse=True,
        )

        # top-k preview for dashboards
        topk_preview = [
            {
                "origin": sc.get("origin"),
                "variant": sc.get("variant"),
                "overall": _to_float(sc.get("score", {}).get("overall")),
                "k": _to_float(sc.get("score", {}).get("k")),
                "verified": _to_bool(sc.get("score", {}).get("verified")),
            }
            for sc in scored[: min(5, len(scored))]
        ]
        await self._emit(
            emit,
            prepare_emit("initial_scored", {"run_id": run_id, "topk": topk_preview}),
            method="round_start",
        )

        beam = scored[: self._beam_w]
        iters: List[List[Dict[str, Any]]] = []
        best_history: List[float] = []
        marginal_history: List[float] = []
        stop_reason = "max_rounds"

        prev_best = _to_float(beam[0]["score"]["overall"]) if beam else 0.0
        prev_toks = self._tok(beam[0]["text"]) if beam else 1
        rounds_run = 0

        for r in range(self._max_rounds):
            rounds_run = r + 1
            await self._emit(
                emit,
                prepare_emit("round_begin", {"run_id": run_id, "round": rounds_run, "prev_best": float(prev_best)}),
                method="round_start",
            )

            # ---- improve & score (parallel, bounded) ----
            improve_jobs = []
            for cand in beam:
                meta = {**(cand.get("meta") or {}), "round": r}
                improve_jobs.append(lambda c=cand, m=meta: self._call_improve(c.get("text", "") or "", m))
            improved_texts: List[str] = await self._bounded_gather(improve_jobs)

            score_jobs = [
                (cand, txt, lambda t=txt: self._call_score(t, section_text))
                for cand, txt in zip(beam, improved_texts)
            ]
            scored_improved: List[Candidate] = []
            score_results = await self._bounded_gather([job for _, _, job in score_jobs])
            for (cand, txt, _), s in zip(score_jobs, score_results):
                scored_improved.append({
                    **cand,
                    "variant": f"{cand.get('variant', 'v')}+r{rounds_run}",
                    "text": txt,
                    "score": s
                })

            scored_improved.sort(
                key=lambda x: (
                    _to_bool(x.get("score", {}).get("verified")),
                    _to_float(x.get("score", {}).get("overall")),
                    len(x.get("text", "") or ""),
                ),
                reverse=True,
            )

            # ---- diversity guard ----
            replaced = False
            scored_improved, replaced = self._apply_diversity_guard(scored_improved, scored)

            curr_best = _to_float(scored_improved[0]["score"]["overall"]) if scored_improved else prev_best
            curr_toks = self._tok(scored_improved[0]["text"]) if scored_improved else prev_toks
            marg = self._marginal_per_ktok(prev_best, curr_best, prev_toks, curr_toks)

            marginal_history.append(float(marg))
            best_history.append(float(curr_best))
            iters.append(
                [
                    {
                        "variant": b.get("variant"),
                        "overall": _to_float(b.get("score", {}).get("overall")),
                        "k": _to_float(b.get("score", {}).get("k")),
                        "verified": _to_bool(b.get("score", {}).get("verified")),
                    }
                    for b in scored_improved
                ]
            )

            await self._emit(
                emit,
                prepare_emit(
                    "round_end",
                    {
                        "run_id": run_id,
                        "round": rounds_run,
                        "best_overall": float(curr_best),
                        "marginal_per_ktok": float(marg),
                        "diversity_replaced": bool(replaced),
                    },
                ),
                method="round_end",
            )

            # ---- early stop checks ----
            if marg < self._min_marg:
                stop_reason = "low_marginal_reward"
                break
            if len(best_history) >= 2 and (best_history[-1] - best_history[-2]) < self._plateau_eps:
                stop_reason = "plateau"
                break

            beam, prev_best, prev_toks = (scored_improved[: self._beam_w], curr_best, curr_toks)

        winner = (beam or scored or [{"text": "", "score": _norm_score(None)}])[0]

        await self._emit(
            emit,
            prepare_emit(
                "arena_stop",
                {
                    "run_id": run_id,
                    "reason": stop_reason,
                    "winner_overall": _to_float(winner.get("score", {}).get("overall")),
                    "rounds_run": int(rounds_run),
                },
            ),
            method="round_end",
        )

        out = {
            "winner": winner,
            "beam": beam,
            "initial_pool": scored,
            "iterations": iters,
            "rounds_run": rounds_run,
            "best_history": best_history,
            "marginal_history": marginal_history,
            "stop_reason": stop_reason,
            "arena_run_id": run_id,
            "started_at": started_at,
            "ended_at": time.time(),
            # compact summary for dashboards
            "summary": {
                "winner_overall": _to_float(winner.get("score", {}).get("overall")),
                "rounds_run": int(rounds_run),
                "reason": stop_reason,
            },
        }

        await self._emit(
            emit,
            prepare_emit(
                "arena_done",
                {"run_id": run_id, "ended_at": out["ended_at"], "summary": out["summary"]},
            ),
            method="done",
        )
        return out
