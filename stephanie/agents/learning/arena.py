# stephanie/agents/learning/arena_service.py
from __future__ import annotations
from typing import Any, Dict, List, Callable, Optional
import logging

_logger = logging.getLogger(__name__)

Score = Dict[str, float]
Candidate = Dict[str, Any]


class ArenaService:
    """
    Self-play tournament: rank -> improve -> re-rank across rounds.
    No DB side effects; caller persists.
    """

    def __init__(
        self,
        cfg,
        memory,
        container,
        logger,
        *,
        token_estimator: Optional[Callable[[str], int]] = None,
    ):
        self.cfg = cfg
        self.memory = memory
        self.container = container
        self.logger = logger

        # cache config with sane defaults
        self._beam_w = max(1, int(cfg.get("beam_width", 5)))
        self._max_rounds = max(0, int(cfg.get("self_play_rounds", 2)))
        self._plateau_eps = max(
            0.0, float(cfg.get("self_play_plateau_eps", 0.005))
        )
        self._min_marg = max(
            0.0, float(cfg.get("min_marginal_reward_per_ktok", 0.05))
        )
        self._enable_diversity_guard = bool(
            cfg.get("enable_diversity_guard", True)
        )

        self._tok = token_estimator or (
            lambda t: max(1, int(len(t or "") / 4))
        )

    # ---- external scoring hooks provided by caller ----
    def score_candidate(self, text: str, section_text: str) -> Score:
        raise NotImplementedError  # caller injects

    def improve(self, cand_text: str, improve_context: Dict[str, Any]) -> str:
        raise NotImplementedError  # caller injects

    # ---- helpers ----
    @staticmethod
    def _norm_score(s: Optional[Score]) -> Score:
        s = s or {}
        return {
            "k": float(s.get("k", 0.0)),
            "c": float(s.get("c", 0.0)),
            "g": float(s.get("g", 0.0)),
            "overall": float(s.get("overall", 0.0)),
            "verified": bool(s.get("verified", False)),
        }

    @staticmethod
    def _safe_improve(call: Callable[[], str], fallback: str) -> str:
        try:
            out = call()
            return out if isinstance(out, str) and out else fallback
        except Exception as e:
            _logger.warning(f"Arena.improve failed; keeping original: {e}")
            return fallback

    @staticmethod
    def _safe_score(call: Callable[[], Score]) -> Score:
        try:
            return ArenaService._norm_score(call())
        except Exception as e:
            _logger.warning(
                f"Arena.score_candidate failed; zeroing score: {e}"
            )
            return ArenaService._norm_score(None)

    # ---- main API ----
    def run(
        self, section_text: str, initial_candidates: List[Candidate]
    ) -> Dict[str, Any]:
        # Empty/degenerate guard
        if not initial_candidates:
            empty = {
                "text": "",
                "score": self._norm_score(None),
                "origin": "empty",
                "variant": "v0",
            }
            return {
                "winner": empty,
                "beam": [empty],
                "initial_pool": [],
                "iterations": [],
                "rounds_run": 0,
                "best_history": [],
                "marginal_history": [],
                "stop_reason": "no_candidates",
            }

        # 1) initial scoring
        scored: List[Candidate] = []
        for c in initial_candidates:
            s = self._safe_score(
                lambda: self.score_candidate(c.get("text", ""), section_text)
            )
            scored.append({**c, "score": s})

        # stable sort: verified desc, overall desc, then text length desc
        scored.sort(
            key=lambda x: (
                bool(x.get("score", {}).get("verified", False)),
                float(x.get("score", {}).get("overall", 0.0)),
                len(x.get("text", "") or ""),
            ),
            reverse=True,
        )

        beam = scored[: self._beam_w]
        iters: List[List[Dict[str, Any]]] = []
        best_history: List[float] = []
        marginal_history: List[float] = []
        stop_reason = "max_rounds"

        # seed baseline for marginal calc
        prev_best = beam[0]["score"]["overall"] if beam else 0.0
        prev_toks = self._tok(beam[0]["text"]) if beam else 1

        def _marginal(
            prev_best: float, curr_best: float, prev_toks: int, curr_toks: int
        ) -> float:
            dr = curr_best - prev_best
            dt = max(1, curr_toks - prev_toks)
            return (dr / dt) * 1000.0

        rounds_run = 0
        for r in range(self._max_rounds):
            rounds_run = r + 1
            new_beam: List[Candidate] = []

            for cand in beam:
                improved_text = self._safe_improve(
                    lambda: self.improve(
                        cand.get("text", ""),
                        {**(cand.get("meta") or {}), "round": r},
                    ),
                    fallback=cand.get("text", "") or "",
                )
                s = self._safe_score(
                    lambda: self.score_candidate(improved_text, section_text)
                )
                new_beam.append(
                    {
                        **cand,
                        "variant": f"{cand.get('variant', 'v')}+r{r + 1}",
                        "text": improved_text,
                        "score": s,
                    }
                )

            # rank new_beam (stable tie-break as above)
            new_beam.sort(
                key=lambda x: (
                    bool(x.get("score", {}).get("verified", False)),
                    float(x.get("score", {}).get("overall", 0.0)),
                    len(x.get("text", "") or ""),
                ),
                reverse=True,
            )

            # optional diversity guard
            if self._enable_diversity_guard:
                origins = {b.get("origin") for b in new_beam}
                if len(origins) == 1:
                    alt = next(
                        (c for c in scored if c.get("origin") not in origins),
                        None,
                    )
                    if alt:
                        new_beam[-1] = alt

            curr_best = (
                new_beam[0]["score"]["overall"] if new_beam else prev_best
            )
            curr_toks = (
                self._tok(new_beam[0]["text"]) if new_beam else prev_toks
            )
            marg = _marginal(prev_best, curr_best, prev_toks, curr_toks)
            marginal_history.append(marg)
            best_history.append(curr_best)

            # compact telemetry for this round
            iters.append(
                [
                    {
                        "variant": b.get("variant"),
                        "overall": float(
                            b.get("score", {}).get("overall", 0.0)
                        ),
                        "k": float(b.get("score", {}).get("k", 0.0)),
                        "verified": bool(
                            b.get("score", {}).get("verified", False)
                        ),
                    }
                    for b in new_beam
                ]
            )

            if marg < self._min_marg:
                stop_reason = "low_marginal_reward"
                break
            if (
                len(best_history) >= 2
                and (best_history[-1] - best_history[-2]) < self._plateau_eps
            ):
                stop_reason = "plateau"
                break

            beam, prev_best, prev_toks = (
                new_beam[: self._beam_w],
                curr_best,
                curr_toks,
            )

        winner = (
            beam or scored or [{"text": "", "score": self._norm_score(None)}]
        )[0]

        return {
            "winner": winner,
            "beam": beam,
            "initial_pool": scored,
            "iterations": iters,
            "rounds_run": rounds_run,
            "best_history": best_history,
            "marginal_history": marginal_history,
            "stop_reason": stop_reason,
        }
