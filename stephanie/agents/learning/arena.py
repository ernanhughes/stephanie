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
