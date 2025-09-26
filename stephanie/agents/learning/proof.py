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
