# stephanie/components/vibe/evaluator.py
from __future__ import annotations

from stephanie.components.vibe.vibe import VibeCandidate, VibeFeatureTask, VibeScore


class VibeEvaluator:
    def __init__(self, cfg, memory, logger):
        self.cfg = cfg or {}
        self.memory = memory
        self.logger = logger

    async def evaluate(self, candidate: VibeCandidate) -> VibeScore:
        task = candidate.task

        # 1) Functional: run F2P + P2P tests
        functional = await self._run_tests(task, candidate)

        # 2) Vibe: apply RubricMemCubes
        vibe, vibe_breakdown = await self._run_rubrics(task, candidate)

        # 3) Robustness: CCS via perturbations
        robustness = await self._compute_ccs(task, candidate)

        # 4) Cost: tokens, latency, etc.
        cost = self._estimate_cost(candidate)

        # 5) Risk: Daimon + HRM/SICQL flags
        risk = await self._estimate_risk(task, candidate)

        breakdown = {"functional": functional, "robustness": robustness, **vibe_breakdown}
        return VibeScore(
            functional=functional,
            vibe=vibe,
            robustness=robustness,
            cost=cost,
            risk=risk,
            breakdown=breakdown,
        )

    async def _run_rubrics(self, task: VibeFeatureTask, candidate: VibeCandidate):
        # 1) RubricMemCubes (structured rubrics)
        rubric_scores = {}
        for rubric in task.rubrics:
            rubric_scores.update(await rubric.evaluate(candidate))

        # 2) Legacy vibe model as an extra rubric
        legacy_scores = await self.legacy_vibe_rubric.score(candidate)
        rubric_scores.update(legacy_scores)  # or keep separate and weight

        # 3) Aggregate into a single `vibe` scalar
        weights = self.cfg.get("vibe_weights", {
            "style": 0.3, "clarity": 0.3, "safety": 0.4
        })

        vibe = 0.0
        for k, w in weights.items():
            vibe += w * rubric_scores.get(k, 0.0)

        return vibe, rubric_scores
