# stephanie/components/ssp/core/algorithm.py
"""
Main SSP (Self-Play System) Algorithm

Implements the complete SSP loop (paper-aligned):
1) Proposer generates a question (optionally with evidence)
2) Solver answers the question with search
3) RAGVerifier adjudicates Proposer's seed (A) vs Solver's answer (B)
4) Rewards are computed and attached
5) (Optional) VPM visualization

This version expects a RAG-style verifier with:
  verify(question, seed_answer, predicted_answer, evidence_docs, context)
    -> (solver_wins: bool, score_1_to_100: float, details: dict)
"""
from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np

from stephanie.components.ssp.core.roles.proposer import Proposer
from stephanie.components.ssp.core.roles.solver import Solver
from stephanie.components.ssp.core.roles.verifier import Verifier

from stephanie.components.ssp.core.protocols import EpisodeContext, SSPMetrics
from stephanie.components.ssp.utils.trace import EpisodeTrace
from stephanie.components.ssp.training.rewards import (
    calculate_self_play_rewards,
    update_episode_with_rewards,
)
from stephanie.components.ssp.services.vpm_visualization_service import (
    VPMVisualizationService,
)

from stephanie.components.ssp.metrics.scorer import SSPScorer
from stephanie.components.ssp.metrics.scorable import SSPScorable



import logging
_logger = logging.getLogger(__name__)

class SSPAlgorithm:
    """
    Paper-aligned SSP orchestrator (A vs B adjudication happens AFTER solving).
    """

    def __init__(
        self,
        proposer: Proposer,
        solver: Solver,
        verifier: Verifier,
        vpm_visualization: Optional[VPMVisualizationService] = None,
        **kwargs: Any,
    ):
        self.proposer = proposer
        self.solver = solver
        self.verifier = verifier
        self.vpm_visualization = vpm_visualization
        self.metrics = SSPMetrics()
        self._ssp_scorer = SSPScorer()
        self.episode_history: List[EpisodeTrace] = []

    # ------------------------- public API -------------------------

    async def run_episode(
        self,
        seed_answer: str,
        context: Optional[EpisodeContext] = None,
    ) -> EpisodeTrace:
        """
        Run one SSP episode from a SEED_ANSWER.
        Returns a fully populated EpisodeTrace (with rewards attached if verified).
        """
        ctx = context or {}
        episode_id = f"ssp-{int(time.time()*1000)}-{uuid.uuid4().hex[:6]}"
        start_time = time.time()
        ctx["vpm_unit"] = episode_id
        try:
            # 1) Proposer: produce a question (and, if available, proposer_evidence)
            # Expected: (question, proposer_evidence, proposer_meta)
            q, prop_evidence, prop_meta = await self._call_proposer(seed_answer, ctx)

            # 2) Solver: answer with search (returns predicted_answer + evidence_docs)
            pred, evidence_docs, solver_steps, solver_meta = await self._call_solver(q, seed_answer, ctx)

            # 3) Verifier (adversarial A vs B, RAG-style): did solver beat the seed?
            solver_wins, judge_score, judge_details = await self._call_verifier(q, seed_answer, pred, evidence_docs, ctx)

            # 4) Build episode
            ep = EpisodeTrace(
                episode_id=episode_id,
                seed_answer=seed_answer,
                question=q,
                proposer_evidence=prop_evidence,
                predicted_answer=pred,
                evidence_docs=evidence_docs,
                verified=bool(solver_wins),
                verifier_score=float(judge_score),
                solver_steps=int(solver_steps),
                difficulty=float(prop_meta.get("difficulty", 0.5)),
                proposer_meta=prop_meta,
                verifier_meta=judge_details,
                solver_meta=solver_meta,
                timestamp=datetime.now(),
                episode_duration=time.time() - start_time,
            )

            # ---- Canonical SSP metrics (single source) ----
            verifier01 = float(judge_score) / 100.0 if judge_score is not None else 0.0
            scorable = SSPScorable(
                episode_id=episode_id,
                question=q,
                seed_answer=seed_answer,
                predicted_answer=pred,
                evidence_snippets=evidence_docs,
                solver_steps=int(solver_steps),
                difficulty01=float(prop_meta.get("difficulty", 0.0)) / 100.0 if prop_meta.get("difficulty", None) and prop_meta["difficulty"] > 1 else float(prop_meta.get("difficulty", 0.0)),
                verifier_score01=verifier01,
                verified01=1.0 if solver_wins else 0.0,
            )
            ssp_metrics = self._ssp_scorer.score(scorable)  # {'names','values','vector'}

            # Attach for downstream consumers (manifest, storage, VPM, etc.)
            ep.solver_meta = (ep.solver_meta or {})
            ep.solver_meta["ssp_metrics"] = ssp_metrics

            # Initial snapshot for the blog's “proposed → solved” narrative
            if self.vpm_visualization:
                dims = ssp_metrics["vector"]  # already 0..1 and canonical
                self.vpm_visualization.snapshot_progress(
                    unit=episode_id, dims=dims, step_idx=0, tag="proposed"
                )
                _logger.info(
                    "Episode proposed snapshot saved",
                    extra={
                        "episode_id": episode_id,
                        "question": ep.question,
                        "dims": {k: round(float(v), 4) for k, v in dims.items()},
                    },
                )

            # 5) Rewards: compute on verified only (paper’s game signal)
            if ep.verified:
                self._calculate_and_apply_rewards([ep], unverified_count=0)

            # 6) Metrics + history
            self._update_metrics(ep)
            self.episode_history.append(ep)

            # 7) VPM visualization (optional)
            if self.vpm_visualization:
                try:
                    # Minimal bar (1×F) for the post
                    self.vpm_visualization.generate_episode_visualization(
                        unit=episode_id, episode=ep, dims_override=ssp_metrics["vector"]
                    )

                    # Build RAW & PHOS using the step snapshots collected during search
                    raw_path  = self.vpm_visualization.generate_raw_vpm_image(unit=episode_id)
                    phos_path = self.vpm_visualization.generate_phos_image(unit=episode_id)
                    film_path = self.vpm_visualization.generate_filmstrip(unit=episode_id)
                    gif_path  = self.vpm_visualization.finalize_progress(unit=episode_id)

                    _logger.info("VPM artifacts",
                        extra={"unit": episode_id, "raw": raw_path, "phos": phos_path,
                               "filmstrip": film_path, "gif": gif_path})
                except Exception:
                    # Keep the loop resilient
                    pass

            return ep

        except Exception as e:
            _logger.error("SSP episode failed", extra={"episode_id": episode_id, "error": str(e)})

            # Count the episode, update pass rate, and emit a safe trace
            self.metrics.total_episodes += 1
            self.metrics.verification_pass_rate = (
                self.metrics.verified_episodes / max(self.metrics.total_episodes, 1)
            )
            return EpisodeTrace(
                episode_id=episode_id,
                seed_answer=seed_answer,
                question="",
                proposer_evidence=[],
                predicted_answer="",
                evidence_docs=[],
                verified=False,
                verifier_score=0.0,
                solver_steps=0,
                difficulty=0.0,
                proposer_meta={"error": str(e)},
                verifier_meta={"error": str(e)},
                solver_meta={"error": str(e)},
                timestamp=datetime.now(),
                episode_duration=time.time() - start_time,
            )

    async def train_step(
        self,
        seed_answers: List[str],
        context: Optional[EpisodeContext] = None,
    ) -> Dict[str, Any]:
        """
        Run multiple episodes in parallel (one per seed), then compute summary rewards/metrics.
        """
        ctx = context or {}
        # Run episodes concurrently
        tasks = [self.run_episode(seed, ctx) for seed in seed_answers]
        episodes: List[EpisodeTrace] = await asyncio.gather(*tasks)

        # Build a filmstrip per episode (uses frames emitted during solver search)
        if self.vpm_visualization:
            for ep in episodes:
                unit = (
                    (ep.solver_meta or {}).get("vpm_unit")
                    or ep.episode_id
                )
                try:
                    film_path = self.vpm_visualization.generate_filmstrip(unit=unit)
                    _logger.info("Filmstrip ready", extra={"unit": unit, "path": film_path})
                except Exception as e:
                    _logger.warning("Filmstrip generation skipped", extra={"unit": unit, "error": str(e)})

        verified = [ep for ep in episodes if ep.verified]
        unverified = len(episodes) - len(verified)

        # Rewards summary across verified episodes (for reporting)
        rewards = calculate_self_play_rewards(verified, unverified)

        # Keep for future analysis / training
        self.episode_history.extend(episodes)

        return {
            **rewards,
            "total_episodes": len(episodes),
            "verified_count": len(verified),
            "unverified_count": unverified,
            "metrics": self.get_metrics().__dict__,
        }

    def get_metrics(self) -> SSPMetrics:
        return self.metrics

    def reset(self) -> None:
        self.metrics = SSPMetrics()
        self.episode_history.clear()

    def is_initialized(self) -> bool:
        return all([self.proposer is not None, self.solver is not None, self.verifier is not None])

    # ------------------------- internals -------------------------

    async def _call_proposer(self, seed_answer: str, ctx: EpisodeContext):
        out = await self.proposer.propose(seed_answer, context=ctx)
        # Backward compatibility: handle (question, meta) form
        if isinstance(out, tuple) and len(out) == 3:
            return out  # (question, proposer_evidence, proposer_meta)
        elif isinstance(out, tuple) and len(out) == 2:
            q, meta = out
            evid = meta.get("evidence_snippets", []) if isinstance(meta, dict) else []
            return q, evid, (meta or {})
        else:
            # Extreme fallback
            return str(out), [], {}

    async def _call_solver(self, question: str, seed_answer: str, ctx: EpisodeContext):
        """
        Expected solver API:
          solve(question, seed_answer, context, use_search=True)
            -> predicted_answer, evidence_docs, solver_steps, solver_meta
        """
        out = await self.solver.solve(question, seed_answer, context=ctx)
        # Backward compatibility for (pred, evid, steps)
        if isinstance(out, tuple) and len(out) == 4:
            return out 
        elif isinstance(out, tuple) and len(out) == 3:
            pred, evid, steps = out
            return pred, evid, steps, {}
        # Worst case: only answer came back
        return str(out), [], 0, {}

    async def _call_verifier(
        self,
        question: str,
        seed_answer: str,
        predicted_answer: str,
        evidence_docs: List[str],
        ctx: EpisodeContext,
    ):
        """
        Expected verifier API (RAG-style judge):
          verify(question, seed_answer, predicted_answer, evidence_docs, context)
            -> (solver_wins: bool, score_1_to_100: float, details: dict)
        """
        out = await self.verifier.verify(
            question=question,
            seed_answer=seed_answer,
            predicted_answer=predicted_answer,
            evidence=evidence_docs,
            context=ctx,
        )
        # Backward compatibility: allow VerificationResult-like objects
        if isinstance(out, tuple) and len(out) == 3:
            return out
        elif hasattr(out, "is_valid"):
            return bool(out.is_valid), float(getattr(out, "score", 0.0)), dict(getattr(out, "verification_details", {}))
        return False, 0.0, {"error": "unsupported verifier return"}

    def _calculate_and_apply_rewards(self, verified_episodes: List[EpisodeTrace], unverified_count: int) -> None:
        rewards = calculate_self_play_rewards(verified_episodes, unverified_count)
        for ep in verified_episodes:
            update_episode_with_rewards(ep, rewards.get("solver_reward", 0.0), rewards.get("proposer_reward", 0.0))

    def _update_metrics(self, ep: EpisodeTrace) -> None:
        self.metrics.total_episodes += 1
        if ep.verified:
            self.metrics.verified_episodes += 1

            # Proposer metrics
            self.metrics.proposer_success_rate = self.metrics.verified_episodes / max(self.metrics.total_episodes, 1)
            self.metrics.avg_question_difficulty = (
                (self.metrics.avg_question_difficulty * max(self.metrics.verified_episodes - 1, 0) + ep.difficulty)
                / max(self.metrics.verified_episodes, 1)
            )

            # Solver metrics
            solved_correct = 1.0 if (ep.predicted_answer or "").strip() == (ep.seed_answer or "").strip() else 0.0
            self.metrics.solver_accuracy = (
                (self.metrics.solver_accuracy * max(self.metrics.verified_episodes - 1, 0) + solved_correct)
                / max(self.metrics.verified_episodes, 1)
            )
            self.metrics.avg_solver_steps = (
                (self.metrics.avg_solver_steps * max(self.metrics.verified_episodes - 1, 0) + ep.solver_steps)
                / max(self.metrics.verified_episodes, 1)
            )

            # Verification metrics
            self.metrics.verification_pass_rate = self.metrics.verified_episodes / max(self.metrics.total_episodes, 1)
            self.metrics.avg_verification_score = (
                (self.metrics.avg_verification_score * max(self.metrics.verified_episodes - 1, 0) + ep.verifier_score)
                / max(self.metrics.verified_episodes, 1)
            )

        # Self-play rewards (if attached)
        if getattr(ep, "proposer_reward", None) is not None and getattr(ep, "solver_reward", None) is not None:
            self.metrics.proposer_adversarial_reward = (
                (self.metrics.proposer_adversarial_reward * max(self.metrics.total_episodes - 1, 0) + ep.proposer_reward)
                / max(self.metrics.total_episodes, 1)
            )
            self.metrics.solver_cooperative_reward = (
                (self.metrics.solver_cooperative_reward * max(self.metrics.total_episodes - 1, 0) + ep.solver_reward)
                / max(self.metrics.total_episodes, 1)
            )

        # Curriculum smoothing
        self.metrics.curriculum_difficulty = (self.metrics.curriculum_difficulty * 0.9) + (ep.difficulty * 0.1)
        self.metrics.last_updated = datetime.now()
