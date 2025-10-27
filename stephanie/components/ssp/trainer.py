from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from typing import Any, Dict, Optional

from omegaconf import DictConfig, OmegaConf

from stephanie.components.ssp.actors import Proposer, Solver
from stephanie.components.ssp.core.curriculum import QMaxCurriculum
from stephanie.components.ssp.core.epistemic import EpistemicRewardCalculator
from stephanie.components.ssp.types import (Episode, EpisodeStatus, Proposal,
                                            Solution, Verification)
from stephanie.components.ssp.util import PlanTrace_safe, get_trace_logger
from stephanie.scoring.scorable import Scorable, ScorableType
from stephanie.services.service_container import ServiceContainer
from stephanie.utils.json_sanitize import sanitize


def _resolve_threshold(sp_cfg: DictConfig) -> float:
    t = OmegaConf.select(sp_cfg, "verification_threshold")
    if t is None:
        t = OmegaConf.select(sp_cfg, "verifier.verification_threshold")
    try:
        return float(t)
    except Exception:
        return 0.85


class Trainer:
    """
    Orchestrates one SSP step:
      - Propose → Solve → Score/Verify (via ScoringService)
      - Compute intrinsic/extrinsic rewards
      - Update curriculum difficulty
      - Emit traces and metrics
    """

    def __init__(self, cfg: DictConfig | dict, container: ServiceContainer):
        root = cfg
        self.root = root
        self.sp = root.self_play
        self.cfg = self.sp
        self.container = container

        # Actors (support both sync/async run styles)
        self.proposer = Proposer(root, container=container)
        self.solver = Solver(root, container=container)

        # Curriculum & reward
        self.curriculum = QMaxCurriculum(
            initial=float(self.sp.qmax.initial_difficulty),
            step=float(self.sp.qmax.difficulty_step),
            maximum=float(self.sp.qmax.max_difficulty),
            window=int(self.sp.qmax.competence_window),
            min_success_rate=float(self.sp.curriculum.min_success_rate),
        )
        self.rewards = EpistemicRewardCalculator(w_ext=0.6, w_lp=0.25, w_nov=0.15)
        self.trace_logger = get_trace_logger()
        self.success_hist = deque(maxlen=int(self.sp.qmax.competence_window))
        self.last_metrics: Dict[str, float] = {}

        # Scoring config
        self.enabled_scorers = (
            OmegaConf.select(self.sp, "verifier.scorers") or ["tiny"]
        )
        self.dimensions = (
            OmegaConf.select(self.sp, "verifier.dimensions")
            or ["reasoning", "knowledge", "clarity", "faithfulness", "coverage"]
        )
        self.threshold = _resolve_threshold(self.sp)

    # ---------------------------- helpers ---------------------------------

    @property
    def success_history(self):
        # returns a plain list (snapshotted) of recent 0/1 success flags
        return list(self.success_hist)

    def _score_with_container(self, context: dict, text: str) -> tuple[float, dict]:
        scoring = self.container.get("scoring")
        scorable = Scorable(
            id=f"sol-{int(time.time()*1_000)}",
            text=text,
            target_type=ScorableType.AGENT_OUTPUT,
            meta={
                "agent_name": "Solver",
                "stage_name": "ssp.solve",
                "pipeline_run_id": context.get("pipeline_run_id"),
                "goal_id": (context.get("goal") or {}).get("id"),
            },
        )

        dim_bag = {d: [] for d in self.dimensions}
        for scorer_name in (self.enabled_scorers or []):
            try:
                bundle = scoring.score(
                    scorer_name,
                    context=context,
                    scorable=scorable,
                    dimensions=self.dimensions,
                )
                for d in self.dimensions:
                    res = bundle.results.get(d)
                    if res is not None:
                        dim_bag[d].append(float(res.score))
            except Exception as e:
                self.trace_logger.log(PlanTrace_safe(
                    trace_id=f"scorer-{int(time.time()*1000)%1_000_000}",
                    role="trainer",
                    goal="scoring",
                    status="warning",
                    input={"scorer": scorer_name, "dimensions": self.dimensions},
                    output="scorer_error",
                    artifacts={"error": str(e)},
                ))
                continue

        dim_avg = {d: (sum(v)/len(v) if v else 0.5) for d, v in dim_bag.items()}
        final = sum(dim_avg.values()) / max(1, len(dim_avg))
        return float(final), {k: float(v) for k, v in dim_avg.items()}

    # ---------------------------- main step --------------------------------

    def train_step(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        t0 = time.time()
        ctx = dict(context or {})

        # 1) Propose
        prop_dict = self.proposer.generate(ctx)
        proposal = Proposal(**prop_dict)

        pipeline_run_id = ctx.get("pipeline_run_id") or (int(time.time()*1000) % 10_000_000)

        goal = {
            "goal_text": proposal.query,                  # ← dynamic goal
            "description": "SSP-generated goal",
            "meta": {
                "difficulty": proposal.difficulty,
                "connections": proposal.connections,
                "mission": self.sp.get("mission"),
            },
        }


        # 2) Solve
        sol_dict = self.solver.solve(prop_dict)
        solution = Solution(
            answer=sol_dict.get("answer", ""),
            reasoning_path=sol_dict.get("reasoning_path", []),
            evidence=sol_dict.get("evidence", []),
            search_depth=int(sol_dict.get("search_depth", 0)),
            report=sol_dict.get("report", {}),
            training_batch=sol_dict.get("training_batch"),
        )

        # 3) Score/Verify via ScoringService (container)
        work_ctx = {
            **ctx,
            "pipeline_run_id": pipeline_run_id,
            "goal": goal,                     # <-- ensures _score_with_container sees goal_id
            "proposal": prop_dict,
            "solution": sol_dict,
        }
        final_score, dim_scores = self._score_with_container(
            work_ctx,
            solution.answer or ""
        )


        verification = Verification(
            is_valid=bool(final_score >= self.threshold and bool(solution.answer.strip())),
            score=float(final_score),
            dimension_scores=dim_scores,
            evidence_count=int(len(solution.evidence)),
            reasoning_steps=int(len(solution.reasoning_path)),
        )

        # 4) Rewards (blend extrinsic = verification score with intrinsic signals)
        lp = 0.0  # placeholder; wire VPM learning-progress here when available
        nov = self.rewards.novelty_bonus_text(solution.answer or "")
        total_reward = self.rewards.blend(verification.score, lp, nov)

        # 5) Curriculum update
        success = verification.is_valid
        self.success_hist.append(1 if success else 0)
        new_diff = self.curriculum.update(success)
        # keep proposer aligned
        if hasattr(self.proposer, "set_difficulty"):
            self.proposer.set_difficulty(new_diff)
        else:
            setattr(self.proposer, "difficulty", new_diff)

        # 6) Trace
        episode = Episode(
            id=f"ssp-{int(time.time()*1000)}",
            proposal=proposal,
            solution=solution,
            verification=verification,
            status=EpisodeStatus.VERIFIED if success else EpisodeStatus.FAILED,
            metrics={
                "reward": total_reward,
                "verification": verification.score,
                "novelty": nov,
                "success_rate": self.curriculum.success_rate,
                "difficulty": new_diff,
                "threshold": self.threshold,
            },
        )

        self.trace_logger.log(
            PlanTrace_safe(
                trace_id=f"ssp-train-{int(time.time()*1000)%1_000_000}",
                role="trainer",
                goal=proposal.query,
                status="completed",
                metadata=episode.metrics,
                input=proposal.raw_response,
                output=solution.answer,
                artifacts=sanitize(
                    {
                        "proposal": prop_dict,
                        "solution": sol_dict,
                        "dimension_scores": dim_scores,
                        "enabled_scorers": self.enabled_scorers,
                    }
                ),
            )
        )

        self.last_metrics = episode.metrics | {"duration_ms": int((time.time() - t0) * 1000)}
        return sanitize(
            {
                "episode_id": episode.id,
                "success": success,
                "metrics": episode.metrics,
                "training_batch": solution.training_batch,  # may be None
            }
        )
