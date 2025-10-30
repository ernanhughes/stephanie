from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from typing import Any, Dict, Optional, List
import traceback
import inspect

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
from stephanie.data.plan_trace import PlanTrace, ExecutionStep
from stephanie.constants import PLAN_TRACE_ID  # Assuming you have a constant for the key

import logging
_logger = logging.getLogger(__name__)

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

    def __init__(self, cfg: DictConfig | dict, memory, container: ServiceContainer):
        root = cfg
        self.memory = memory
        self.root = root
        self.sp = root.self_play
        self.cfg = self.sp
        self.container = container

        # Actors (support both sync/async run styles)
        self.proposer = Proposer(root, memory, container=container)

        # NEW: be compatible with both old and new Solver signatures
        try:
            self.solver = Solver(root, memory, container)  # (cfg, container) — new
        except TypeError:
            self.solver = Solver(root, memory, container=container)  # old fallback

        # Curriculum & reward
        self.curriculum = self._build_qmax_curriculum()
        self.rewards = EpistemicRewardCalculator(w_ext=0.6, w_lp=0.25, w_nov=0.15)
        self.trace_logger = get_trace_logger()
        self.success_hist = deque(maxlen=int(self.sp.qmax.competence_window))
        self.last_metrics: Dict[str, float] = {}

        # Scoring config
        self.enabled_scorers = (
            OmegaConf.select(self.sp, "verifier.scorers") or ["tiny"]
        )
        self.dimensions: List[str] = (
            OmegaConf.select(self.sp, "verifier.dimensions")
            or ["reasoning", "knowledge", "clarity", "faithfulness", "coverage"]
        )
        self.threshold = _resolve_threshold(self.sp)

    # ---------------------------- helpers ---------------------------------

    def _build_qmax_curriculum(self) -> QMaxCurriculum:
        """
        Build QMaxCurriculum regardless of whether its ctor expects
        (initial/step/maximum/window/min_success_rate) or older/newer aliases.
        """
        sig = inspect.signature(QMaxCurriculum)
        params = set(sig.parameters.keys())

        # Values from config
        v_initial = float(self.sp.qmax.initial_difficulty)
        v_step = float(self.sp.qmax.difficulty_step)
        v_max = float(self.sp.qmax.max_difficulty)
        v_window = int(self.sp.qmax.competence_window)
        v_min_succ = float(self.sp.curriculum.min_success_rate)

        # Candidate name maps (left to right preference)
        name_map = {
            "initial": ["initial", "start", "initial_difficulty", "initial_value"],
            "step": ["step", "delta", "difficulty_step"],
            "maximum": ["maximum", "max", "max_difficulty", "upper"],
            "window": ["window", "window_size", "competence_window", "history"],
            "min_success_rate": ["min_success_rate", "target_success_rate", "success_threshold", "min_success"],
        }
        values = {
            "initial": v_initial,
            "step": v_step,
            "maximum": v_max,
            "window": v_window,
            "min_success_rate": v_min_succ,
        }

        kwargs = {}
        for logical, candidates in name_map.items():
            for cand in candidates:
                if cand in params:
                    kwargs[cand] = values[logical]
                    break

        # If something critical is missing (e.g., initial), still pass sensible defaults
        # using any remaining param names
        for p in params:
            if p not in kwargs:
                # naive fallback based on type/semantics
                if "init" in p or "start" in p:
                    kwargs[p] = v_initial
                elif "step" in p or "delta" in p:
                    kwargs[p] = v_step
                elif "max" in p or "upper" in p:
                    kwargs[p] = v_max
                elif "window" in p or "hist" in p:
                    kwargs[p] = v_window
                elif "rate" in p or "threshold" in p:
                    kwargs[p] = v_min_succ

        return QMaxCurriculum(**kwargs)

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
    async def train_step(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute one full Self-Play System (SSP) training step.
        This method now creates a proper PlanTrace with multiple ExecutionSteps.
        """
        t0 = time.time()
        ctx = dict(context or {})
        pipeline_run_id = ctx.get("pipeline_run_id") or f"ssp-{int(time.time()*1000)}"

        # --- START OF TRAIN_STEP ---
        print(f"\n--- SSP TRAINING STEP START [{time.strftime('%H:%M:%S')}] ---", flush=True)

        # === Create the Master PlanTrace ===
        plan_trace = PlanTrace(
            trace_id=f"ssp-step-{pipeline_run_id}",
            pipeline_run_id=pipeline_run_id,
            goal_text="Self-Play Improvement Cycle",
            goal_id=hash(pipeline_run_id),
            input_data={
                "initial_context_keys": list(ctx.keys()),
                "mission": self.sp.get("mission"),
            },
            plan_signature="SSP-v1",
            execution_steps=[],
            final_output_text="",
            status="in_progress",
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            meta={
                "ssp_version": "1.0",
                "start_time": t0,
                "difficulty": getattr(self.proposer, "difficulty", 0.3),
            }
        )

        # Inject the trace ID into the context so child agents can find it
        ctx[PLAN_TRACE_ID] = plan_trace.trace_id

        try:
            # --- 1) PROPOSE PHASE ---
            propose_start = time.time()
            print("🔄 Generating proposal...", end="", flush=True)

            in_goal_text = ((context or {}).get("goal") or {}).get("goal_text", "")

            propose_step = ExecutionStep(
                step_id="propose-1",
                pipeline_run_id=pipeline_run_id,
                step_order=1,
                step_type="proposal_generation",
                description="Generate a novel, verifiable research question.",
                agent_name="Proposer",
                scores={},
                output_text="",
                agent_role="retrieve",
                start_time=propose_start,
                input_text=in_goal_text,
                meta={"connections": getattr(self.proposer, "_get_recent_connections", lambda: [])()},
            )

            prop_dict = await self.proposer.generate_async(ctx)
            proposal = Proposal(**prop_dict)

            propose_step.output_text = f"Query: {proposal.query}\nDifficulty: {proposal.difficulty:.2f}"
            propose_step.end_time = time.time()
            propose_step.duration = propose_step.end_time - propose_start
            propose_step.status = "completed"

            plan_trace.execution_steps.append(propose_step)
            print(" ✅")

            # --- 2) SOLVE PHASE ---
            solve_start = time.time()
            print("🧠 Solving proposal...", end="", flush=True)

            solve_step = ExecutionStep(
                step_id="solve-1",
                pipeline_run_id=pipeline_run_id,
                step_order=2,
                step_type="problem_solving",
                description="Use agentic search to solve the proposed query.",
                agent_name="Solver",
                agent_role="revise",
                start_time=solve_start,
                input_text=proposal.query,
                scores={},
                output_text="",
                meta={"use_grpo": getattr(self.solver, "use_grpo", False)},
            )

            sol_dict = await self.solver.solve(proposal.to_dict())
            solution = Solution(**sol_dict)

            solve_step.output_text = f"Answer preview: {solution.answer[:100]}..."
            solve_step.end_time = time.time()
            solve_step.duration = solve_step.end_time - solve_start
            solve_step.status = "completed"
            solve_step.output_size = len(solution.answer or "")

            plan_trace.execution_steps.append(solve_step)
            solve_time = int((time.time() - solve_start) * 1000)
            print(f" ✅ ({solve_time}ms)")

            # --- 3) VERIFY PHASE ---
            verify_start = time.time()
            print("🔍 Verifying solution...", end="", flush=True)

            verify_step = ExecutionStep(
                step_id="verify-1",
                pipeline_run_id=pipeline_run_id,
                step_order=3,
                step_type="solution_verification",
                description="Score the solution across multiple dimensions for validity.",
                agent_name="Verifier",
                agent_role="retain",
                start_time=verify_start,
                input_text=solution.answer,
                scores={},
                output_text=solution.answer,
                meta={"scorers_used": self.enabled_scorers, "dimensions": self.dimensions},
            )

            work_ctx = {
                **ctx,
                "goal": {"goal_text": proposal.query},
                "solution": sol_dict,
            }
            final_score, dim_scores = self._score_with_container(work_ctx, solution.answer or " ")

            verification = Verification(
                is_valid=bool(final_score >= self.threshold and bool((solution.answer or "").strip())),
                score=float(final_score),
                dimension_scores=dim_scores,
                evidence_count=int(len(solution.evidence or [])),
                reasoning_steps=int(len(solution.reasoning_path or [])),
            )

            verify_results = "\n".join([f"{k}: {v:.3f}" for k, v in dim_scores.items()])
            verify_step.output_text = f"Verification Score: {final_score:.3f}\n{verify_results}"
            verify_step.end_time = time.time()
            verify_step.duration = verify_step.end_time - verify_start
            verify_step.status = "completed"

            plan_trace.execution_steps.append(verify_step)
            print(" ✅")

            # --- POST-PROCESSING ---
            success = verification.is_valid
            plan_trace.final_output_text = solution.answer or ""
            plan_trace.status = "completed" if success else "failed"
            plan_trace.meta["total_duration"] = time.time() - t0
            plan_trace.meta["success"] = success
            plan_trace.meta["verification_score"] = verification.score
            plan_trace.reward_signal = {"total_reward": float(self.rewards.blend(verification.score, 0.0, 0.0))}

            # Curriculum update
            self.success_hist.append(1 if success else 0)
            new_diff = self.curriculum.update(success)
            if hasattr(self.proposer, "set_difficulty"):
                self.proposer.set_difficulty(new_diff)
            else:
                setattr(self.proposer, "difficulty", new_diff)
            plan_trace.meta["new_difficulty"] = new_diff

            # Logging & persistence
            print(f"✅ SUCCESS: {success}, VERIFICATION SCORE: {verification.score:.3f}")
            print(f"📊 METRICS: difficulty={new_diff:.2f}, success_rate={self.curriculum.success_rate:.2%}")
            print(f"📈 DIMENSION SCORES: {', '.join([f'{k}={v:.2f}' for k, v in dim_scores.items()])}")
            print(f"--- SSP TRAINING STEP END [{time.strftime('%H:%M:%S')}] ---\n", flush=True)

            if self.container and hasattr(self.container, "get"):
                try:
                    memory = self.container.get("memory")
                    if hasattr(memory, "plan_traces") and hasattr(memory.plan_traces, "upsert"):
                        memory.plan_traces.upsert(plan_trace)
                        print(f"💾 PlanTrace saved to memory: {plan_trace.trace_id}", flush=True)
                    else:
                        print("⚠️ Memory service missing 'plan_traces' repository.", flush=True)
                except Exception as e:
                    print(f"❌ Failed to save PlanTrace: {str(e)}", flush=True)

            episode_id = plan_trace.trace_id
            result = {
                "episode_id": episode_id,
                "success": success,
                "metrics": {
                    "reward": self.rewards.blend(verification.score, 0.0, 0.0),
                    "verification": float(verification.score),
                    "novelty": 0.0,
                    "success_rate": self.curriculum.success_rate,
                    "difficulty": new_diff,
                    "threshold": self.threshold,
                    "duration_ms": int((time.time() - t0) * 1000),
                },
                "training_batch": sol_dict.get("training_batch"),
                "plan_trace_id": episode_id,
            }

            return result

        except Exception as e:
            plan_trace.status = "failed"
            plan_trace.meta["error"] = {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
            }
            print(f"❌ SSP training step failed: {str(e)}", flush=True)

            if self.container:
                try:
                    memory = self.container.get("memory")
                    if hasattr(memory, "plan_traces") and hasattr(memory.plan_traces, "upsert"):
                        memory.plan_traces.upsert(plan_trace)
                except Exception:
                    pass

            raise
