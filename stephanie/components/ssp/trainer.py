# stephanie/components/ssp/trainer.py
from __future__ import annotations

import time
import traceback
import inspect
from collections import deque
from typing import Any, Dict, Optional, List, Tuple

from omegaconf import DictConfig, OmegaConf

from stephanie.components.ssp.actors import Proposer, Solver
from stephanie.components.ssp.core.curriculum import QMaxCurriculum
from stephanie.components.ssp.core.epistemic import EpistemicRewardCalculator
from stephanie.components.ssp.types import Proposal, Solution, Verification
from stephanie.scoring.scorable import Scorable, ScorableType
from stephanie.services.service_container import ServiceContainer

import logging
_logger = logging.getLogger(__name__)


# ---------- small utils ----------

def _sel(cfg: DictConfig | dict, path: str, default: Any = None) -> Any:
    """OmegaConf-safe select with dict fallback."""
    if isinstance(cfg, DictConfig):
        val = OmegaConf.select(cfg, path)
        return default if val is None else val
    cur = cfg
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur

def _resolve_threshold(sp_cfg: DictConfig | dict) -> float:
    t = _sel(sp_cfg, "verification_threshold")
    if t is None:
        t = _sel(sp_cfg, "verifier.verification_threshold")
    try:
        return float(t)
    except Exception:
        return 0.85

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------- normalizers (fix for your error) ----------

def _as_mapping(x: Any) -> Dict[str, Any]:
    """Best-effort convert x into a dict mapping."""
    if x is None:
        return {}
    if isinstance(x, dict):
        return dict(x)
    if hasattr(x, "to_dict"):
        try:
            return dict(x.to_dict())
        except Exception:
            pass
    if hasattr(x, "__dict__"):
        try:
            return dict(vars(x))
        except Exception:
            pass
    return {}

def _proposal_dict(raw: Any, default_difficulty: float) -> Dict[str, Any]:
    """
    Accepts dict / list / tuple / string / object and produces a {query, difficulty, ...} dict.
    This prevents: Proposal() argument after ** must be a mapping, not list
    """
    # If list/tuple → take first non-empty item
    if isinstance(raw, (list, tuple)):
        if not raw:
            return {"query": "", "difficulty": default_difficulty}
        return _proposal_dict(raw[0], default_difficulty)

    # String → treat as query
    if isinstance(raw, str):
        return {"query": raw, "difficulty": default_difficulty}

    # Mapping / object-ish
    d = _as_mapping(raw)
    # try common keys for query
    query = (
        d.get("query")
        or d.get("goal_text")
        or d.get("prompt")
        or d.get("text")
        or d.get("question")
        or ""
    )
    # difficulty
    diff = d.get("difficulty")
    try:
        diff = default_difficulty if diff is None else float(diff)
    except Exception:
        diff = default_difficulty

    out = dict(d)
    out["query"] = query
    out["difficulty"] = diff
    return out

def _solution_dict(raw: Any) -> Dict[str, Any]:
    """
    Normalize solver output into a {answer, evidence, reasoning_path, training_batch} dict.
    """
    if isinstance(raw, (list, tuple)):
        if not raw:
            return {"answer": ""}
        return _solution_dict(raw[0])

    if isinstance(raw, str):
        return {"answer": raw}

    d = _as_mapping(raw)
    ans = d.get("answer") or d.get("text") or d.get("output") or d.get("response") or ""
    ev = d.get("evidence") or d.get("sources") or []
    rp = d.get("reasoning_path") or d.get("trace") or d.get("steps") or []
    tb = d.get("training_batch") or None

    out = dict(d)
    out["answer"] = ans
    out["evidence"] = ev
    out["reasoning_path"] = rp
    if tb is not None:
        out["training_batch"] = tb
    return out


class Trainer:
    """
    Self-Play System (SSP) Trainer — bus-first.

    Orchestrates one SSP step:
      1) Propose  →  2) Solve  →  3) Verify/Score
      - Computes intrinsic/extrinsic rewards
      - Updates curriculum (QMax)
      - Emits *bus events*; PlanTrace persistence is optional (off by default)
    """

    def __init__(self, cfg: DictConfig | dict, memory, container: ServiceContainer, logger):
        # Config
        self.root = cfg if isinstance(cfg, DictConfig) else OmegaConf.create(cfg or {})
        self.sp = _sel(self.root, "self_play", {}) or {}
        self.memory = memory
        self.container = container
        self.logger = logger

        # Actors
        self.proposer = Proposer(self.root, memory, container=container, logger=logger)
        self.solver   = Solver(self.root,   memory, container=container, logger=logger)

        # Curriculum & reward
        self.curriculum = QMaxCurriculum(
            window=int(_sel(self.sp, "qmax.competence_window", 200)),
            target_success=float(_sel(self.sp, "curriculum.min_success_rate", 0.65)),
            min_diff=float(_sel(self.sp, "qmax.min_difficulty", 0.1)),
            max_diff=float(_sel(self.sp, "qmax.max_difficulty", 1.0)),
        )
        try:
            init_d = float(_sel(self.sp, "qmax.initial_difficulty", 0.30))
            if hasattr(self.curriculum, "_d"):
                self.curriculum._d = init_d  # internal
        except Exception:
            pass

        self.rewards = EpistemicRewardCalculator(
            w_ext=float(_sel(self.sp, "reward.w_ext", 0.60)),
            w_lp=float(_sel(self.sp, "reward.w_lp", 0.25)),
            w_nov=float(_sel(self.sp, "reward.w_nov", 0.15)),
        )
        self.success_hist = deque(maxlen=int(_sel(self.sp, "qmax.competence_window", 200)))

        # Scoring config
        self.enabled_scorers: List[str] = list(_sel(self.sp, "verifier.scorers", ["tiny"]))
        self.dimensions: List[str] = list(
            _sel(self.sp, "verifier.dimensions",
                 ["reasoning", "knowledge", "clarity", "faithfulness", "coverage"])
        )
        self.threshold = _resolve_threshold(self.sp)

        # Emission knobs
        self.emit_plan_traces: bool = bool(_sel(self.sp, "emit_plan_traces", False))
        self.emit_events: bool = bool(_sel(self.sp, "emit_events", True))

        # Adaptive gate knobs
        self.adaptive_gate: bool = bool(_sel(self.sp, "verifier.adaptive", True))
        self.adapt_start: float = float(_sel(self.sp, "verifier.start", 0.55))
        self.adapt_cap: float = float(_sel(self.sp, "verifier.max", 0.85))
        self.adapt_rise_per_100: float = float(_sel(self.sp, "verifier.rise_per_100", 0.06))

    # ---------------- internal helpers ----------------

    async def _maybe_call(self, fn, *args, **kwargs):
        """Call sync or async function seamlessly; fallback to .run if needed."""
        try:
            res = fn(*args, **kwargs)
            if inspect.isawaitable(res):
                return await res
            return res
        except TypeError:
            if hasattr(fn, "run"):
                res = fn.run(*args, **kwargs)  # type: ignore[attr-defined]
                return await res if inspect.isawaitable(res) else res
            raise

    async def _publish(self, subject: str, payload: dict) -> None:
        if not self.emit_events:
            return
        # bus publish (best-effort)
        try:
            if hasattr(self.memory, "bus") and self.memory.bus:
                await self.memory.bus.publish(subject, payload)
        except Exception as e:
            if self.logger:
                self.logger.log("SSPPublishBusError", {"subject": subject, "error": str(e)})
        # local event store mirror (best-effort)
        try:
            if hasattr(self.memory, "bus_events") and self.memory.bus_events:
                self.memory.bus_events.insert(subject, payload)
        except Exception as e:
            if self.logger:
                self.logger.log("SSPBusEventsInsertError", {"subject": subject, "error": str(e)})

    def _adaptive_threshold(self) -> float:
        if not self.adaptive_gate:
            return self.threshold
        base = float(self.adapt_start)
        cap = float(self.adapt_cap)
        growth = float(self.adapt_rise_per_100) * (len(self.success_hist) / 100.0)
        return min(cap, base + growth)

    def _compute_novelty(self, solution_text: str) -> float:
        # Hook point for novelty — safe default keeps pipeline stable
        return 0.0

    def _score_with_container(self, context: dict, text: str) -> Tuple[float, Dict[str, float]]:
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
        dim_bag: Dict[str, List[float]] = {d: [] for d in self.dimensions}
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
                if self.logger:
                    self.logger.log("SSPScorerError", {"scorer": scorer_name, "error": str(e)})
        dim_avg = {d: (sum(v) / len(v) if v else 0.5) for d, v in dim_bag.items()}
        final = sum(dim_avg.values()) / max(1, len(dim_avg))
        return float(final), {k: float(v) for k, v in dim_avg.items()}

    # ---------------- public ----------------

    async def train_step(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        t0 = time.time()
        ctx: Dict[str, Any] = dict(context or {})
        run_id = ctx.get("pipeline_run_id") or str(int(time.time() * 1000) % 1_000_000)
        ctx["pipeline_run_id"] = run_id

        await self._publish("ssp.episode.started", {
            "episode_id": str(run_id),
            "at": _now_iso(),
            "ssp_version": "1.0",
            "config": {
                "scorers": self.enabled_scorers,
                "dimensions": self.dimensions,
                "adaptive_gate": self.adaptive_gate,
            },
        })
        if self.logger:
            self.logger.log("SSPEpisodeStarted", {"run_id": run_id})

        steps: List[Dict[str, Any]] = []

        try:
            # 1) PROPOSE  ---------------------------------------------------
            p_t = time.time()
            goal_text = ((ctx.get("goal") or {}).get("goal_text") or "").strip()
            raw_prop = await self._maybe_call(self.proposer.propose, ctx)
            prop_d = _proposal_dict(raw_prop, default_difficulty=float(_sel(self.sp, "qmax.initial_difficulty", 0.30)))
            proposal = Proposal(**prop_d)
            steps.append({
                "id": "propose-1",
                "type": "proposal_generation",
                "agent": "Proposer",
                "start_time": p_t,
                "end_time": time.time(),
                "duration": time.time() - p_t,
                "input_text": goal_text,
                "output": {"query": proposal.query, "difficulty": float(proposal.difficulty or 0.0)},
                "status": "completed",
            })

            # 2) SOLVE  -----------------------------------------------------
            s_t = time.time()
            raw_sol = await self._maybe_call(self.solver.solve, proposal.to_dict())
            sol_d = _solution_dict(raw_sol)
            solution = Solution(**sol_d)
            steps.append({
                "id": "solve-1",
                "type": "problem_solving",
                "agent": "Solver",
                "start_time": s_t,
                "end_time": time.time(),
                "duration": time.time() - s_t,
                "input_text": proposal.query,
                "output_size": len((solution.answer or "")),
                "status": "completed",
            })

            # 3) VERIFY  ----------------------------------------------------
            self.threshold = self._adaptive_threshold()
            v_t = time.time()
            work_ctx = {**ctx, "goal": {"goal_text": proposal.query}, "solution": sol_d}
            final_score, dim_scores = self._score_with_container(work_ctx, solution.answer or " ")
            verification = Verification(
                is_valid=bool(final_score >= self.threshold and bool((solution.answer or "").strip())),
                score=float(final_score),
                dimension_scores=dim_scores,
                evidence_count=int(len(solution.evidence or [])),
                reasoning_steps=int(len(solution.reasoning_path or [])),
            )
            steps.append({
                "id": "verify-1",
                "type": "solution_verification",
                "agent": "Verifier",
                "start_time": v_t,
                "end_time": time.time(),
                "duration": time.time() - v_t,
                "input_size": len((solution.answer or "")),
                "scores": {"final": final_score, **dim_scores},
                "threshold": self.threshold,
                "status": "completed",
            })

            # POST: reward + curriculum ------------------------------------
            success: bool = bool(verification.is_valid)
            ext_reward = float(verification.score)
            nov_reward = float(self._compute_novelty(solution.answer or ""))
            total_reward = float(self.rewards.blend(ext_reward, 0.0, nov_reward))

            self.success_hist.append(1 if success else 0)
            self.curriculum.update(ext_reward, success)

            if callable(getattr(self.curriculum, "difficulty", None)):
                difficulty = float(self.curriculum.difficulty())
            else:
                difficulty = float(getattr(self.curriculum, "_d", 0.3))

            sr = getattr(self.curriculum, "success_rate", None)
            if sr is None and hasattr(self.curriculum, "snapshot"):
                try:
                    sr = float(self.curriculum.snapshot().get("success_rate", 0.0))
                except Exception:
                    sr = 0.0
            success_rate = float(sr or 0.0)

            result = {
                "episode_id": str(run_id),
                "success": success,
                "metrics": {
                    "reward": total_reward,
                    "verification": float(verification.score),
                    "novelty": nov_reward,
                    "success_rate": success_rate,
                    "difficulty": difficulty,
                    "threshold": self.threshold,
                    "duration_ms": int((time.time() - t0) * 1000),
                    "dimensions": verification.dimension_scores,
                },
                "plan_trace_id": str(run_id),
            }

            await self._publish("ssp.episode.completed", {
                **result,
                "at": _now_iso(),
                "steps": steps,
                "proposal": {"query": proposal.query, "difficulty": proposal.difficulty},
                "solution": {"preview": (solution.answer or "")[:200]},
            })
            if self.logger:
                self.logger.log("SSPEpisodeCompleted", {"run_id": run_id, "success": success})

            return result

        except Exception as e:
            tb = traceback.format_exc()
            await self._publish("ssp.episode.failed", {
                "episode_id": str(run_id),
                "at": _now_iso(),
                "error": {"type": type(e).__name__, "message": str(e)},
                "traceback": tb,
                "steps": steps,
            })
            if self.logger:
                self.logger.log("SSPEpisodeFailed", {"run_id": run_id, "error": str(e)})
            raise
