# stephanie/agents/learning/strategy_manager.py
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from stephanie.data.strategy import \
    Strategy  # <- new domain dataclass (validated + normalize + to_dict)
from stephanie.utils.json_sanitize import dumps_safe

_logger = logging.getLogger(__name__)

def _log_state(prefix: str, state: Strategy):
    _logger.info("[%s] Strategy v%s | skeptic=%.2f editor=%.2f risk=%.2f threshold=%.2f",
        prefix, state.version, state.skeptic_weight, state.editor_weight,
        state.risk_weight, state.verification_threshold)

class StrategyManager:
    """
    Owns strategy knobs and orchestrates A/B via ExperimentsStore.
    - Proposes next Strategy based on avg gain
    - Registers/assigns variants with ExperimentsStore
    - Records completed trials with final performance + metrics
    - Validates and (optionally) commits winning strategy versions
    """

    EXPERIMENT_NAME = "verification_strategy"

    def __init__(self, cfg, memory, container, logger):
        self.cfg = cfg
        self.memory = memory
        self.container = container
        self.logger = logger

        # Scoping / tagging
        self.casebook_tag: str = cfg.get("casebook_action", "blog")
        self.domain: str = str(cfg.get("domain", "default")).lower()

        # Evolution thresholds
        self.min_gain: float = float(cfg.get("min_gain", 0.01))
        self.high_gain: float = float(cfg.get("high_gain", 0.03))

        # A/B validation knobs (store will also enforce)
        self.window_seconds: Optional[int] = int(cfg["ab_window_seconds"]) if "ab_window_seconds" in cfg else None
        self.min_per_arm: int = int(cfg.get("ab_min_per_arm", 8))
        self._cooldown_sec: float = float(cfg.get("ab_cooldown_sec", 1800))
        self._last_commit_ts: float = 0.0

        # Current state (load or default)
        self.state: Strategy = self._load_or_default().normalize()

        # Track last assignment per case (variant_id, experiment_id)
        self._assignments: Dict[int, Dict[str, int]] = {}

    # ---------- State I/O ----------
    def _load_or_default(self) -> Strategy:
        try:
            exp = self.memory.experiments.get_or_create_experiment(
                self.EXPERIMENT_NAME, domain=self.domain, config={"source": "strategy_manager"}
            )
            snap = self.memory.experiments.get_latest_model_snapshot(
                experiment_name=self.EXPERIMENT_NAME,
                name="learning_strategy",
                domain=self.domain,
            )
            if not snap or not snap.get("payload"):
                _logger.info("[StrategyManager] No snapshot found, using defaults")
                return Strategy()
            s = Strategy.from_dict(snap["payload"])
            _log_state("Loaded snapshot", s)
            return s
        except Exception as e:
            _logger.warning("[StrategyManager] Failed to load snapshot: %s", e)
            return Strategy()

    def get_state(self) -> Strategy:
        return self.state

    def as_dict(self) -> Dict[str, Any]:
        return self.state.to_dict()

    def record_state(self, context: Optional[Dict[str, Any]], tag: str = "pre_change"):
        payload = {**self.as_dict(), "tag": tag, "timestamp": time.time()}
        try:
            case_id = (context or {}).get("case_id")
            if case_id:
                self.memory.casebooks.add_scorable(
                    case_id=case_id,
                    role="strategy_state",
                    text=dumps_safe(payload),
                    pipeline_run_id=(context or {}).get("pipeline_run_id"),
                    meta={"tag": tag},
                )
        except Exception:
            pass

    # ---------- Proposal logic ----------
    def _propose(self, avg_gain: float) -> Strategy:
        """
        Create the next proposed Strategy based on avg gain; clamp/normalize via Strategy.normalize().
        """
        _logger.info("[StrategyManager] Proposing new strategy (avg_gain=%.4f)", avg_gain)
        s = self.state
        if avg_gain < self.min_gain:
            # shift attention to skeptic; gently trim editor/risk
            change = 0.06 if avg_gain < 0.005 else 0.03
            proposed = s.apply_changes(
                skeptic_weight=min(0.60, s.skeptic_weight + change),
                editor_weight=max(0.20, s.editor_weight - change / 2),
                risk_weight=max(0.20, s.risk_weight - change / 2),
            )
            _log_state(f"Proposed {avg_gain:.4f} < {self.min_gain:.4f}", proposed)
            return proposed.normalize()
        elif avg_gain > self.high_gain:
            # lower the bar slightly if we’re cruising (prevents over-polishing)
            proposed = s.apply_changes(
                verification_threshold=max(0.80, s.verification_threshold - 0.01)
            )
            _log_state(f"Proposed {avg_gain:.4f} > {self.min_gain:.4f}", proposed)
            return proposed.normalize()
        _log_state("Proposed", s)
        return s.normalize()

    # ---------- A/B: register variants + assign next work ----------
    def evolve(self, iterations: List[Dict[str, Any]], context: Optional[Dict[str, Any]]):
        """
        Called after a section’s verify loop. Computes avg_gain, proposes Strategy B,
        registers both variants in ExperimentsStore, and assigns A or B for the next unit.
        """
        if not iterations or len(iterations) < 2:
            self.record_state(context, "pre_change")
            return

        # compute avg_gain
        scores = [float(it.get("score", 0.0)) for it in iterations]
        gains = [scores[i] - scores[i - 1] for i in range(1, len(scores))]
        avg_gain = (sum(gains) / len(gains)) if gains else 0.0

        # breadcrumbs
        old_payload = self.as_dict()
        _log_state("Before evolve", self.state)
        self.record_state(context, "pre_change")

        # experiment record
        exp = self.memory.experiments.get_or_create_experiment(
            self.EXPERIMENT_NAME, domain=self.domain, config={"source": "strategy_manager"}
        )

        # variants: A = current strategy; B = proposed (can equal A if no change)
        prop = self._propose(avg_gain)
        va = self.memory.experiments.upsert_variant(exp.id, "A", is_control=True, payload=self.as_dict())
        vb = self.memory.experiments.upsert_variant(exp.id, "B", is_control=False, payload=prop.to_dict())
        vc = self.memory.experiments.upsert_variant(
            exp.id, "C", is_control=False,
            payload=prop.to_dict() | {"knob":"extra"}  # placeholder for 3rd arm
        )
        # assign deterministically by case_id if present
        case_id = (context or {}).get("case_id")
        chosen = self.memory.experiments.assign_variant(exp, case_id=case_id, deterministic=True)

        # save enrollment breadcrumb
        self._record_ab_enrollment(context, group=chosen.name, avg_gain=avg_gain, proposed=prop)

        _logger.info("[StrategyManager] Assigned variant=%s case_id=%s exp_id=%s",
                 chosen.name, case_id, exp.id)
        # update in-memory state if we’re assigned to B (use its payload as new knobs for NEXT unit)
        if chosen.name.upper() == "B":
            self.state = Strategy.from_dict(vb.payload or {}).normalize().apply_changes(version=self.state.version + 1)

        # remember which variant this case is on (to record trial on completion)
        if case_id is not None:
            self._assignments[int(case_id)] = {"variant_id": int(chosen.id), "experiment_id": int(exp.id)}

        # optional evolution log
        try:
            _log_state("State updated", self.state)
            if chosen.name.upper() == "B":
                self.logger.log("LfL_Strategy_Evolved(AB)", {
                    "avg_gain": round(avg_gain, 4),
                    "old": old_payload,
                    "new": self.as_dict(),
                    "experiment_id": exp.id,
                    "timestamp": time.time(),
                })
        except Exception:
            pass

    def _record_ab_enrollment(self, context: Optional[Dict[str, Any]], *, group: str, avg_gain: float, proposed: Strategy):
        payload = {
            "experiment": self.EXPERIMENT_NAME,
            "test_group": group,
            "avg_gain": float(avg_gain),
            "old_strategy": self.as_dict(),
            "new_strategy": proposed.to_dict(),
            "timestamp": time.time(),
        }
        try:
            case_id = (context or {}).get("case_id")
            if case_id:
                self.memory.casebooks.add_scorable(
                    case_id=case_id,
                    role="strategy_ab_enroll",
                    text=dumps_safe(payload),
                    pipeline_run_id=(context or {}).get("pipeline_run_id"),
                    meta={"group": group, "experiment": self.EXPERIMENT_NAME},
                )
        except Exception:
            pass

    # ---------- Section rollup + trial completion ----------
    def track_section(self, case, iterations: List[Dict[str, Any]], context: Optional[Dict[str, Any]] = None):
        """
        Roll up section metrics + complete the A/B trial for this case (if enrolled).
        """
        try:
            if not iterations:
                return

            # basic metrics
            scores = [float(it.get("score", 0.0)) for it in iterations]
            ka_flags = [bool(it.get("knowledge_applied")) for it in iterations]
            start_score = scores[0]
            final_score = scores[-1]
            gains = [scores[i] - scores[i - 1] for i in range(1, len(scores))]
            avg_gain = (sum(gains) / len(gains)) if gains else 0.0


            first_ka_idx = next((i for i, f in enumerate(ka_flags) if f), None)
            k_lift = (final_score - float(scores[first_ka_idx])) if first_ka_idx is not None else 0.0
            k_iters = sum(1 for f in ka_flags if f)

            # wall/timing if you have it in the iteration dicts
            step_secs = sum(float(it.get("elapsed_sec", 0.0)) for it in iterations)
            verify_wall = float(iterations[-1].get("verify_wall_sec", step_secs))

            _logger.info("[StrategyManager] track_section case_id=%s run_id=%s "
             "final_score=%.3f avg_gain=%.3f k_lift=%.3f iters=%d",
             getattr(case, "id", None),
             (context or {}).get("pipeline_run_id"),
             final_score, avg_gain, k_lift, len(iterations))

            # persist compact attribute for dashboards
            rollup = {
                "timestamp": time.time(),
                "run_id": (context or {}).get("pipeline_run_id"),
                "agent": "strategy_manager",
                "strategy": {
                    **self.as_dict(),
                    "domain": self.domain,
                },
                "scores": {
                    "start": round(start_score, 6),
                    "final": round(final_score, 6),
                    "total_gain": round(final_score - start_score, 6),
                    "avg_gain": round(avg_gain, 6),
                },
                "knowledge": {
                    "applied_iters": int(k_iters),
                    "first_applied_iter": int(first_ka_idx + 1) if first_ka_idx is not None else None,
                    "applied_lift": round(float(k_lift), 6),
                },
                "timing": {
                    "verify_wall_sec": round(verify_wall, 3),
                    "sum_step_secs": round(step_secs, 3),
                },
                "timeline": [
                    {"i": int(it.get("iteration", idx + 1)),
                     "s": float(it.get("score", 0.0)),
                     "ka": bool(it.get("knowledge_applied", False))}
                    for idx, it in enumerate(iterations[-24:])
                ],
            }
            self.memory.casebooks.set_case_attr(case.id, "strategy_evolution", value_json=rollup)

            # emit lightweight event
            try:
                reporter = self.container.get("reporting")
                coro = reporter.emit(
                    context=(context or {}),
                    stage="learning",
                    event="strategy.section_rollup",
                    run_id=(context or {}).get("pipeline_run_id"),
                    agent="strategy_manager",
                    case_id=case.id,
                    final_score=rollup["scores"]["final"],
                    avg_gain=rollup["scores"]["avg_gain"],
                    k_lift=rollup["knowledge"]["applied_lift"],
                    iters=len(iterations),
                    strategy_version=self.state.version,
                    domain=self.domain,
                )
                import asyncio
                loop = asyncio.get_running_loop()
                loop.create_task(coro)
            except Exception:
                pass

            # ---- COMPLETE THE TRIAL (A/B write goes to ExperimentsStore) ----
            case_id = int(getattr(case, "id", 0))
            assign = self._assignments.get(case_id)
            if assign:
                self.memory.experiments.complete_trial(
                    variant_id=assign["variant_id"],
                    case_id=case_id,
                    performance=final_score,
                    metrics={"avg_gain": avg_gain, "k_lift": k_lift},
                    tokens=(context or {}).get("tokens"),
                    cost=(context or {}).get("cost"),
                    wall_sec=(context or {}).get("wall_sec", verify_wall),
                    pipeline_run_id=(context or {}).get("pipeline_run_id"),
                    domain=self.domain,
                    meta={"section_iters": len(iterations)},
                    experiment_group=(context or {}).get("experiment_group"),     # NEW
                    tags_used=(context or {}).get("corpus_tags", []),             # NEW
                )
                # one trial per case; you can keep or clear the assignment
                # del self._assignments[case_id]

        except Exception:
            # never break the agent on telemetry
            pass

    # ---------- Validation + optional commit ----------
    def validate_ab(self, context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Ask ExperimentsStore for a robust summary over recent trials.
        Returns the raw stats (you can show these in dashboards) and emits a small event.
        """
        try:
            exp = self.memory.experiments.get_or_create_experiment(self.EXPERIMENT_NAME, domain=self.domain)
            stats = self.memory.experiments.validate_groups(
                exp.id,
                groups=context.get("experiment_groups"),   # optional, restrict to ["experimental","control","null"]
                window_seconds=self.window_seconds,
                min_per_group=self.min_per_arm,
            )
            _logger.info("[StrategyManager] validate_ab exp_id=%s stats=%s", exp.id, stats)

            if not stats:
                return None

            # Lightweight event for dashboards
            try:
                reporter = self.container.get("reporting")
                if reporter:
                    coro = reporter.emit(
                        context=(context or {}),
                        run_id=(context or {}).get("pipeline_run_id"),
                        agent="strategy_manager",
                        stage="learning",
                        event="strategy.ab_validation",
                        experiment_id=exp.id,
                        **{k: (float(v) if isinstance(v, (int, float)) else v) for k, v in stats.items()},
                    )
                    import asyncio
                    loop = asyncio.get_running_loop()
                    loop.create_task(coro)
            except Exception:
                pass

            # Structured log
            self.logger.log("StrategyAB_Validation", {
                "experiment_id": exp.id,
                "groups": stats.get("groups", {}),
                "comparisons": stats.get("comparisons", {}),
            })


            return stats
        except Exception:
            return None

    def maybe_commit_strategy(self, validation: Optional[Dict[str, Any]], context: Dict[str, Any]) -> bool:
        """
        If validation indicates B > A and cooldown/samples pass, persist the current state
        as a versioned "learning_strategy" model snapshot.
        """
        if not validation:
            return False
        now = time.time()
        if (now - self._last_commit_ts) < self._cooldown_sec:
            return False

        # Heuristic: positive delta + reasonable p-value via either test + min rel improvement
        comparisons = validation.get("comparisons", {})
        best = max(comparisons.items(), key=lambda kv: kv[1].get("delta", 0.0)) if comparisons else None

        if not best:
            return False

        delta = best[1]["delta"]
        p_ok = (
            best[1].get("p_value_welch", 1.0) <= float(self.cfg.get("max_p_value", 0.10))
            or best[1].get("p_value_mann_whitney", 1.0) <= float(self.cfg.get("max_p_value", 0.10))
        )
        rel = delta / max(1e-6, validation["groups"][best[0].split("_minus_")[0]]["mean"])
        min_rel = float(self.cfg.get("min_strategy_improvement", 0.02))

        _logger.info("[StrategyManager] maybe_commit_strategy called delta=%.4f rel=%.4f p_ok=%s",
             delta, rel, p_ok)
        
        if not (delta > 0.0 and p_ok and rel >= min_rel):
            return False
        for g, stats_g in validation.get("groups", {}).items():
            if stats_g["n"] < self.min_per_arm:
                return False



        try:
            exp = self.memory.experiments.get_or_create_experiment(
                self.EXPERIMENT_NAME, domain=self.domain, config={"source": "strategy_manager"}
            )
            self.memory.experiments.save_model_snapshot(
                experiment_id=exp.id,
                name="learning_strategy",
                version=int(self.state.version),
                payload=self.as_dict(),
                validation=validation,
                domain=self.domain,
            )
        except Exception:
            pass

        # Breadcrumb on the case (if available)
        try:
            case_id = (context or {}).get("case_id")
            if case_id:
                self.memory.casebooks.add_scorable(
                    case_id=case_id,
                    role="strategy_commit",
                    text=dumps_safe({"state": self.as_dict(), "validation": validation, "timestamp": now}),
                    pipeline_run_id=(context or {}).get("pipeline_run_id"),
                    meta={"reason": "ok"},
                )
        except Exception:
            pass

        self._last_commit_ts = now
        _log_state("Committed strategy", self.state)
        return True
