# stephanie/components/ssp/services/vpm_control_service.py
from __future__ import annotations

import os
import time
from dataclasses import asdict
from typing import Any, Callable, Dict, Optional

from stephanie.logging.json_logger import JSONLogger
from stephanie.memory.memory_tool import MemoryTool
from stephanie.services.service_protocol import Service
# Reuse your existing controller (as shared earlier)
from stephanie.zeromodel.vpm_controller import (Decision, Policy, Signal,
                                                Thresholds, VPMController,
                                                VPMRow)


class VPMControlService(Service):
    """
    Policy/Decision service (EDIT / RESAMPLE / ESCALATE / STOP / SPINOFF / HOLD)
    driven by metric trends. Stateless API on top of per-unit VPMController instances.
    """

    def __init__(self, cfg: Dict[str, Any], memory: MemoryTool, logger: JSONLogger, container=None):
        self.cfg = cfg or {}
        self.memory = memory
        self.logger = logger
        self.container = container
        self._initialized = False

        self._controllers: Dict[str, VPMController] = {}
        self._bus = getattr(memory, "bus", None)

        # --- thresholds/policy from config (vpm_control: {...}) ---
        c = (self.cfg.get("vpm_control") or {})
        self._thr_code = Thresholds(
            mins=c.get("mins_code", {
                "tests_pass_rate": 1.0,
                "coverage": 0.70,
                "type_safe": 1.0,
                "lint_clean": 1.0,
                "complexity_ok": 0.8,
            }),
            stop_margin=float(c.get("stop_margin_code", 0.0)),
            edit_margin=float(c.get("edit_margin_code", 0.0)),
        )
        self._thr_text = Thresholds(
            mins=c.get("mins_text", {
                "coverage": 0.80,
                "correctness": 0.75,
                "coherence": 0.75,
                "citation_support": 0.65,
                "entity_consistency": 0.80,
            }),
            stop_margin=float(c.get("stop_margin_text", 0.02)),
            edit_margin=float(c.get("edit_margin_text", 0.01)),
        )
        self._policy = Policy(
            window=int(c.get("window", 5)),
            ema_alpha=float(c.get("ema_alpha", 0.4)),
            edit_margin=float(c.get("edit_margin", 0.05)),
            patience=int(c.get("patience", 3)),
            escalate_after=int(c.get("escalate_after", 2)),
            oscillation_window=int(c.get("oscillation_window", 6)),
            oscillation_threshold=int(c.get("oscillation_threshold", 3)),
            cooldown_steps=int(c.get("cooldown_steps", 1)),
            spinoff_dim=str(c.get("spinoff_dim", "novelty")),
            stickiness_dim=str(c.get("stickiness_dim", "stickiness")),
            spinoff_gate=tuple(c.get("spinoff_gate", (0.75, 0.45))),
            max_regressions=int(c.get("max_regressions", 2)),
            zscore_clip_dims=list(c.get("zscore_clip_dims", ["coverage","coherence","correctness","tests_pass_rate"])),
            zscore_clip_sigma=float(c.get("zscore_clip_sigma", 3.5)),
            local_gap_dims=list(c.get("local_gap_dims", ["citation_support","entity_consistency","lint_clean","type_safe"])),
            max_steps=int(c.get("max_steps", 50)),
            goal_kind=c.get("goal_kind"),
            goal_name=c.get("goal_name"),
            goal_min_score=float(c.get("goal_min_score", 0.75)),
            goal_allow_unmet=int(c.get("goal_allow_unmet", 0)),
        )

        # optional: external bandit hooks (set later via attach_bandit)
        self._bandit_choose: Optional[Callable[[list[str]], str]] = None
        self._bandit_update: Optional[Callable[[str, float], None]] = None

        odir = c.get("state_dir", "./runs")
        os.makedirs(odir, exist_ok=True)
        self._state_dir = odir

    # ---------------- Service protocol ----------------
    def initialize(self, **kwargs) -> None:
        if self._initialized:
            return
        self._initialized = True
        self.logger.log("VPMControlServiceInit", {"state_dir": self._state_dir})

    def shutdown(self) -> None:
        self._controllers.clear()
        self._initialized = False
        self.logger.log("VPMControlServiceShutdown", {})

    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy" if self._initialized else "uninitialized",
            "units": len(self._controllers),
            "timestamp": time.time(),
        }

    @property
    def name(self) -> str:
        return "vpm-control-service"

    # ---------------- Public API ----------------
    def decide(
        self,
        unit: str,
        *,
        kind: str,                      # "text" | "code"
        dims: Dict[str, float],
        step_idx: Optional[int] = None,
        meta: Optional[Dict[str, Any]] = None,
        candidate_exemplars: Optional[list[str]] = None,
    ) -> Decision:
        """
        Feed a metric frame; get a Decision. Safe to call at any cadence.
        """
        ctrl = self._get_controller(unit)
        m = dict(meta or {})
        if candidate_exemplars:
            m["candidate_exemplars"] = candidate_exemplars

        row = VPMRow(
            unit=unit,
            kind=("code" if kind == "code" else "text"),
            timestamp=time.time(),
            step_idx=step_idx,
            dims={k: float(v) for k, v in dims.items()},
            meta=m,
        )
        dec = ctrl.add(row, candidate_exemplars=candidate_exemplars)

        # emit + persist
        self._publish(unit, dec)
        self._persist_trace(unit, row, dec)
        return dec

    def decide_many(self, unit: str, frames: list[dict]) -> list[Decision]:
        """
        Convenience for batch calls.
        frames: [{kind, dims, step_idx?, meta?, candidate_exemplars?}, ...]
        """
        out = []
        for f in frames:
            out.append(self.decide(
                unit,
                kind=f.get("kind","text"),
                dims=f.get("dims",{}),
                step_idx=f.get("step_idx"),
                meta=f.get("meta"),
                candidate_exemplars=f.get("candidate_exemplars"),
            ))
        return out

    def reset_unit(self, unit: str) -> None:
        if unit in self._controllers:
            del self._controllers[unit]
        self.logger.log("VPMControlResetUnit", {"unit": unit})

    def get_unit_state(self, unit: str) -> Dict[str, Any]:
        """
        Returns lightweight controller state (resample counts, cooldowns, last signal).
        """
        ctrl = self._controllers.get(unit)
        if not ctrl:
            return {}
        return {
            "resample_counts": dict(ctrl.resample_counts),
            "cooldown_until_step": dict(ctrl.cooldown_until_step),
            "last_signal": {k:v.name for k,v in ctrl.last_signal.items()},
        }

    def set_goal_gate(self, *, goal_kind: Optional[str], goal_name: Optional[str],
                      min_score: float = 0.75, allow_unmet: int = 0) -> None:
        """
        Update goal-awareness on-the-fly for all controllers.
        """
        self._policy.goal_kind = goal_kind
        self._policy.goal_name = goal_name
        self._policy.goal_min_score = float(min_score)
        self._policy.goal_allow_unmet = int(allow_unmet)
        for ctrl in self._controllers.values():
            ctrl.p = self._policy  # share updated policy
        self.logger.log("VPMControlGoalGateUpdated", {
            "goal_kind": goal_kind, "goal_name": goal_name,
            "min_score": min_score, "allow_unmet": allow_unmet
        })

    def attach_bandit(self, choose_fn: Callable[[list[str]], str], update_fn: Callable[[str, float], None]) -> None:
        """
        Attach bandit hooks (optional). Applied to subsequently-created controllers.
        """
        self._bandit_choose = choose_fn
        self._bandit_update = update_fn
        self.logger.log("VPMControlBanditAttached", {})

    # ---------------- Internals ----------------
    def _get_controller(self, unit: str) -> VPMController:
        if unit not in self._controllers:
            state_path = os.path.join(self._state_dir, f"vpm_state_{unit.replace(':','_')}.json")
            ctrl = VPMController(
                thresholds_code=self._thr_code,
                thresholds_text=self._thr_text,
                policy=self._policy,
                bandit_choose=self._bandit_choose,
                bandit_update=self._bandit_update,
                logger=lambda ev, d: self.logger.log(ev, d),
                state_path=state_path,
            )
            self._controllers[unit] = ctrl
        return self._controllers[unit]

    def _publish(self, unit: str, dec: Decision) -> None:
        try:
            if self._bus and hasattr(self._bus, "publish"):
                self._bus.publish("vpm.control.decision", {"unit": unit, **asdict(dec)})
        except Exception:
            pass

    def _persist_trace(self, unit: str, row: VPMRow, dec: Decision) -> None:
        # best-effort; works with your MemoryTool repos if present
        try:
            repo = getattr(self.memory, "plan_traces", None) or getattr(self.memory, "traces", None)
            if repo and hasattr(repo, "insert"):
                repo.insert({
                    "ts": time.time(),
                    "kind": "vpm_control_decision",
                    "unit": unit,
                    "signal": dec.signal.name,
                    "reason": dec.reason,
                    "params": dec.params,
                    "snapshot": dec.snapshot,
                })
        except Exception:
            pass
