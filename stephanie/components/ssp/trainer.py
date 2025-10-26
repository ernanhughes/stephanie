# stephanie/components/ssp/trainer.py
from __future__ import annotations

import time
from typing import Optional, Dict, Any
from stephanie.components.ssp.proposer import Proposer
from stephanie.components.ssp.solver import Solver
from stephanie.components.ssp.verifier import Verifier
from stephanie.components.ssp.buffer import EpisodeBuffer
from stephanie.components.ssp.rewards import epistemic_reward
from stephanie.components.ssp.util import (
    get_trace_logger,
    VPMEvolverSafe,
    PlanTrace_safe,
)
from stephanie.utils.json_sanitize import sanitize
from omegaconf import DictConfig
from stephanie.components.ssp.config import ensure_cfg


class Trainer:
    def __init__(self, cfg: DictConfig | dict):
        cfg = ensure_cfg(cfg)
        self.cfg = cfg                  # keep the full, merged config
        self.sp  = cfg.self_play
        self.trace_logger = get_trace_logger()
        self.proposer = Proposer(cfg)
        self.solver = Solver(cfg)
        self.verifier = Verifier(cfg)
        self.buffer = EpisodeBuffer(capacity=4096)
        self.vpm = VPMEvolverSafe(cfg)
        self.success_history: list[int] = []
        self.step = 0
        self.hrm = lambda v: v
        self.mars = lambda v: v

    def train_step(self) -> Dict[str, Any]:
        self.step += 1
        context = {
            "recent_success_rate": (sum(self.success_history)/len(self.success_history)) if self.success_history else 0.0
        }
        prop = self.proposer.generate(context)
        pchk = self.verifier.verify_proposal(prop)

        if not pchk["can_verify"]:
            ver = {"score": 0.0, "is_valid": False}
            sol = {"answer": "", "reasoning_path": [], "evidence": []}
        else:
            sol = self.solver.solve(prop)
            ver = self.verifier.verify_solution(sol)

        vpm_b = self.vpm.get_current_state()["tensor"]
        vpm_a = self.vpm.evolve_once(vpm_b)
        # important: pass self-play cfg to reward
        reward, rb = epistemic_reward(self.hrm, self.mars, vpm_b, vpm_a, ver.get("score", 0.0), self.sp)

        success = 1 if ver.get("is_valid", False) else 0
        self.success_history.append(success)
        if len(self.success_history) > int(self.sp.qmax.competence_window):
            self.success_history.pop(0)

        sr = (sum(self.success_history) / len(self.success_history)) if self.success_history else 0.0
        self.proposer.update_difficulty(sr)

        tr = PlanTrace_safe(
            trace_id=f"ssp-train-{int(time.time() * 1000) % 1000000}",
            role="trainer",
            goal="self-play step",
            status="completed",
            metadata={"success": success, "score": ver.get("score", 0.0), "success_rate": sr},
            input="",
            output="ok",
            artifacts=sanitize({"proposal": prop, "solution": sol, "verification": ver}),
        )
        self.trace_logger.log(tr)
        return {"success": success, "score": ver.get("score", 0.0), "success_rate": sr}

    def run_continuous(self, max_steps: Optional[int] = None):
        i = 0
        while max_steps is None or i < max_steps:
            try:
                self.train_step()
            except Exception as e:
                tr = PlanTrace_safe(
                    trace_id=f"ssp-trn-err-{int(time.time() * 1000) % 1000000}",
                    role="trainer",
                    goal="self-play step",
                    status="error",
                    metadata={"error": str(e)},
                    input="",
                    output="error",
                    artifacts={},
                )
                self.trace_logger.log(tr)
            i += 1
