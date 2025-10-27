# stephanie/components/ssp/engine.py

from __future__ import annotations
import time
from typing import Dict, Any, Optional
from omegaconf import DictConfig
from stephanie.utils.json_sanitize import sanitize
from stephanie.data.plan_trace import PlanTrace
from stephanie.utils.trace_logger import trace_logger
from .solver_adapter import TreeSolverAdapter
from .grpo_bridge import make_tree_grpo_adapter, rollout_training_batch

class SSPEngine:
    """SSP component orchestrator: Propose → Solve(Tree) → Verify → Train(GRPO) → Jitter feed."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.enabled = bool(cfg.ssp.enabled)
        self.base_agent = None # get_model("solver")  # shared LLM/tooling interface
        self.solver = TreeSolverAdapter(cfg, self.base_agent)
        self.proposer = None # get_model(cfg.ssp.proposer.model)
        self.adapter = make_tree_grpo_adapter(cfg, self.base_agent)
        self._running = False
        self._last_tick = 0.0
        self._episode = 0

    async def start(self) -> None:
        if not self.enabled:
            return
        self._running = True
        trace_logger.log(PlanTrace(
            trace_id="ssp-comp-start",
            role="system",
            goal="ssp",
            status="started",
            metadata={"cfg": sanitize(dict(self.cfg.ssp))},
            input="", output="SSP started", artifacts={}
        ))

    async def stop(self) -> None:
        self._running = False
        trace_logger.log(PlanTrace(
            trace_id="ssp-comp-stop",
            role="system",
            goal="ssp",
            status="completed",
            metadata={}, input="", output="SSP stopped", artifacts={}
        ))

    async def step(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """One SSP step: propose → solve (tree) → (optionally) rollout GRPO batch."""
        if not self._running:
            return {"status": "idle"}

        ctx = context or {}
        seed = ctx.get("seed", "Propose a concrete, verifiable research task about our system’s reasoning quality.")
        # Propose
        prompt = (
            f"Generate a verifiable task at difficulty ~0.3.\n"
            f"Return: Research Question + how to verify."
        )
        proposal = str(self.proposer(prompt))
        trace_logger.log(PlanTrace(
            trace_id=f"proposer-{abs(hash(proposal)) % 1_000_000}",
            role="proposer", goal="proposal", status="proposed",
            metadata={}, input=prompt, output=proposal, artifacts={}
        ))

        # Solve (Tree)
        sol = await self.solver.solve(proposal, {"task_type": self.cfg.ssp.solver.task_type})

        # Optional training batch (Tree-GRPO)
        batch = await rollout_training_batch(self.adapter, goal_text=proposal, value=0.0)

        self._episode += 1
        return sanitize({
            "episode": self._episode,
            "proposal": proposal,
            "solution": sol,
            "training_batch": batch,
        })

    # ---------- Jitter substrate ----------
    async def jitter_tick(self) -> Dict[str, Any]:
        if not self._running:
            return {"status": "inactive"}

        now = time.time()
        if (now - self._last_tick) < float(self.cfg.ssp.jitter.tick_interval_sec):
            return {"status": "waiting"}

        self._last_tick = now
        # Minimal SCM/epistemic stub; swap with real HRM/MARS/VPM later.
        payload = {
            "vpm": {"heatmap": [[0.2,0.4],[0.6,0.9]], "meta": {"shape": [2,2]}},
            "scm": {"coherence": 0.74, "novelty": 0.61, "complexity": 0.48},
            "epistemic": {"success_rate": 0.62, "difficulty": 0.30}
        }
        trace_logger.log(PlanTrace(
            trace_id=f"jitter-bridge-{self._episode}",
            role="jitter", goal="substrate", status="completed",
            metadata={}, input="", output="tick", artifacts=sanitize(payload)
        ))
        return sanitize(payload)

    def status(self) -> Dict[str, Any]:
        return {"status": "running" if self._running else "stopped",
                "episode_count": self._episode,
                "tick_interval": float(self.cfg.ssp.jitter.tick_interval_sec)}
