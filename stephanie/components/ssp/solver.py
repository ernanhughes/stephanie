# stephanie/components/ssp/solver.py
from __future__ import annotations

import time
from dataclasses import asdict
from typing import Dict, Any
from stephanie.components.ssp.tree_events import TreeEventEmitter
from stephanie.components.ssp.types import Solution
from stephanie.components.ssp.util import get_model_safe, get_trace_logger, PlanTrace_safe
from omegaconf import DictConfig
from stephanie.components.ssp.config import ensure_cfg
from stephanie.utils.json_sanitize import sanitize

class Solver:
    def __init__(self, cfg: DictConfig | dict):
        root = ensure_cfg(cfg)
        self.root = root
        self.sp = root.self_play
        self.cfg = self.sp.solver
        self.model = get_model_safe("solver")
        self.tree = TreeSearchPipeline(root, event_emitter=TreeEventEmitter())
        self.trace_logger = get_trace_logger()

    def solve(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        goal = proposal["query"]
        result = self.tree.execute(
            goal=goal,
            max_depth=int(self.cfg.search_depth),
            reasoning_budget=int(self.cfg.reasoning_budget)
        )
        # result = { "answer": str, "reasoning_path": [...], "evidence": [...], "search_depth": int }
        self.trace_logger.log(PlanTrace_safe(
          trace_id=f"solver-{abs(hash(goal)) % 1_000_000}",
          role="solver", goal=goal, status="completed",
          metadata={"search_depth": result.get("search_depth", 0)},
          input=goal, output=result.get("answer",""), artifacts=sanitize(result)
        ))
        return result