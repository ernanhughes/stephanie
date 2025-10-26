# stephanie/components/ssp/solver.py
from __future__ import annotations

import time
from dataclasses import asdict
from typing import Dict, Any
from stephanie.components.ssp.types import Solution
from stephanie.components.ssp.util import get_model_safe, get_trace_logger, PlanTrace_safe
from omegaconf import DictConfig
from stephanie.components.ssp.config import ensure_cfg

class Solver:
    def __init__(self, cfg: DictConfig | dict):
        root = ensure_cfg(cfg)
        self.root = root
        self.sp = root.self_play
        self.cfg = self.sp.solver
        self.model = get_model_safe("solver")
        self.trace_logger = get_trace_logger()

    def solve(self, proposal: Dict[str,Any]) -> Dict[str,Any]:
        q = proposal["query"]
        answer = f"Hypothesis and analysis for: {q}"
        reasoning = [{"step": 1, "description": "Break problem"}, {"step": 2, "description": "Synthesize evidence"}]
        evidence = [{"source": "mem", "content": "prior observation"}]

        tr = PlanTrace_safe(
            trace_id=f"solver-{int(time.time()*1000)%1000000}", role="solver",
            goal=q, status="completed",
            metadata={"search_depth": 2, "evidence_count": len(evidence)},
            input=q, output=answer,
            artifacts={"reasoning_path": reasoning, "evidence": evidence}
        )
        self.trace_logger.log(tr)

        sol = Solution(answer=answer, reasoning_path=reasoning, evidence=evidence, search_depth=2, trace_id=tr.trace_id)
        return asdict(sol)
