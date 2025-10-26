# stephanie/components/ssp/verifier.py
from __future__ import annotations

import time
from dataclasses import asdict
from typing import Dict, Any
from stephanie.components.ssp.types import Verification
from stephanie.components.ssp.verifier_rules import basic_proposal_checks
from stephanie.components.ssp.util import (
    get_trace_logger,
    PlanTrace_safe,
    MemCubeSafe,
)
from stephanie.utils.json_sanitize import sanitize


class Verifier:
    def __init__(self, cfg):
        self.cfg = cfg.self_play
        self.trace_logger = get_trace_logger()
        self.memcube = MemCubeSafe()

    def verify_proposal(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        res = basic_proposal_checks(
            self.memcube, proposal, threshold=self.cfg.verification_threshold
        )
        tr = PlanTrace_safe(
            trace_id=f"verif-prop-{int(time.time() * 1000) % 1000000}",
            role="verifier",
            goal="proposal",
            status="completed",
            metadata={"score": res["score"]},
            input=str(proposal),
            output=f"proposal {'valid' if res['can_verify'] else 'invalid'}",
            artifacts=res,
        )
        self.trace_logger.log(tr)
        return res

    def verify_solution(self, solution: Dict[str, Any]) -> Dict[str, Any]:
        coh = 0.75
        nov = 0.6
        cau = 0.7
        cons = 0.72
        weights = {
            "coherence": 0.3,
            "novelty": 0.2,
            "causality": 0.25,
            "consistency": 0.25,
        }
        dims = {
            "coherence": coh,
            "novelty": nov,
            "causality": cau,
            "consistency": cons,
        }
        score = sum(dims[k] * w for k, w in weights.items())
        is_valid = score >= self.cfg.verification_threshold
        ver = Verification(
            is_valid=is_valid,
            score=score,
            dimension_scores=dims,
            evidence_count=len(solution.get("evidence", [])),
            reasoning_steps=len(solution.get("reasoning_path", [])),
        )
        tr = PlanTrace_safe(
            trace_id=f"verif-sol-{int(time.time() * 1000) % 1000000}",
            role="verifier",
            goal="solution",
            status="completed",
            metadata={"final_score": score, "is_valid": is_valid},
            input=str(solution),
            output=f"score={score:.2f}",
            artifacts=sanitize(asdict(ver)),
        )
        self.trace_logger.log(tr)
        return asdict(ver)
