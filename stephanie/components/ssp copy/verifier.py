from __future__ import annotations
from typing import Dict, Any
from omegaconf import DictConfig, OmegaConf

from stephanie.components.ssp.config import ensure_cfg
from stephanie.components.ssp.util import get_trace_logger, PlanTrace_safe
from stephanie.utils.json_sanitize import sanitize

def _resolve_threshold(sp_cfg: DictConfig) -> float:
    t = OmegaConf.select(sp_cfg, "verification_threshold")
    if t is None:
        t = OmegaConf.select(sp_cfg, "verifier.verification_threshold")
    return float(t) if t is not None else 0.85

class Verifier:
    def __init__(self, cfg: DictConfig | dict):
        root = cfg
        self.root = root
        self.sp = root.self_play
        self.cfg = self.sp.verifier
        self.threshold = _resolve_threshold(self.cfg)
        self.trace_logger = get_trace_logger()

    def verify_proposal(self, proposal: Dict[str, Any], threshold: float | None = None) -> Dict[str, Any]:
        thr = float(threshold) if threshold is not None else self.threshold

        has_verification = bool(proposal.get("verification_approach"))
        has_connections = bool(proposal.get("connections"))
        rationale = proposal.get("difficulty_rationale") or ""
        has_rationale = len(rationale) > 20

        score = 0.3 * has_verification + 0.3 * has_connections + 0.4 * has_rationale
        can_verify = score >= thr

        result = {
            "can_verify": can_verify,
            "score": float(score),
            "threshold": thr,
            "verification_checks": {
                "has_verification": has_verification,
                "has_connections": has_connections,
                "has_difficulty_rationale": has_rationale,
            },
        }

        self.trace_logger.log(PlanTrace_safe(
            trace_id=f"verif-prop-{abs(hash(proposal.get('query',''))) % 1_000_000}",
            role="verifier",
            goal="proposal verification",
            status="completed",
            metadata={"score": score, "threshold": thr},
            input=proposal.get("query", ""),
            output="valid" if can_verify else "invalid",
            artifacts=sanitize(result),
        ))
        return result

    def verify_solution(self, solution: Dict[str, Any], threshold: float | None = None) -> Dict[str, Any]:
        thr = float(threshold) if threshold is not None else self.threshold

        answer = solution.get("answer", "")
        reasoning_path = solution.get("reasoning_path", []) or []
        evidence = solution.get("evidence", []) or []

        coherence = 0.2 + min(0.8, len(reasoning_path) / 10.0)
        consistency = 0.2 + min(0.8, len(evidence) / 10.0)
        novelty = 0.4  # placeholder

        weights = self.cfg.verifier.get("hrms", [
            {"name": "coherence", "weight": 0.3},
            {"name": "novelty", "weight": 0.2},
            {"name": "causality", "weight": 0.25},
            {"name": "consistency", "weight": 0.25},
        ])
        dims = {"coherence": coherence, "novelty": novelty, "causality": coherence, "consistency": consistency}
        total = sum(float(d.get("weight", 0.25)) * float(dims.get(d.get("name"), 0.5)) for d in weights)
        denom = sum(float(d.get("weight", 0.25)) for d in weights)
        final_score = total / max(1e-9, denom)

        is_valid = final_score >= thr
        result = {
            "is_valid": is_valid,
            "score": float(final_score),
            "threshold": thr,
            "dimension_scores": dims,
            "evidence_count": len(evidence),
            "reasoning_steps": len(reasoning_path),
        }

        self.trace_logger.log(PlanTrace_safe(
            trace_id=f"verif-sol-{abs(hash(answer[:48])) % 1_000_000}",
            role="verifier",
            goal="solution verification",
            status="completed",
            metadata={"final_score": final_score, "threshold": thr, "is_valid": is_valid},
            input={"len_reasoning": len(reasoning_path), "len_evidence": len(evidence)},
            output=f"{final_score:.3f}",
            artifacts=sanitize(result),
        ))
        return result
