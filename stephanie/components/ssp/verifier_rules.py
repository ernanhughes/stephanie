# stephanie/components/ssp/verifier_rules.py
from __future__ import annotations

from typing import Dict, Any

def basic_proposal_checks(memcube, proposal: Dict[str, Any], k: int = 5, threshold: float = 0.85) -> Dict[str, Any]:
    hits = memcube.search_neighbors(proposal.get("query",""), k=k) if memcube else []
    has_neighbors = len(hits) >= 2
    plan = proposal.get("verification_approach","").lower()
    has_plan = any(w in plan for w in ["compare","measure","ablation","confidence interval","bootstrap","cross-validate","retrieve","replicate"])
    diff = float(proposal.get("difficulty", 0.5))
    sane_diff = 0.1 <= diff <= 0.95
    score = 0.4*has_neighbors + 0.4*has_plan + 0.2*sane_diff
    return {"can_verify": score >= threshold, "score": float(score),
            "checks": {"neighbors": has_neighbors, "plan": has_plan, "sane_diff": sane_diff}}
