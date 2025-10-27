# stephanie/components/ssp/rewards.py
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from stephanie.components.ssp.types import RewardBreakdown


def epistemic_reward(hrm_scorer, mars_scorer, vpm_before, vpm_after, verifier_score: float, cfg: Dict) -> Tuple[float, RewardBreakdown]:
    def _score(f, v):
        try:
            return float(np.mean(f(v)))
        except Exception:
            return 0.5
    h0 = _score(hrm_scorer, vpm_before); h1 = _score(hrm_scorer, vpm_after)
    m0 = _score(mars_scorer, vpm_before); m1 = _score(mars_scorer, vpm_after)
    d_hrm  = max(-1.0, min(1.0, h1 - h0))
    d_mars = max(-1.0, min(1.0, m1 - m0))
    v_bonus = max(0.0, min(1.0, float(verifier_score)))
    W = cfg.reward if hasattr(cfg, "reward") else {"w_hrm":0.6,"w_mars":0.3,"w_verifier":0.1,"length_penalty":0.0}
    if isinstance(W, dict):
        w_hrm=W.get("w_hrm",0.6); w_mars=W.get("w_mars",0.3); w_ver=W.get("w_verifier",0.1); lp=W.get("length_penalty",0.0)
    else:
        w_hrm=W.w_hrm; w_mars=W.w_mars; w_ver=W.w_verifier; lp=W.length_penalty
    reward = w_hrm*d_hrm + w_mars*d_mars + w_ver*v_bonus - lp
    rb = RewardBreakdown(d_hrm, d_mars, v_bonus, lp)
    return float(reward), rb
