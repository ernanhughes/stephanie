# stephanie/components/ssp/reward/heads/naive_quarkish.py
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple


def _tokens(s: str):
    return [t.lower() for t in (s or "").split()]


def _f1(a: str, b: str):
    A, B = set(_tokens(a)), set(_tokens(b))
    if not A or not B:
        return 0.0
    p = len(A & B) / max(len(B), 1)
    r = len(A & B) / max(len(A), 1)
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)


def _coverage(response: str, evidence: list[str]):
    if not evidence:
        return 0.0
    R = set(_tokens(response))
    if not R:
        return 0.0
    covs = []
    for e in evidence:
        E = set(_tokens(e))
        covs.append(len(R & E) / max(len(E), 1))
    return sum(covs) / len(covs)


class NaiveQuarkishReward:
    def __init__(self, w_f1=0.5, w_cov=0.3, w_len=0.2, target_len=80):
        self.w_f1, self.w_cov, self.w_len, self.target_len = (
            w_f1,
            w_cov,
            w_len,
            target_len,
        )

    def score(
        self,
        *,
        prompt: str,
        response: str,
        ground_truth: str = "",
        meta: Dict[str, Any] | None = None,
    ) -> Dict[str, float]:
        meta = meta or {}
        f1 = _f1(ground_truth or prompt, response)  # proxy if GT missing
        cov = _coverage(response, meta.get("evidence_docs") or [])
        # smooth length reward around target_len
        L = len(response.split())
        len_r = math.exp(-abs(L - self.target_len) / max(self.target_len, 1))
        # weighted
        reward = self.w_f1 * f1 + self.w_cov * cov + self.w_len * len_r
        # clamp into [0,1]
        reward = max(0.0, min(1.0, reward))
        return {
            "reward": reward,
            "f1": f1,
            "coverage": cov,
            "len_reward": len_r,
            "resp_len": float(L) / 256.0,  # handy extra for VPM
        }

    def compute_reward(
        self,
        *,
        question: str,
        predicted_answer: str,
        seed_answer: Optional[str],
        evidence_docs: List[str],
        meta_in: Dict[str, Any] | None = None,
    ) -> Tuple[float, Dict[str, float], Dict[str, Any]]:
        """
        Returns: (reward, feature_dict, out_meta_additions)
        """
        meta_in = dict(meta_in or {})
        meta_in["evidence_docs"] = evidence_docs
        # Favor the SSP adapter if present
        return self.score(
            prompt=question,
            response=predicted_answer,
            ground_truth=seed_answer,
            meta=meta_in,
        )
