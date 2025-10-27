# stephanie/components/ssp/actors/verifier.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

from omegaconf import DictConfig, OmegaConf

from stephanie.components.ssp.util import PlanTrace_safe, get_trace_logger
from stephanie.services.service_container import ServiceContainer
from stephanie.utils.json_sanitize import sanitize

# --------------------------- helpers ---------------------------

def _select_float(cfg: DictConfig, *paths: str, default: float) -> float:
    """Try multiple OmegaConf paths; first non-None wins."""
    for p in paths:
        v = OmegaConf.select(cfg, p)
        if v is not None:
            try:
                return float(v)
            except Exception:
                pass
    return float(default)

def _resolve_threshold(root: DictConfig) -> float:
    """
    Resolve verification threshold robustly.
    Accepts either a root config or any nested view.
    Tries:
      - self_play.verification_threshold
      - self_play.verifier.verification_threshold
      - verifier.verification_threshold
    """
    return _select_float(
        root,
        "self_play.verification_threshold",
        "self_play.verifier.verification_threshold",
        "verifier.verification_threshold",
        default=0.85,
    )

def _hrm_weights(sp_verifier_cfg: DictConfig) -> List[Dict[str, Any]]:
    # Expect hrms list under self_play.verifier.hrms
    hrms = getattr(sp_verifier_cfg, "hrms", None)
    if not hrms:
        return [
            {"name": "coherence", "weight": 0.3},
            {"name": "novelty",   "weight": 0.2},
            {"name": "causality", "weight": 0.25},
            {"name": "consistency","weight":0.25},
        ]
    # normalize structure to list of dicts with name/weight
    out = []
    for d in hrms:
        name = str(d.get("name", "coherence"))
        w = float(d.get("weight", 0.25))
        out.append({"name": name, "weight": w})
    return out

def _bounded01(x: float) -> float:
    return 0.0 if x != x else max(0.0, min(1.0, float(x)))  # NaN-safe clamp


# --------------------------- Verifier ---------------------------

class Verifier:
    """
    Lightweight verifier with:
      - Proposal sanity checks (verifiable & connected)
      - Solution HRM-style dimension scores (coherence/consistency/causality/novelty)
      - Configurable threshold and min evidence count
      - JSON-safe trace logging
    """

    def __init__(self, cfg: DictConfig | dict, container: ServiceContainer):
        root = cfg
        self.root: DictConfig = root
        self.sp: DictConfig = root.self_play
        self.cfg: DictConfig = self.sp.verifier
        self.container = container

        self.threshold: float = _resolve_threshold(root)
        self.min_evidence: int = int(getattr(self.cfg, "min_evidence_count", 2))
        self.hrm_weights = _hrm_weights(self.cfg)

        self.trace_logger = get_trace_logger()

    # ------------------- proposals -------------------

    def verify_proposal(
        self,
        proposal: Dict[str, Any],
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """Check if a proposal is plausibly verifiable and well-formed."""
        thr = float(threshold) if threshold is not None else self.threshold

        query = str(proposal.get("query", "")).strip()
        vapp = str(proposal.get("verification_approach", "")).strip()
        conns = proposal.get("connections", []) or []
        diff  = proposal.get("difficulty", None)

        # simple normalized checks
        has_query = len(query) >= 8
        has_verification = len(vapp) >= 20
        has_connections = isinstance(conns, list) and len(conns) > 0
        has_difficulty = isinstance(diff, (int, float)) and (0.0 <= float(diff) <= 1.0)

        # weighted heuristic score (kept simple and transparent)
        score = (
            0.45 * (1.0 if has_verification else 0.0)
            + 0.30 * (1.0 if has_connections else 0.0)
            + 0.15 * (1.0 if has_query else 0.0)
            + 0.10 * (1.0 if has_difficulty else 0.0)
        )

        can_verify = score >= thr

        result = {
            "can_verify": bool(can_verify),
            "score": float(score),
            "threshold": float(thr),
            "checks": {
                "has_query": has_query,
                "has_verification": has_verification,
                "has_connections": has_connections,
                "has_difficulty": has_difficulty,
            },
        }

        self.trace_logger.log(PlanTrace_safe(
            trace_id=f"verif-prop-{abs(hash(query)) % 1_000_000}",
            role="verifier",
            goal="proposal verification",
            status="completed",
            metadata={"score": score, "threshold": thr},
            input=query,
            output="valid" if can_verify else "invalid",
            artifacts=sanitize(result),
        ))
        return result

    # ------------------- solutions -------------------

    def verify_solution(
        self,
        solution: Dict[str, Any],
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Score solution across HRM-inspired dimensions.
        Enforce min_evidence gate; if not met, mark invalid but still report score.
        """
        thr = float(threshold) if threshold is not None else self.threshold

        answer = str(solution.get("answer", "") or "")
        reasoning_path = list(solution.get("reasoning_path", []) or [])
        evidence = list(solution.get("evidence", []) or [])

        # --- dimension scores (simple, deterministic baselines) ---
        coherence = self._score_coherence(answer, reasoning_path)      # 0..1
        consistency = self._score_consistency(answer, evidence)        # 0..1
        causality = self._score_causality(reasoning_path)              # 0..1
        novelty = self._score_novelty(answer, evidence)                # 0..1

        dims = {
            "coherence": coherence,
            "consistency": consistency,
            "causality": causality,
            "novelty": novelty,
        }

        # weighted aggregation
        num = 0.0
        den = 0.0
        for w in self.hrm_weights:
            name = w["name"]
            wt = float(w["weight"])
            num += wt * float(dims.get(name, 0.5))
            den += wt
        final_score = num / max(den, 1e-9)

        # policy gates
        meets_evidence = len(evidence) >= self.min_evidence
        is_valid = (final_score >= thr) and meets_evidence

        result = {
            "is_valid": bool(is_valid),
            "score": float(final_score),
            "threshold": float(thr),
            "dimension_scores": {k: float(v) for k, v in dims.items()},
            "evidence_count": int(len(evidence)),
            "reasoning_steps": int(len(reasoning_path)),
            "policy_overrides": {
                "min_evidence_met": bool(meets_evidence),
                "min_evidence_required": int(self.min_evidence),
            },
        }

        self.trace_logger.log(PlanTrace_safe(
            trace_id=f"verif-sol-{abs(hash(answer[:64])) % 1_000_000}",
            role="verifier",
            goal="solution verification",
            status="completed",
            metadata={
                "final_score": final_score,
                "threshold": thr,
                "is_valid": is_valid,
                "min_evidence_met": meets_evidence,
            },
            input={"len_reasoning": len(reasoning_path), "len_evidence": len(evidence)},
            output=f"{final_score:.3f}",
            artifacts=sanitize(result),
        ))
        return result

    # ---------------- dimension baselines ----------------

    def _score_coherence(self, answer: str, reasoning_path: List[Dict[str, Any]]) -> float:
        """
        Heuristic: more structured steps → higher baseline.
        Reward light referencing of step descriptions inside answer.
        """
        if not reasoning_path:
            return 0.25
        # step count contribution
        sc = min(1.0, 0.2 + 0.08 * len(reasoning_path))  # saturate around ~10–12 steps
        # reference match contribution
        a_low = answer.lower()
        refs = 0
        for step in reasoning_path[:8]:  # limit shallow scan
            desc = str(step.get("description", "")).lower()
            if desc and desc[:24] in a_low:
                refs += 1
        refc = min(0.8, refs * 0.15)
        return _bounded01(0.5 * sc + 0.5 * (0.3 + refc))

    def _score_consistency(self, answer: str, evidence: List[Dict[str, Any]]) -> float:
        """
        Heuristic: count how many evidence snippets are echoed in the answer (substring hit).
        """
        if not evidence:
            return 0.2
        a_low = answer.lower()
        hits = 0
        total = 0
        for ev in evidence[:10]:
            content = str(ev.get("content", "")).lower()
            if not content:
                continue
            total += 1
            if content[:40] and content[:40] in a_low:
                hits += 1
        if total == 0:
            return 0.25
        ratio = hits / total
        return _bounded01(0.3 + 0.7 * ratio)

    def _score_causality(self, reasoning_path: List[Dict[str, Any]]) -> float:
        """
        Heuristic: detect causal linking words across steps.
        """
        if len(reasoning_path) < 2:
            return 0.35
        causal_kw = ("because", "therefore", "thus", "leads to", "causes", "results in", "hence")
        count = 0
        for step in reasoning_path[1:]:
            desc = str(step.get("description", "")).lower()
            if any(k in desc for k in causal_kw):
                count += 1
        frac = count / max(1, len(reasoning_path) - 1)
        return _bounded01(0.3 + 0.7 * frac)

    def _score_novelty(self, answer: str, evidence: List[Dict[str, Any]]) -> float:
        """
        Heuristic: encourage synthesis beyond pasted evidence.
        If answer reuses many evidence substrings verbatim → lower novelty.
        """
        if not evidence:
            return 0.5  # unknown → neutral
        a_low = answer.lower()
        reused = 0
        total = 0
        for ev in evidence[:10]:
            content = str(ev.get("content", "")).lower()
            if not content:
                continue
            total += 1
            if content[:50] and content[:50] in a_low:
                reused += 1
        if total == 0:
            return 0.5
        reuse_ratio = reused / total
        # invert reuse → novelty, keep within [0.2, 1.0]
        return _bounded01(0.2 + 0.8 * (1.0 - reuse_ratio))
