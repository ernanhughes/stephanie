# stephanie/components/vericot/vericot_verifier.py
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple

import z3  # pip install z3-solver

from stephanie.utils.hash_utils import hash_text

Status = Literal["valid", "contradiction", "ungrounded", "untranslatable"]
UngroundedDetail = Literal[
    "no_premises",
    "premises_rejected",
    "premises_untranslatable",
    "premises_insufficient",
]

@dataclass
class StepVerification:
    step_idx: int
    status: Status
    ungrounded_detail: Optional[UngroundedDetail] = None
    # Telemetry / display
    fol_formula: Optional[str] = None
    premises: List[str] = field(default_factory=list)            # ONLY used premises (minimal core)
    premise_votes: List[Tuple[str, bool]] = field(default_factory=list)  # (premise, accepted_by_LLMAJ)
    llmaj_ok: Optional[bool] = None
    solver_log: Optional[str] = None
    hr_tags: List[str] = field(default_factory=list)
    elapsed_ms: float = 0.0
    unsat_core_size: int = 0  # size of minimal premise set (goal excluded)

    def __post_init__(self):
        # HRM auto-tags
        if self.status == "untranslatable":
            self.hr_tags.append(
                RiskTagger.tag(
                    reason="STEP_UNTRANSLATABLE",
                    level=RiskLevel.HIGH,
                    mitigation="HUMAN_REVIEW",
                )
            )
        if self.ungrounded_detail == "premises_rejected":
            self.hr_tags.append(
                RiskTagger.tag(
                    reason="PREMISES_REJECTED",
                    level=RiskLevel.MEDIUM,
                    mitigation="RETRY_WITH_SOURCE",
                )
            )
        # Convenience summary for dashboards
        if self.premise_votes:
            self.llmaj_ok = any(v for _, v in self.premise_votes)

@dataclass
class TraceVerification:
    steps: List[StepVerification]
    pass_rate: float
    context_size_final: int = 0
    sat_calls: int = 0
    unsat_calls: int = 0
    avg_check_ms: float = 0.0
    errors_hist: Dict[Status, int] = field(default_factory=dict)

    def __post_init__(self):
        from collections import Counter
        self.errors_hist = dict(Counter(s.status for s in self.steps))
        if self.steps:
            self.avg_check_ms = sum(s.elapsed_ms for s in self.steps) / len(self.steps)

class VeriCoTVerifier:
    """
    Production verifier with:
      - Correct contradiction / entailment checks against persistent context C
      - Minimal-premise extraction via UNSAT cores on labeled assumptions
      - Deterministic atom keys (audit-friendly)
      - Granular 'ungrounded' diagnostics for self-repair
      - HRM/VPM-ready telemetry
    """

    def __init__(
        self,
        smt_backend: str = "z3",
        enable_llmaj: bool = True,
        max_retry_translate: int = 2,
        max_retry_premise: int = 2,
        deterministic_atoms: bool = True,
        short_circuit: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.smt_backend = smt_backend
        self.enable_llmaj = enable_llmaj
        self.max_retry_translate = max_retry_translate
        self.max_retry_premise = max_retry_premise
        self.deterministic_atoms = deterministic_atoms
        self.short_circuit = short_circuit or {
            "on_contradiction": True,
            "max_ungrounded": 3,
        }
        self.logger = logger or logging.getLogger("VeriCoT")

        # Persistent solver holds the accumulated verified context C
        self._base_solver = z3.Solver()
        self._ctx_fingerprints: set[str] = set()  # dedup context asserts
        self._stats = {"sat_calls": 0, "unsat_calls": 0, "total_checks": 0}

    # ------------------------------ Public API ------------------------------

    def verify(self, cot_steps: List[str], context_docs: List[str]) -> TraceVerification:
        steps_verified: List[StepVerification] = []
        ungrounded_streak = 0

        for i, step_text in enumerate(cot_steps):
            t0 = time.perf_counter()
            sv = self._verify_single_step(step_text, i, context_docs)
            sv.elapsed_ms = (time.perf_counter() - t0) * 1000.0
            steps_verified.append(sv)

            # Grow context C ONLY with verified facts/premises
            if sv.status == "valid":
                try:
                    expr = self._autoformalize(step_text, context_docs)
                    self._add_to_context(expr)
                    for p in sv.premises:
                        pexpr = self._autoformalize(p, context_docs)
                        self._add_to_context(pexpr)
                except Exception:
                    pass

            # Short-circuits
            if sv.status == "contradiction" and self.short_circuit.get("on_contradiction", True):
                self.logger.warning(f"[VeriCoT] Short-circuit at step {i} due to contradiction")
                break

            if sv.status == "ungrounded":
                ungrounded_streak += 1
                if ungrounded_streak >= int(self.short_circuit.get("max_ungrounded", 3)):
                    self.logger.warning(f"[VeriCoT] Short-circuit at step {i} after {ungrounded_streak} ungrounded steps")
                    break
            else:
                ungrounded_streak = 0

        total = max(1, len(cot_steps))
        pass_rate = sum(s.status == "valid" for s in steps_verified) / total

        return TraceVerification(
            steps=steps_verified,
            pass_rate=pass_rate,
            context_size_final=len(self._base_solver.assertions()),
            sat_calls=self._stats["sat_calls"],
            unsat_calls=self._stats["unsat_calls"],
            avg_check_ms=(sum(s.elapsed_ms for s in steps_verified) / len(steps_verified) if steps_verified else 0.0),
        )

    # ------------------------- Core per-step verification -------------------------

    def _verify_single_step(self, step_text: str, step_idx: int, context_docs: List[str]) -> StepVerification:
        # 1) Autoformalize NL → z3 BoolRef (Horn-clause aware)
        expr = None
        last_err = None
        for _ in range(self.max_retry_translate + 1):
            try:
                expr = self._autoformalize(step_text, context_docs)
                break
            except Exception as e:
                last_err = e
        if expr is None:
            return StepVerification(step_idx, "untranslatable", solver_log=str(last_err))

        # 2) CONTRADICTION: is C ∧ expr UNSAT?
        if self._check_context_unsat([expr]):
            return StepVerification(step_idx, "contradiction", solver_log="C ∧ expr UNSAT")

        # 3) ENTAILMENT: is C ∧ ¬expr UNSAT?
        if self._check_context_unsat([z3.Not(expr)]):
            return StepVerification(step_idx, "valid")

        # 4) PREMISES
        raw_premises = self._generate_premises(step_text, context_docs)
        premise_votes: List[Tuple[str, bool]] = []
        accepted: List[str] = []
        if not raw_premises:
            return StepVerification(step_idx, "ungrounded", "no_premises", premise_votes=[])

        for p in raw_premises:
            vote = True
            if self.enable_llmaj:
                vote = self._llmaj_filter(p, context_docs)
            premise_votes.append((p, vote))
            if vote:
                accepted.append(p)

        if not accepted:
            detail = "premises_rejected" if self.enable_llmaj else "premises_untranslatable"
            return StepVerification(step_idx, "ungrounded", detail, premises=raw_premises, premise_votes=premise_votes)

        # Translate accepted premises; drop untranslatable ones
        premise_exprs: List[z3.BoolRef] = []
        kept_premises: List[str] = []
        for p in accepted:
            try:
                premise_exprs.append(self._autoformalize(p, context_docs))
                kept_premises.append(p)
            except Exception:
                # skip this premise
                pass

        if not kept_premises:
            return StepVerification(
                step_idx,
                "ungrounded",
                "premises_untranslatable",
                premises=[],
                premise_votes=premise_votes,
            )

        # 5) Minimal-premise check via UNSAT core on assumptions
        # Build labeled assumptions: (switch_i -> prem_i) and (goal_sw -> ¬expr)
        switches = [z3.Bool(f"prem_{step_idx}_{i}") for i in range(len(kept_premises))]
        goal_sw = z3.Bool(f"goal_neg_{step_idx}")
        pairs: List[Tuple[z3.BoolRef, z3.BoolRef]] = list(zip(switches, premise_exprs)) + [(goal_sw, z3.Not(expr))]

        res, core = self._check_core_on_list(pairs)
        if res == z3.unsat:
            # Extract minimal premises from core (exclude goal switch)
            used_idx = [i for i, sw in enumerate(switches) if sw in core]
            used_premises = [kept_premises[i] for i in used_idx]
            return StepVerification(
                step_idx,
                "valid",
                premises=used_premises,
                premise_votes=premise_votes,
                unsat_core_size=len(used_idx),
                solver_log=f"core={len(used_idx)}/{len(switches)}",
            )

        # Still not entailed with accepted premises
        return StepVerification(
            step_idx,
            "ungrounded",
            "premises_insufficient",
            premises=kept_premises,
            premise_votes=premise_votes,
        )

    # --------------------------- z3 helpers & context ---------------------------

    def _add_to_context(self, expr: z3.BoolRef) -> None:
        """Deduplicate and add to persistent solver context."""
        # fingerprint on s-expression (stable across runs for same formula)
        fp = expr.sexpr()
        if fp in self._ctx_fingerprints:
            return
        self._ctx_fingerprints.add(fp)
        self._base_solver.add(expr)

    def _check_context_unsat(self, formulas: List[z3.BoolRef]) -> bool:
        """Check if C ∧ formulas is UNSAT; updates stats."""
        s = z3.Solver()
        s.add(*self._base_solver.assertions())
        s.add(*formulas)
        s.set(unsat_core=True)
        res = s.check()
        self._stats["total_checks"] += 1
        if res == z3.unsat:
            self._stats["unsat_calls"] += 1
            return True
        if res == z3.sat:
            self._stats["sat_calls"] += 1
        return False  # unknown treated as not-unsat

    def _check_core_on_list(
        self,
        guarded_pairs: List[Tuple[z3.BoolRef, z3.BoolRef]],
    ) -> Tuple[z3.CheckSatResult, List[z3.BoolRef]]:
        """
        Check C with labeled assumptions: for each (switch, formula), assert Implies(switch, formula),
        then call s.check(*switches). If UNSAT, returns the UNSAT core (subset of switches).
        """
        s = z3.Solver()
        s.set(unsat_core=True)
        s.add(*self._base_solver.assertions())

        switches = [sw for sw, _ in guarded_pairs]
        for sw, f in guarded_pairs:
            s.add(z3.Implies(sw, f))

        res = s.check(*switches)
        self._stats["total_checks"] += 1
        if res == z3.unsat:
            self._stats["unsat_calls"] += 1
            return res, list(s.unsat_core())
        if res == z3.sat:
            self._stats["sat_calls"] += 1
        return res, []

    # -------------------------- NL → FOL (z3 BoolRef) --------------------------

    def _autoformalize(self, text: str, context_docs: List[str]) -> z3.BoolRef:
        """
        Lightweight rule-based translator:
          - Detect simple implication templates (if/then, because, implies)
          - Fallback to atomic proposition
        Replace with an LLM translator when ready (keeping BoolRef return).
        """
        t = text.lower()

        def atom(s: str) -> z3.BoolRef:
            return z3.Bool(self._atom_key(s))

        # Templates: "if/when/whenever/because X then Y" / "X implies Y"
        tmp = t.replace(" therefore ", " then ").replace(" so ", " then ")
        for trigger in ("if", "when", "whenever", "because"):
            if trigger in tmp and " then " in tmp:
                lhs, rhs = tmp.split(" then ", 1)
                return z3.Implies(atom(lhs.strip()), atom(rhs.strip()))
        if "implies" in tmp:
            lhs, rhs = tmp.split("implies", 1)
            return z3.Implies(atom(lhs.strip()), atom(rhs.strip()))

        # Fallback: propositional atom
        return atom(text)

    def _atom_key(self, text: str) -> str:
        """
        Deterministic, collision-resistant z3 identifier (<=50 chars).
        Always appends a short suffix to prevent collisions on similar long strings.
        """
        cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")
        base = (cleaned[:40] or "prop")
        suffix = (
            hash_text(text)[:8]
            if self.deterministic_atoms
            else f"{abs(hash(text)) % 10**6}"
        )
        return f"{base}_{suffix}"[:50]

    # -------------------------- Premises & LLMAJ stubs -------------------------

    def _generate_premises(self, step_text: str, context_docs: List[str]) -> List[str]:
        """
        Prefer source-attributed context over commonsense. Replace with your
        real retriever / attribution strategy.
        """
        if context_docs:
            return [f"From doc[0]: {context_docs[0][:160]}"]
        return ["Commonsense: All mammals breathe air."]

    def _llmaj_filter(self, premise: str, context_docs: List[str]) -> bool:
        """Accept only source-attributed premises by default."""
        return "From doc" in premise

    # ------------------------------- Optional util ------------------------------

    def compute_vcar(self, trace_verif: TraceVerification, final_answer: str, ground_truth: str) -> float:
        """
        Verified-Correct Answer Rate (requires labels):
          VCAR = 1.0 if final answer == ground truth AND all steps are verified; else 0.0
        """
        is_correct = final_answer.strip().lower() == ground_truth.strip().lower()
        is_verified = all(s.status == "valid" for s in trace_verif.steps)
        return 1.0 if (is_correct and is_verified) else 0.0


    
class RiskLevel(Enum):
    """
    Severity classes used across HRM, VPM, SICQL, and Nexus governance.
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskTag:
    """
    A structured risk signal emitted by any reasoning subsystem.
    HRM ingests this as part of the reasoning meta-trace.
    """
    id: str
    reason: str
    level: RiskLevel
    mitigation: Optional[str] = None
    created_at: float = 0.0
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["level"] = self.level.value
        return d


class RiskTagger:
    """
    Factory for generating standardized HRM risk tags.

    All agents—VERICOT, SICQLTrainer, PlanTraceScorer, NexusGraphPolicy—
    use this to log safety-relevant events in a uniform format.
    """

    @staticmethod
    def tag(
        reason: str,
        level: RiskLevel,
        mitigation: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        tag = RiskTag(
            id=str(uuid.uuid4())[:8],   # short, stable identifier for VPM tiles
            reason=reason,
            level=level,
            mitigation=mitigation,
            created_at=time.time(),
            metadata=metadata or {},
        )
        return tag.to_dict()
