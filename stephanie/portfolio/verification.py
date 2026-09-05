# stephanie/portfolio/verification.py
"""Deterministic verification registry (§13–§14). Verification outranks voting."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence
from uuid import uuid4

from stephanie.portfolio.disagreement import Disagreement
from stephanie.portfolio.executor import PortfolioExecution


@dataclass(frozen=True)
class VerificationResult:
    verification_id: str

    claim_id: str | None

    method: str

    passed: bool | None

    confidence: float | None = None

    evidence_ids: tuple[str, ...] = ()

    metadata: Mapping[str, Any] = field(default_factory=dict)


VerifierFn = Callable[[PortfolioExecution, Disagreement | None], VerificationResult | None]


def exact_match_verifier(expected: str) -> VerifierFn:
    def _verify(execution: PortfolioExecution, _disagreement) -> VerificationResult:
        passed = expected.strip().lower() in (execution.output_text or "").strip().lower()
        return VerificationResult(
            verification_id=f"ver_{uuid4().hex[:10]}",
            claim_id=None,
            method="exact_assertion",
            passed=passed,
            confidence=1.0 if passed else 0.0,
        )

    return _verify


def json_schema_verifier(required_keys: Sequence[str]) -> VerifierFn:
    def _verify(execution: PortfolioExecution, _disagreement) -> VerificationResult:
        try:
            parsed = json.loads(execution.output_text)
            missing = [k for k in required_keys if k not in parsed]
            return VerificationResult(
                verification_id=f"ver_{uuid4().hex[:10]}",
                claim_id=None,
                method="schema_validation",
                passed=not missing,
                confidence=1.0 if not missing else 0.4,
                metadata={"missing_keys": missing},
            )
        except (json.JSONDecodeError, TypeError) as exc:
            return VerificationResult(
                verification_id=f"ver_{uuid4().hex[:10]}",
                claim_id=None,
                method="schema_validation",
                passed=False,
                confidence=0.9,
                metadata={"error": str(exc)[:120]},
            )

    return _verify


def numeric_check_verifier(pattern: str, expected: float, tolerance: float = 1e-6) -> VerifierFn:
    def _verify(execution: PortfolioExecution, disagreement) -> VerificationResult:
        match = re.search(pattern, execution.output_text or "")
        if not match:
            return VerificationResult(
                verification_id=f"ver_{uuid4().hex[:10]}",
                claim_id=disagreement.disagreement_id if disagreement else None,
                method="calculation",
                passed=None,
                confidence=0.0,
                metadata={"reason": "pattern not found"},
            )
        try:
            value = float(match.group(1))
        except (IndexError, ValueError):
            return VerificationResult(
                verification_id=f"ver_{uuid4().hex[:10]}",
                claim_id=disagreement.disagreement_id if disagreement else None,
                method="calculation",
                passed=False,
                confidence=0.8,
            )
        passed = abs(value - expected) <= tolerance
        return VerificationResult(
            verification_id=f"ver_{uuid4().hex[:10]}",
            claim_id=disagreement.disagreement_id if disagreement else None,
            method="calculation",
            passed=passed,
            confidence=0.95,
            metadata={"observed": value, "expected": expected},
        )

    return _verify


class VerifierRegistry:
    """Consumers register verifiers; the runtime runs them all (§14)."""

    def __init__(self) -> None:
        self._verifiers: list[VerifierFn] = []

    def register(self, verifier: VerifierFn) -> None:
        self._verifiers.append(verifier)

    def verify(
        self,
        executions: Sequence[PortfolioExecution],
        disagreements: Sequence[Disagreement] | None = None,
    ) -> list[VerificationResult]:
        results: list[VerificationResult] = []
        disagreements = list(disagreements or [None])
        for execution in executions:
            if not execution.success:
                continue
            for verifier in self._verifiers:
                for disagreement in disagreements:
                    try:
                        result = verifier(execution, disagreement)
                    except Exception:
                        continue
                    if result is not None:
                        from dataclasses import replace

                        results.append(
                            replace(
                                result,
                                metadata={**(result.metadata or {}),
                                          "candidate_id": execution.candidate_id},
                            )
                        )
        return results
