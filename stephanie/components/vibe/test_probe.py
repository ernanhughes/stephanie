# stephanie/components/vibe/test_probe.py
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums: probe kind & evaluation strategy
# ---------------------------------------------------------------------------

class ProbeKind(str, Enum):
    """
    Whether this probe is:
      - F2P (feature-to-probe): new capability that SHOULD change/activate.
      - P2P (probe-to-probe): existing behavior that MUST remain stable.
    """

    F2P = "f2p"
    P2P = "p2p"


class ProbeType(str, Enum):
    """
    How this probe is evaluated.

    We don't hard-code execution logic here; we just describe the intention.
    A higher-level executor (e.g., your test harness) decides how to run it.
    """

    # Direct execution: run code or function and check outputs
    EXECUTION = "execution"

    # LLM-based judging: ask an LLM if behavior is acceptable
    LLM_JUDGE = "llm_judge"

    # Text / log inspection: regex / simple predicate over text
    TEXT_MATCH = "text_match"

    # Query-based: e.g., ask a model a question and compare answer
    QA_MATCH = "qa_match"

    # Custom / domain-specific executor
    CUSTOM = "custom"


# ---------------------------------------------------------------------------
# Predicate / expectation types
# ---------------------------------------------------------------------------

PredicateFn = Callable[[Any], bool]
AsyncPredicateFn = Callable[[Any], Awaitable[bool]]


@dataclass
class ExpectedOutcome:
    """
    Declarative description of what “pass” means for this probe.

    Only ONE of these should typically be used per probe, but we keep them
    all here and let your executor decide which fields to honor.
    """

    # For EXECUTION / QA_MATCH: exact or approximate expected value
    expected_value: Any = None

    # For TEXT_MATCH: required substring / regex pattern
    contains: Optional[str] = None
    not_contains: Optional[str] = None
    regex: Optional[str] = None

    # For LLM_JUDGE / CUSTOM: natural-language success criteria
    natural_language_criteria: Optional[str] = None

    # For CUSTOM / PROGRAMMATIC checks
    # (you will generally inject these at runtime, not via config)
    predicate: Optional[PredicateFn] = field(default=None, repr=False)
    async_predicate: Optional[AsyncPredicateFn] = field(default=None, repr=False)

    # Tolerance for numeric comparisons, etc.
    tolerance: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize only the safe/serializable fields.
        Callable predicates are intentionally excluded.
        """
        return {
            "expected_value": self._safe_serialize(self.expected_value),
            "contains": self.contains,
            "not_contains": self.not_contains,
            "regex": self.regex,
            "natural_language_criteria": self.natural_language_criteria,
            "tolerance": self.tolerance,
        }

    @staticmethod
    def _safe_serialize(x: Any) -> Any:
        try:
            json.dumps(x, default=str)
            return x
        except Exception:
            return str(x)


# ---------------------------------------------------------------------------
# TestProbe
# ---------------------------------------------------------------------------

@dataclass
class TestProbe:
    """
    Atomic functional probe for a VibeFeatureTask.

    A TestProbe does NOT implement execution itself — it is a declarative
    descriptor that a higher-level TestRunner/TestHarness uses to:
      - execute the right thing (code, model, query, etc.)
      - evaluate the result against ExpectedOutcome
      - produce a scalar pass/fail or score in [0, 1] or [0, 100].

    Typical usages:
      - For code changes:
          - F2P: "Given input X, the new function should support feature Y."
          - P2P: "Given input Z, existing behavior must remain unchanged."

      - For reasoning/writing:
          - F2P: "Answer should now mention concept C and explain reason R."
          - P2P: "Still correctly handles edge case E."

    Fields:
      - id: unique identifier for this probe within a feature task
      - kind: F2P or P2P
      - probe_type: how to evaluate it (EXECUTION, LLM_JUDGE, etc.)
      - description: human-readable what/why
      - input_payload: arbitrary data that the executor understands
      - expected: ExpectedOutcome describing pass conditions
      - weight: importance of this probe when aggregating scores
      - tags: arbitrary labels (domain, feature name, risk level, etc.)
      - meta: free-form extra metadata
    """

    id: str
    kind: ProbeKind
    probe_type: ProbeType
    description: str

    # Arbitrary payload for the executor (e.g., function name + args,
    # QA pair, request template, etc.)
    input_payload: Dict[str, Any]

    expected: ExpectedOutcome

    weight: float = 1.0
    tags: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    # Optional: whether failure of this probe is critical (hard fail)
    critical: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """
        JSON-serializable representation, suitable for logging / storage.
        """
        return {
            "id": self.id,
            "kind": self.kind.value,
            "probe_type": self.probe_type.value,
            "description": self.description,
            "input_payload": self._safe_payload(),
            "expected": self.expected.to_dict(),
            "weight": self.weight,
            "tags": list(self.tags),
            "meta": self.meta,
            "critical": self.critical,
        }

    def short_label(self, max_len: int = 120) -> str:
        base = f"{self.kind.value.upper()}:{self.description}"
        return base if len(base) <= max_len else base[: max_len - 3] + "..."

    def _safe_payload(self) -> Dict[str, Any]:
        """
        Ensure input_payload is safe to serialize to JSON.
        """
        try:
            json.dumps(self.input_payload, default=str)
            return self.input_payload
        except Exception:
            return {
                "raw": str(self.input_payload),
                "note": "input_payload was not fully JSON-serializable; stored as string",
            }


# ------------------------------------------
