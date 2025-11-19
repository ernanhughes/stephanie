# stephanie/memory/memcubes/rubric.py
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence
import datetime as _dt
import json
import logging

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RubricScope(str, Enum):
    """
    High-level domain this rubric is intended for.
    """

    WRITING = "writing"
    CODE = "code"
    REASONING = "reasoning"
    SAFETY = "safety"
    GENERAL = "general"


class CriterionType(str, Enum):
    """
    The primitive type of a rubric criterion.

    SCORE_0_100 is the default for numeric “grades”.
    """

    SCORE_0_100 = "score_0_100"   # numeric 0–100
    SCORE_0_1 = "score_0_1"       # numeric 0–1
    BOOLEAN = "boolean"           # pass/fail
    ENUM = "enum"                 # one of allowed_values
    TEXT_COMMENT = "text_comment" # free-form explanation / note


class CriterionScope(str, Enum):
    """
    What part of the target this criterion applies to.
    """

    RESPONSE = "response"         # the generated answer / artifact
    REQUEST = "request"           # the user question / task
    CONTEXT = "context"           # background info, notes, sources
    PAIR = "pair"                 # relationship between request + response
    GENERAL = "general"


# ---------------------------------------------------------------------------
# Core dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RubricCriterion:
    """
    One dimension of evaluation inside a rubric.

    Example (for writing):
      - name: "clarity"
      - type: SCORE_0_100
      - scope: RESPONSE
      - dimension: "writing_clarity"
    """

    name: str                       # internal id, e.g. "clarity"
    display_name: str               # human label, e.g. "Clarity"
    description: str                # what this criterion measures
    type: CriterionType
    scope: CriterionScope

    # Dimension key used when storing results in metrics tables / MemCubes.
    dimension: str

    # Optional weighting when aggregating multiple criteria into a single score.
    weight: float = 1.0

    # For ENUM/BOOLEAN criteria, allowed values.
    allowed_values: Optional[List[str]] = None

    # Optional default scoring hint / target (e.g. desired >= 80).
    target_min: Optional[float] = None
    target_max: Optional[float] = None

    # Optional LLM prompt hint for this criterion (overrides generic one).
    prompt_hint: Optional[str] = None

    # Arbitrary extra metadata.
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["type"] = self.type.value
        d["scope"] = self.scope.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RubricCriterion:
        return cls(
            name=data["name"],
            display_name=data.get("display_name", data["name"]),
            description=data.get("description", ""),
            type=CriterionType(data["type"]),
            scope=CriterionScope(data.get("scope", CriterionScope.RESPONSE.value)),
            dimension=data.get("dimension", data["name"]),
            weight=float(data.get("weight", 1.0)),
            allowed_values=data.get("allowed_values"),
            target_min=data.get("target_min"),
            target_max=data.get("target_max"),
            prompt_hint=data.get("prompt_hint"),
            meta=data.get("meta") or {},
        )


@dataclass
class AntiHackingCriterion:
    """
    Anti-gaming / anti-optimization instructions.

    These are combined into the LLM prompt to discourage behaviors like:
      - directly referencing the rubric
      - “roleplaying” scoring
      - gaming specific thresholds
    """

    name: str
    instruction: str
    severity: float = 1.0           # how important this is
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def default(cls) -> AntiHackingCriterion:
        """
        Generic anti-hacking instruction.
        """
        return cls(
            name="generic_anti_hacking",
            instruction=(
                "Do not optimize your answer by referring to this rubric, "
                "its scores, or how you will be graded. Focus on writing the "
                "best possible answer for the user, not on gaming the scoring."
            ),
            severity=1.0,
        )

    @classmethod
    def default_for_writing(cls) -> AntiHackingCriterion:
        """
        Anti-hacking tailored for writing tasks.
        """
        return cls(
            name="writing_anti_hacking",
            instruction=(
                "Do not mention that you are being graded or evaluated. "
                "Do not restate the rubric or talk about 'scores', 'criteria', "
                "or 'checklists'. Instead, write as a clear, honest technical "
                "explainer for the user, focused on helping them implement and "
                "understand the idea."
            ),
            severity=1.0,
        )


@dataclass
class RubricMemCube:
    """
    A structured, reusable rubric “memcube”:

      - name/version/scope: identifies the rubric
      - criteria: dimensions we’ll score
      - anti_hacking: behavioral constraints
      - meta: extra config

    This object is intentionally *evaluation-agnostic*: it does not call LLMs
    or compute scores. A separate evaluator service/agent uses it to:
      - build prompts
      - parse scores
      - store results in MemCubes / ORM tables.
    """

    name: str                       # e.g. "writing_quality_v1"
    version: str                    # e.g. "1.0"
    scope: RubricScope
    description: str

    criteria: List[RubricCriterion]
    anti_hacking: List[AntiHackingCriterion] = field(default_factory=list)

    # Optional global metadata (domain, intended_use, tags, etc.)
    meta: Dict[str, Any] = field(default_factory=dict)

    created_at: _dt.datetime = field(default_factory=lambda: _dt.datetime.utcnow())

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "scope": self.scope.value,
            "description": self.description,
            "criteria": [c.to_dict() for c in self.criteria],
            "anti_hacking": [a.to_dict() for a in self.anti_hacking],
            "meta": self.meta,
            "created_at": self.created_at.isoformat() + "Z",
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RubricMemCube:
        return cls(
            name=data["name"],
            version=data.get("version", "1.0"),
            scope=RubricScope(data.get("scope", RubricScope.GENERAL.value)),
            description=data.get("description", ""),
            criteria=[RubricCriterion.from_dict(c) for c in data.get("criteria", [])],
            anti_hacking=[
                AntiHackingCriterion(
                    name=a.get("name", "anti_hacking"),
                    instruction=a.get("instruction", ""),
                    severity=float(a.get("severity", 1.0)),
                    meta=a.get("meta") or {},
                )
                for a in data.get("anti_hacking", [])
            ],
            meta=data.get("meta") or {},
            created_at=_dt.datetime.fromisoformat(
                data.get("created_at", _dt.datetime.utcnow().isoformat())
            ),
        )

    # ------------------------------------------------------------------
    # Prompt-building helpers
    # ------------------------------------------------------------------

    def build_llm_prompt(
        self,
        target_text: str,
        request_text: Optional[str] = None,
        context_text: Optional[str] = None,
        extra_instructions: Optional[str] = None,
    ) -> str:
        """
        Build a scoring prompt for an LLM, based on this rubric.

        This is deliberately verbose and human-readable; your evaluator can
        further adapt/shorten if needed.
        """
        parts: List[str] = []

        parts.append(
            "You are an expert evaluator. Your task is to score the following "
            "text according to a rubric."
        )

        if extra_instructions:
            parts.append("\nAdditional instructions:\n" + extra_instructions)

        # Anti-hacking hints
        if self.anti_hacking:
            parts.append("\nAnti-gaming rules:")
            for ah in self.anti_hacking:
                parts.append(f"- {ah.instruction}")

        # Optional request/context
        if request_text:
            parts.append("\nUser request / question:\n" + request_text)
        if context_text:
            parts.append("\nContext / notes:\n" + context_text)

        # Target
        parts.append("\nText to evaluate:\n" + target_text)

        # Criteria table
        parts.append("\nEvaluation criteria:")
        for c in self.criteria:
            parts.append(
                f"- {c.display_name} ({c.name}): {c.description} "
                f"[type={c.type.value}, scope={c.scope.value}]"
            )

        # Output format instruction
        parts.append(
            "\nReturn your scores as a strict JSON object mapping criterion "
            "names to values. Use the following keys:\n"
            + ", ".join(repr(c.name) for c in self.criteria)
            + ".\n"
            "For SCORE_0_100, use integers 0–100. For BOOLEAN, use true/false. "
            "For ENUM, use one of the allowed values. For TEXT_COMMENT, return "
            "a short explanation string."
        )

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Validation / aggregation helpers
    # ------------------------------------------------------------------

    def validate_scores(self, scores: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize a raw scores dict from an evaluator.

        Returns a cleaned copy with:
          - missing criteria filled with 0 / None
          - numeric types normalized to float
        """
        cleaned: Dict[str, Any] = {}

        for c in self.criteria:
            raw = scores.get(c.name)

            if c.type in (CriterionType.SCORE_0_100, CriterionType.SCORE_0_1):
                try:
                    v = float(raw)
                except (TypeError, ValueError):
                    v = 0.0
                # clamp to [0, 100] / [0, 1]
                if c.type == CriterionType.SCORE_0_100:
                    if v < 0.0:
                        v = 0.0
                    if v > 100.0:
                        v = 100.0
                else:
                    if v < 0.0:
                        v = 0.0
                    if v > 1.0:
                        v = 1.0
                cleaned[c.name] = v
            elif c.type == CriterionType.BOOLEAN:
                cleaned[c.name] = bool(raw)
            elif c.type == CriterionType.ENUM:
                if c.allowed_values and raw not in c.allowed_values:
                    log.warning(
                        "RubricMemCube %s: invalid enum value %r for criterion %s",
                        self.name,
                        raw,
                        c.name,
                    )
                    cleaned[c.name] = None
                else:
                    cleaned[c.name] = raw
            elif c.type == CriterionType.TEXT_COMMENT:
                cleaned[c.name] = str(raw) if raw is not None else ""
            else:
                cleaned[c.name] = raw

        return cleaned

    def aggregate_score(
        self,
        scores: Dict[str, Any],
        criterion_names: Optional[Sequence[str]] = None,
    ) -> float:
        """
        Aggregate numeric criteria into a single score via weighted average.

        Only SCORE_0_100 and SCORE_0_1 criteria are considered. SCORE_0_1
        criteria are scaled to 0–100 for aggregation.

        If criterion_names is provided, only those criteria are used.
        """
        cleaned = self.validate_scores(scores)
        total_weight = 0.0
        acc = 0.0

        for c in self.criteria:
            if criterion_names and c.name not in criterion_names:
                continue
            if c.type not in (CriterionType.SCORE_0_100, CriterionType.SCORE_0_1):
                continue

            v = cleaned.get(c.name)
            if v is None:
                continue

            v = float(v)
            if c.type == CriterionType.SCORE_0_1:
                v = v * 100.0

            w = float(c.weight or 1.0)
            acc += v * w
            total_weight += w

        if total_weight <= 0.0:
            return 0.0

        return acc / total_weight


# ---------------------------------------------------------------------------
# Utility: pretty JSON dump (for debugging)
# ---------------------------------------------------------------------------

def dump_rubric_json(rubric: RubricMemCube) -> str:
    """
    Convenience helper for logging/debugging rubrics in JSON form.
    """
    return json.dumps(rubric.to_dict(), indent=2, ensure_ascii=False, default=str)
