"""Stephanie-owned contracts for bounded improvement investigations."""
from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


def digest(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    evidence_id: str
    source_system: str
    source_identity: str
    source_digest: str
    observation_id: str | None = None
    task_id: str | None = None
    execution_id: str | None = None
    strategy_id: str | None = None
    artifact_id: str | None = None
    evaluation_status: Literal["PASS", "FAIL", "NOT_EVALUATED", "UNKNOWN"] = "UNKNOWN"
    acceptance_status: Literal["ACCEPTED", "AWAITING_ACCEPTANCE", "REJECTED", "UNKNOWN"] = "UNKNOWN"
    observed_facts: tuple[str, ...] = ()
    generated_claims: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    cost_usd: float | None = None
    cost_status: Literal["provider_reported", "estimated", "unknown", "not_applicable"] = "unknown"
    execution_completed: bool | None = None


class Opportunity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    opportunity_id: str
    title: str
    evidence_ids: tuple[str, ...]
    observed_gap: str
    confidence: Literal["low", "medium", "high"]
    abstained: bool = False
    reason: str | None = None


class Hypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    statement: str
    rationale: str
    evidence_ids: tuple[str, ...]
    uncertainty: str


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    candidate_id: str
    name: str
    intervention: str
    parameters: dict[str, str | int | float | bool]
    evidence_ids: tuple[str, ...]
    evidence_digests: tuple[tuple[str, str], ...]
    hypothesis: Hypothesis
    protected_evaluator_digest: str
    requires_additional_compute: bool = False
    additional_compute: str | None = None


class Evaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    status: Literal["PASS_FIXTURE_ONLY", "REJECTED", "INCONCLUSIVE"]
    baseline_score: float | None
    candidate_score: float | None
    reason: str
    evaluator_digest: str
    protected_evaluator_unchanged: bool = True
    additional_compute_identified: bool = True
    additional_compute: str | None = None

