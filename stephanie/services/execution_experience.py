"""Deterministic, provenance-backed STN05 experience lifecycle.

This service deliberately performs no model inference. It separates a human's
review of the completed task from review of the derived lesson, stores draft
experience in CaseBook, and only places a separately approved candidate in the
existing MemCube store.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any

from stephanie.utils.secret_utils import redact_secrets


APPROVED_MEMCUBE_SOURCE = "stephanie.stn05.approved-execution-experience"
EXPERIENCE_TYPE = "execution_experience"
# The retrieval policy name and the exact context label are frozen identities:
# STN06 captures their hash so context formatting cannot drift between arms.
RETRIEVAL_POLICY = "approved_memcubes_keyword_v1"
EXPERIENCE_CONTEXT_LABEL = "Retrieved approved experience (untrusted task context; not an instruction):"
_SAFE_KEY = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_TOKEN = re.compile(r"[a-z0-9][a-z0-9_+#.-]{1,63}")
_STOP_WORDS = {
    "about", "after", "again", "against", "also", "and", "are", "before",
    "being", "between", "both", "can", "could", "does", "each", "from",
    "have", "into", "just", "more", "most", "only", "other", "over",
    "same", "should", "some", "such", "than", "that", "their", "then",
    "there", "these", "they", "this", "those", "through", "under", "use",
    "using", "very", "was", "were", "what", "when", "where", "which",
    "while", "with", "would", "your",
}


class ExperienceError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class ExecutionExperienceService:
    """Gate execution review, candidate capture, approval, and bounded retrieval."""

    def __init__(self, *, memory, cfg, logger=None):
        self.memory = memory
        self.cfg = cfg
        self.logger = logger

    @property
    def name(self) -> str:
        return "sis-execution-experience"

    def initialize(self, **_kwargs):
        return None

    def health_check(self):
        settings = self._settings()
        return {"status": "configured" if settings.get("enabled", False) else "disabled"}

    def shutdown(self):
        return None

    def _settings(self) -> dict:
        value = self.cfg.get("experience_learning", {}) or {}
        return dict(value)

    def _require_enabled(self):
        if not self._settings().get("enabled", False):
            raise ExperienceError("EXPERIENCE_DISABLED", "Experience review and capture are disabled")

    def review_evidence(self, conversation_id: int, execution_id: str) -> dict:
        self._require_enabled()
        record = self._execution(conversation_id, execution_id)
        artifact_paths = [
            item["id"] for item in record.get("artifacts", [])
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        ]
        refs = []
        if record.get("worker_request_id"):
            refs.append(f"worker_request:{record['worker_request_id']}")
        if record.get("trace_id"):
            refs.append(f"plan_trace:{record['trace_id']}")
        if record.get("verification_command"):
            refs.append(
                "verification:"
                f"{record['verification_command']}:"
                f"{record.get('verification_status', 'unknown')}:"
                f"{record.get('verification_exit_code', 'unknown')}"
            )
        refs.extend(f"artifact:{path}" for path in artifact_paths)
        return {
            "execution_id": execution_id,
            "status": record.get("status"),
            "human_review": record.get("human_review"),
            "human_review_history": record.get("human_review_history", []),
            "experience_id": record.get("experience_id"),
            "experience_status": record.get("experience_status", "NOT_EXTRACTED"),
            "objective": _bounded(record.get("objective"), 12000),
            "result_summary": _bounded(record.get("result_summary"), 12000),
            "artifacts": record.get("artifacts", []),
            "verification": {
                "command": record.get("verification_command"),
                "status": record.get("verification_status"),
                "exit_code": record.get("verification_exit_code"),
            },
            "evidence_refs": refs,
            "diff_artifact_available": False,
            "diff_note": "No before/after patch artifact was persisted; inspect the registered disposable workspace and retained task summary.",
        }

    def review_execution(
        self,
        *,
        conversation_id: int,
        execution_id: str,
        reviewer: str,
        decision: str,
        review_id: str,
        notes: str = "",
        evidence_refs: list[str] | None = None,
        accepted_scope: list[str] | None = None,
    ) -> dict:
        self._require_enabled()
        if not _SAFE_KEY.fullmatch(review_id or ""):
            raise ExperienceError("REVIEW_ID_INVALID", "Idempotency-Key must contain safe review identity characters")
        decision = str(decision).upper()
        if decision not in {"ACCEPT", "REJECT", "NEEDS_REVISION"}:
            raise ExperienceError("REVIEW_DECISION_INVALID", "Choose ACCEPT, REJECT, or NEEDS_REVISION")
        normalized_notes = _bounded(notes, 2000).strip()
        if not normalized_notes:
            raise ExperienceError("REVIEW_NOTES_REQUIRED", "Record a brief rationale for the human decision")
        record = self._execution(conversation_id, execution_id)
        refs = _string_list(evidence_refs, 50, 500)
        scope = _string_list(accepted_scope, 50, 250)
        available_refs = set(self.review_evidence(conversation_id, execution_id)["evidence_refs"])
        if any(ref not in available_refs for ref in refs):
            raise ExperienceError("REVIEW_EVIDENCE_INVALID", "Review references must identify evidence recorded for this execution")
        artifact_paths = {
            item["id"] for item in record.get("artifacts", [])
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        }
        if any(path not in artifact_paths for path in scope):
            raise ExperienceError("REVIEW_SCOPE_INVALID", "Accepted scope may include only Worker-reported artifacts")
        if decision == "ACCEPT":
            if record.get("status") != "COMPLETED_UNVERIFIED":
                raise ExperienceError("EXECUTION_NOT_REVIEWABLE", "Only completed, unverified Worker executions can be accepted here")
            if record.get("verification_status") != "passed" or record.get("verification_exit_code") != "0":
                raise ExperienceError("VERIFICATION_NOT_PASSED", "Acceptance requires a recorded independent verifier pass")
            required = {"verification:" + f"{record['verification_command']}:passed:0"}
            if record.get("worker_request_id"):
                required.add(f"worker_request:{record['worker_request_id']}")
            if record.get("trace_id"):
                required.add(f"plan_trace:{record['trace_id']}")
            required.update(f"artifact:{path}" for path in scope)
            if not scope or not required.issubset(set(refs)):
                raise ExperienceError("REVIEW_EVIDENCE_REQUIRED", "Acceptance must reference the Worker result, PlanTrace, passing verifier, and accepted artifact scope")
        review = {
            "review_id": review_id,
            "reviewer": _bounded(reviewer, 160),
            "decision": decision,
            "notes": normalized_notes,
            "evidence_refs": refs,
            "accepted_scope": scope if decision == "ACCEPT" else [],
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            stored = self.memory.chats.record_external_execution_review(
                conversation_id,
                execution_id,
                review_id=review["review_id"],
                reviewer=review["reviewer"],
                decision=decision,
                notes=review["notes"],
                evidence_refs=refs,
                accepted_scope=review["accepted_scope"],
                reviewed_at=review["reviewed_at"],
            )
        except ValueError as exc:
            raise ExperienceError("REVIEW_CONFLICT", str(exc)) from exc
        if stored is None:
            raise ExperienceError("EXECUTION_NOT_FOUND", "Execution was not found in this conversation")
        return _review_result(stored)

    def extract_candidate(self, *, conversation_id: int, execution_id: str) -> dict:
        self._require_enabled()
        record = self._execution(conversation_id, execution_id)
        review = record.get("human_review") or {}
        if review.get("decision") != "ACCEPT":
            raise ExperienceError("HUMAN_ACCEPTANCE_REQUIRED", "A human must accept the execution before experience extraction")
        if record.get("status") != "COMPLETED_UNVERIFIED":
            raise ExperienceError("EXECUTION_NOT_COMPLETE", "The Worker result is not a completed unverified execution")
        if record.get("verification_status") != "passed" or record.get("verification_exit_code") != "0":
            raise ExperienceError("VERIFICATION_NOT_PASSED", "A passing independent verifier is required for experience capture")

        trace = self.memory.plan_traces.get_by_trace_id(record.get("trace_id"))
        if trace is None:
            raise ExperienceError("PLANTRACE_NOT_FOUND", "The source PlanTrace is unavailable")
        trace_meta = trace.meta or {}
        if (
            trace_meta.get("execution_id") != execution_id
            or trace_meta.get("conversation_id") != conversation_id
            or trace_meta.get("worker_request_id") != record.get("worker_request_id")
            or trace_meta.get("backend") != record.get("backend")
            or trace_meta.get("requested_model") != record.get("requested_model")
        ):
            raise ExperienceError("PROVENANCE_MISMATCH", "Execution, conversation, Worker request, and PlanTrace identities do not match")
        step_ids = {getattr(step, "step_id", None) for step in (trace.execution_steps or [])}
        if not {"worker_submission", "worker_result"}.issubset(step_ids):
            raise ExperienceError("PLANTRACE_INCOMPLETE", "The source PlanTrace does not contain both submission and result steps")

        messages = self.memory.chats.get_messages(conversation_id)
        user_message = next((item for item in messages if item.id == record.get("originating_message_id")), None)
        assistant_message = next((item for item in messages if item.id == record.get("assistant_message_id")), None)
        if (
            user_message is None or user_message.role != "user"
            or assistant_message is None or assistant_message.role != "assistant"
            or trace_meta.get("user_message_id") != record.get("originating_message_id")
        ):
            raise ExperienceError("PROVENANCE_MISMATCH", "ChatStore message, assistant result, and PlanTrace links do not match")

        accepted_scope = set((record.get("human_review") or {}).get("accepted_scope") or [])
        artifacts = [
            {"path": item["id"], "kind": item.get("kind", "unknown")}
            for item in record.get("artifacts", [])
            if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"] in accepted_scope
        ]
        if not artifacts:
            raise ExperienceError("ARTIFACT_EVIDENCE_MISSING", "Worker did not report a changed artifact")
        objective = _bounded(record.get("objective"), 12000)
        if not objective:
            raise ExperienceError("OBJECTIVE_MISSING", "Execution objective is missing")
        try:
            goal = self.memory.goals.get_or_create({
                "goal_text": objective,
                "goal_type": "coding_execution_experience",
                "focus_area": "verified_execution_learning",
                "strategy": "human_gated_experience_capture",
                "source": "stn05_execution_experience",
            })
            goal_id = int(goal.id)
        except Exception as exc:
            raise ExperienceError("EXPERIENCE_GOAL_PERSISTENCE_FAILED", "The source task could not be linked to an existing Stephanie Goal") from exc

        source = {
            "conversation_id": conversation_id,
            "originating_message_id": record.get("originating_message_id"),
            "execution_id": execution_id,
            "worker_request_id": record.get("worker_request_id"),
            "plan_trace_id": record.get("trace_id"),
            "opencode_session_id": record.get("opencode_session_id"),
            "backend": record.get("backend"),
            "requested_model": record.get("requested_model"),
            "provider_request_model": record.get("provider_request_model"),
            "provider_response_status": record.get("provider_response_status"),
            "container_id": record.get("container_id"),
            "image_digest": record.get("image_digest"),
        }
        observations = [
            {
                "kind": "worker_report",
                "claim": "Worker reported the listed artifact change.",
                "artifact_paths": [item["path"] for item in artifacts],
                "evidence_refs": [f"worker_request:{record.get('worker_request_id')}"] if record.get("worker_request_id") else [],
            },
            {
                "kind": "independent_verification",
                "claim": "The configured independent test command passed.",
                "command": record.get("verification_command"),
                "exit_code": record.get("verification_exit_code"),
                "evidence_refs": [f"plan_trace:{record.get('trace_id')}"] if record.get("trace_id") else [],
            },
        ]
        lesson = (
            "For a small, localized code defect, inspect the failing tests, make the narrowest change, "
            "and verify the result independently before review."
        )
        candidate = {
            "schema": "stephanie.execution-experience.v1",
            "experience_id": _experience_id(execution_id),
            "candidate_hash": "",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "task": {
                "objective": objective,
                "category": "coding",
                "workspace_id": _bounded(record.get("workspace_id"), 120),
                "project_id": _bounded(record.get("project_id"), 120),
                "goal_id": goal_id,
            },
            "observations": observations,
            "reported_agent_summary": _bounded(record.get("result_summary"), 6000),
            "artifacts": artifacts,
            "verification": {
                "command": record.get("verification_command"),
                "status": record.get("verification_status"),
                "exit_code": record.get("verification_exit_code"),
            },
            "lesson_hypothesis": {
                "text": lesson,
                "classification": "hypothesis",
                "validation_status": "untested_for_transfer",
                "authoring": "deterministic_template",
                "applicability": ["small localized coding defects", "projects with independent tests"],
                "limitations": ["derived from one small task", "not evidence of general coding reliability", "transfer has not been measured"],
            },
            "source": source,
            "execution_review_status": "ACCEPT",
            "execution_review": {
                "review_id": review.get("review_id"),
                "reviewer": review.get("reviewer"),
                "reviewed_at": review.get("reviewed_at"),
                "accepted_scope": sorted(accepted_scope),
                "evidence_refs": list(review.get("evidence_refs") or []),
                "worker_acceptance_id": None,
            },
            "diff_artifact_available": False,
            "candidate_review_status": "PENDING_REVIEW",
            "reuse_eligible": False,
            "promotion_status": "NOT_PROMOTED",
            "model_assisted_extraction": False,
        }
        candidate["candidate_hash"] = _candidate_hash(candidate)
        result = self.memory.casebooks.upsert_execution_experience_candidate(candidate, goal_id=goal_id)
        stored = self.memory.chats.update_external_execution(conversation_id, execution_id, {
            "experience_id": candidate["experience_id"],
            "experience_status": "PENDING_REVIEW",
            "experience_casebook_id": result["casebook_id"],
            "experience_case_id": result["case_id"],
        })
        if stored is None:
            raise ExperienceError("EXECUTION_NOT_FOUND", "Experience candidate was stored but its source execution disappeared")
        return {
            "experience_id": candidate["experience_id"],
            "casebook_id": result["casebook_id"],
            "case_id": result["case_id"],
            "candidate": result["candidate"],
            "created": result["created"],
            "reuse_eligible": False,
        }

    def get_candidate(self, experience_id: str) -> dict | None:
        self._require_enabled()
        return self.memory.casebooks.get_execution_experience_candidate(experience_id)

    def review_candidate(
        self,
        *,
        conversation_id: int,
        execution_id: str,
        reviewer: str,
        decision: str,
        review_id: str,
        notes: str = "",
        evidence_refs: list[str] | None = None,
    ) -> dict:
        self._require_enabled()
        if not _SAFE_KEY.fullmatch(review_id or ""):
            raise ExperienceError("REVIEW_ID_INVALID", "Idempotency-Key must contain safe review identity characters")
        decision = str(decision).upper()
        if decision not in {"APPROVE", "REJECT", "NEEDS_REVISION"}:
            raise ExperienceError("EXPERIENCE_REVIEW_INVALID", "Choose APPROVE, REJECT, or NEEDS_REVISION")
        normalized_notes = _bounded(notes, 2000).strip()
        if not normalized_notes:
            raise ExperienceError("EXPERIENCE_REVIEW_NOTES_REQUIRED", "Record a brief rationale for the experience decision")
        record = self._execution(conversation_id, execution_id)
        if (record.get("human_review") or {}).get("decision") != "ACCEPT":
            raise ExperienceError("HUMAN_ACCEPTANCE_REQUIRED", "The source execution must be explicitly accepted before the experience can be approved")
        experience_id = record.get("experience_id")
        if not experience_id:
            raise ExperienceError("EXPERIENCE_NOT_CAPTURED", "Extract a candidate after reviewing the execution")
        existing = self.memory.casebooks.get_execution_experience_candidate(experience_id)
        if not existing:
            raise ExperienceError("EXPERIENCE_NOT_FOUND", "Experience candidate is missing")
        candidate = existing["candidate"]
        candidate_source = candidate.get("source") or {}
        if candidate_source.get("execution_id") != execution_id or candidate_source.get("conversation_id") != conversation_id:
            raise ExperienceError("PROVENANCE_MISMATCH", "Experience candidate does not match the source execution and conversation")
        valid_refs = {
            f"worker_request:{record.get('worker_request_id')}",
            f"plan_trace:{record.get('trace_id')}",
            f"verification:{record.get('verification_command')}:{record.get('verification_status')}:{record.get('verification_exit_code')}",
            f"experience:{experience_id}",
        }
        refs = _string_list(evidence_refs, 50, 500)
        if any(ref not in valid_refs for ref in refs):
            raise ExperienceError("EXPERIENCE_REVIEW_EVIDENCE_INVALID", "Experience review references must come from this candidate's recorded provenance")
        if decision == "APPROVE" and not {f"experience:{experience_id}", f"plan_trace:{record.get('trace_id')}", f"verification:{record.get('verification_command')}:{record.get('verification_status')}:{record.get('verification_exit_code')}"}.issubset(set(refs)):
            raise ExperienceError("EXPERIENCE_REVIEW_EVIDENCE_REQUIRED", "Approval must reference the candidate, PlanTrace, and independent verifier")
        review = {
            "review_id": review_id,
            "reviewer": _bounded(reviewer, 160),
            "decision": decision,
            "notes": normalized_notes,
            "evidence_refs": refs,
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
        }
        existing_candidate = self.memory.casebooks.get_execution_experience_candidate(experience_id)
        if not existing_candidate:
            raise ExperienceError("EXPERIENCE_NOT_FOUND", "Experience candidate is missing")
        current_candidate = existing_candidate["candidate"]
        promotion_retry = (
            current_candidate.get("candidate_review_status") == "APPROVED"
            and current_candidate.get("promotion_status") == "APPROVED_PENDING_PROMOTION"
        )
        if promotion_retry:
            if decision != "APPROVE":
                raise ExperienceError("EXPERIENCE_REVIEW_CONFLICT", "The lesson is approved; only idempotent promotion recovery is allowed")
            candidate = current_candidate
        else:
            try:
                reviewed = self.memory.casebooks.review_execution_experience_candidate(experience_id, review)
            except ValueError as exc:
                raise ExperienceError("EXPERIENCE_REVIEW_CONFLICT", str(exc)) from exc
            if reviewed is None:
                raise ExperienceError("EXPERIENCE_NOT_FOUND", "Experience candidate is missing")
            candidate = reviewed["candidate"]
        memcube_id = None
        if decision == "APPROVE":
            if candidate.get("candidate_review_status") != "APPROVED" or candidate.get("execution_review_status") != "ACCEPT":
                raise ExperienceError("EXPERIENCE_NOT_ELIGIBLE", "Experience approval did not satisfy the execution review gate")
            cube = self.memory.memcubes.insert(
                self._memcube_record(candidate),
                if_exists="skip",
            )
            memcube_id = cube.id
            promoted = self.memory.casebooks.mark_execution_experience_promoted(
                experience_id, memcube_id, datetime.now(timezone.utc).isoformat()
            )
            if promoted is None:
                raise ExperienceError("EXPERIENCE_PROMOTION_RECORD_MISSING", "MemCube was written but the CaseBook promotion record is missing")
            self.memory.chats.update_external_execution(conversation_id, execution_id, {
                "experience_status": "ACTIVE",
                "experience_memcube_id": memcube_id,
            })
            candidate = promoted["candidate"]
        return {
            "experience_id": experience_id,
            "candidate_review_status": candidate.get("candidate_review_status"),
            "reuse_eligible": bool(candidate.get("reuse_eligible")),
            "promotion_status": candidate.get("promotion_status"),
            "memcube_id": memcube_id or candidate.get("memcube_id"),
            "candidate": candidate,
        }

    def revise_candidate(
        self,
        *,
        conversation_id: int,
        execution_id: str,
        reviewer: str,
        revision_id: str,
        lesson_text: str,
    ) -> dict:
        self._require_enabled()
        if not _SAFE_KEY.fullmatch(revision_id or ""):
            raise ExperienceError("REVISION_ID_INVALID", "Idempotency-Key must contain safe revision identity characters")
        text = _bounded(lesson_text, 1200).strip()
        if len(text) < 20:
            raise ExperienceError("EXPERIENCE_REVISION_INVALID", "Revised lesson must contain at least 20 non-secret characters")
        if text != lesson_text.strip():
            raise ExperienceError("EXPERIENCE_REVISION_CONTAINS_SECRET", "Revised lesson contains redacted or credential-like content")
        record = self._execution(conversation_id, execution_id)
        if (record.get("human_review") or {}).get("decision") != "ACCEPT":
            raise ExperienceError("HUMAN_ACCEPTANCE_REQUIRED", "The source execution must be accepted before revising its lesson")
        experience_id = record.get("experience_id")
        if not experience_id:
            raise ExperienceError("EXPERIENCE_NOT_CAPTURED", "Extract an experience candidate before revising its lesson")
        found = self.memory.casebooks.get_execution_experience_candidate(experience_id)
        if not found:
            raise ExperienceError("EXPERIENCE_NOT_FOUND", "Experience candidate is missing")
        candidate = dict(found["candidate"])
        if (candidate.get("source") or {}).get("execution_id") != execution_id or (candidate.get("source") or {}).get("conversation_id") != conversation_id:
            raise ExperienceError("PROVENANCE_MISMATCH", "Experience candidate does not match the source execution and conversation")
        if candidate.get("candidate_review_status") != "PENDING_REVIEW":
            raise ExperienceError("EXPERIENCE_REVISION_REQUIRED", "A candidate marked NEEDS_REVISION must receive a new human-authored revision before another approval decision")
        if candidate.get("candidate_review_status") != "NEEDS_REVISION" or candidate.get("reuse_eligible"):
            raise ExperienceError("EXPERIENCE_REVISION_NOT_REQUESTED", "A human must first mark this candidate NEEDS_REVISION")
        history = list(candidate.get("revision_history") or [])
        prior = next((item for item in history if item.get("revision_id") == revision_id), None)
        if prior is not None:
            if prior.get("lesson_text") != text or prior.get("reviewer") != reviewer:
                raise ExperienceError("EXPERIENCE_REVISION_CONFLICT", "Revision idempotency key was already used for different content")
            return {"experience_id": experience_id, "candidate": candidate, "created": False}
        if len(history) >= 10:
            raise ExperienceError("EXPERIENCE_REVISION_LIMIT", "Experience candidate revision history is full")
        old_hash = candidate.get("candidate_hash")
        candidate["lesson_hypothesis"] = {
            **dict(candidate.get("lesson_hypothesis") or {}),
            "text": text,
            "classification": "hypothesis",
            "validation_status": "untested_for_transfer",
            "authoring": "human_revision",
        }
        candidate["revision_history"] = history + [{
            "revision_id": revision_id,
            "reviewer": _bounded(reviewer, 160),
            "lesson_text": text,
            "revises_candidate_hash": old_hash,
            "revised_at": datetime.now(timezone.utc).isoformat(),
        }]
        candidate["candidate_review_status"] = "PENDING_REVIEW"
        candidate["candidate_review"] = None
        candidate["reuse_eligible"] = False
        candidate["promotion_status"] = "NOT_PROMOTED"
        candidate["candidate_hash"] = _candidate_hash(candidate)
        try:
            result = self.memory.casebooks.revise_execution_experience_candidate(
                experience_id, candidate, revision_id=revision_id
            )
        except ValueError as exc:
            raise ExperienceError("EXPERIENCE_REVISION_CONFLICT", str(exc)) from exc
        if result is None:
            raise ExperienceError("EXPERIENCE_NOT_FOUND", "Experience candidate is missing")
        self.memory.chats.update_external_execution(conversation_id, execution_id, {
            "experience_status": "PENDING_REVIEW",
        })
        return {"experience_id": experience_id, "candidate": result["candidate"], "created": result["created"]}

    def deactivate_experience(
        self,
        *,
        conversation_id: int,
        experience_id: str,
        reviewer: str,
        reason: str,
        review_id: str,
        disposition: str = "WITHDRAWN",
        superseded_by: str | None = None,
    ) -> dict:
        self._require_enabled()
        if not _SAFE_KEY.fullmatch(review_id or ""):
            raise ExperienceError("REVIEW_ID_INVALID", "Idempotency-Key must contain safe review identity characters")
        disposition = str(disposition).upper()
        if disposition not in {"WITHDRAWN", "SUPERSEDED", "STALE"}:
            raise ExperienceError("EXPERIENCE_DISPOSITION_INVALID", "Disposition must be WITHDRAWN, SUPERSEDED, or STALE")
        if disposition == "SUPERSEDED" and not _SAFE_KEY.fullmatch(superseded_by or ""):
            raise ExperienceError("EXPERIENCE_SUPERSESSION_INVALID", "SUPERSEDED requires the replacement experience ID")
        found = self.memory.casebooks.get_execution_experience_candidate(experience_id)
        if not found:
            raise ExperienceError("EXPERIENCE_NOT_FOUND", "Experience candidate is missing")
        candidate = found["candidate"]
        if (candidate.get("source") or {}).get("conversation_id") != conversation_id:
            raise ExperienceError("EXPERIENCE_NOT_FOUND", "Experience is not owned by this conversation")
        if disposition == "SUPERSEDED":
            replacement = self.memory.casebooks.get_execution_experience_candidate(superseded_by)
            if (
                not replacement
                or not replacement["candidate"].get("reuse_eligible")
                or (replacement["candidate"].get("source") or {}).get("conversation_id") != conversation_id
            ):
                raise ExperienceError("EXPERIENCE_SUPERSESSION_INVALID", "Replacement experience must already be approved and active")
        memcube_id = candidate.get("memcube_id")
        if memcube_id:
            self.memory.memcubes.set_refined_result(
                memcube_id,
                extra_data={
                    "reuse_eligible": False,
                    "promotion_status": disposition,
                    "deactivated_by": _bounded(reviewer, 160),
                    "deactivation_reason": _bounded(reason, 2000),
                    "deactivation_review_id": review_id,
                    "superseded_by": superseded_by,
                    "deactivated_at": datetime.now(timezone.utc).isoformat(),
                },
                merge_extra=True,
            )
        result = self.memory.casebooks.deactivate_execution_experience_candidate(
            experience_id,
            reviewer=_bounded(reviewer, 160),
            reason=_bounded(reason, 2000),
            review_id=review_id,
            disposition=disposition,
            superseded_by=superseded_by,
        )
        # Keep the source execution's lifecycle label consistent with the
        # CaseBook/MemCube disposition. This updates only STN05 lifecycle
        # metadata; it never rewrites the STN04 evidence or review history.
        execution_id = (candidate.get("source") or {}).get("execution_id")
        if execution_id:
            self.memory.chats.update_external_execution(conversation_id, execution_id, {
                "experience_status": disposition,
            })
        return result

    def retrieve_for_task(self, query: str, *, limit: int = 3, experience_ids: list[str] | None = None) -> dict:
        self._require_enabled()
        retrieval_cfg = self._settings().get("retrieval", {}) or {}
        if not retrieval_cfg.get("enabled", False):
            raise ExperienceError("EXPERIENCE_RETRIEVAL_DISABLED", "Experience retrieval is disabled by configuration")
        task = _bounded(query, 4000)
        tokens = _tokens(task)
        if not tokens:
            return {"query": task, "results": [], "context": "", "retrieval_policy": "approved_memcubes_keyword_v1"}
        max_results = min(max(1, int(limit)), max(1, int(retrieval_cfg.get("max_results", 3))))
        max_chars = min(max(500, int(retrieval_cfg.get("max_context_chars", 2500))), 8000)
        selected_ids = None
        if experience_ids is not None:
            if len(experience_ids) > max_results or any(not re.fullmatch(r"exp_[a-f0-9]{32}", item or "") for item in experience_ids):
                raise ExperienceError("EXPERIENCE_SELECTION_INVALID", "Selected experience IDs are invalid or exceed the configured result limit")
            selected_ids = set(experience_ids)
        rows = self.memory.memcubes.search(
            scorable_type=EXPERIENCE_TYPE,
            source=APPROVED_MEMCUBE_SOURCE,
            sensitivity="internal",
            limit=max(1, int(retrieval_cfg.get("candidate_pool", 100))),
        )
        ranked = []
        seen_ids = set()
        for cube in rows:
            extra = cube.extra_data or {}
            if (
                extra.get("reuse_eligible") is not True
                or extra.get("execution_review_decision") != "ACCEPT"
                or extra.get("candidate_review_decision") != "APPROVE"
                or extra.get("promotion_status") != "ACTIVE"
            ):
                continue
            candidate = extra.get("candidate") or {}
            experience_id = candidate.get("experience_id") or extra.get("experience_id")
            if selected_ids is not None and experience_id not in selected_ids:
                continue
            if experience_id in seen_ids:
                continue
            seen_ids.add(experience_id)
            lesson = candidate.get("lesson_hypothesis") or {}
            searchable = " ".join([
                str(cube.content or ""),
                str((candidate.get("task") or {}).get("objective", "")),
                " ".join(str(item) for item in lesson.get("applicability", [])),
                " ".join(str(item) for item in candidate.get("tags", [])),
            ])
            overlap = tokens.intersection(_tokens(searchable))
            if not overlap:
                continue
            score = len(overlap) / max(1, len(tokens))
            ranked.append((score, cube, candidate, lesson))
        if selected_ids is not None and selected_ids != {item[2].get("experience_id") or item[1].extra_data.get("experience_id") for item in ranked}:
            raise ExperienceError("EXPERIENCE_SELECTION_UNAVAILABLE", "One or more selected experiences are not approved, active, or relevant to this task")
        ranked.sort(key=lambda item: (item[0], str(item[1].id)), reverse=True)
        results = []
        sections = []
        used = 0
        for score, cube, candidate, lesson in ranked[:max_results]:
            result = {
                "experience_id": candidate.get("experience_id"),
                "memcube_id": cube.id,
                "score": round(score, 4),
                "source": candidate.get("source"),
                "lesson_hypothesis": _bounded(lesson.get("text"), 800),
                "limitations": [_bounded(item, 300) for item in lesson.get("limitations", [])[:8]],
            }
            section = format_experience_section(result)
            room = max_chars - used
            if room <= 0:
                break
            section = section[:room]
            results.append(result)
            sections.append(section)
            used += len(section)
        return {
            "query": task,
            "results": results,
            "selected_experience_ids": [item["experience_id"] for item in results],
            "context": "\n\n".join(sections),
            "context_chars": sum(len(section) for section in sections),
            "retrieval_policy": RETRIEVAL_POLICY,
        }

    def _memcube_record(self, candidate: dict) -> dict:
        lesson = candidate["lesson_hypothesis"]
        extra_data = {
            "experience_id": candidate["experience_id"],
            "candidate_hash": candidate["candidate_hash"],
            "candidate": candidate,
            "execution_review_decision": "ACCEPT",
            "candidate_review_decision": "APPROVE",
            "reuse_eligible": True,
            "promotion_status": "ACTIVE",
            "tags": ["stn05", "approved", "coding", "execution_experience"],
        }
        return {
            "id": f"{candidate['experience_id']}:v1",
            "scorable_id": int(candidate["candidate_hash"][:8], 16) % 2_000_000_000,
            "scorable_type": EXPERIENCE_TYPE,
            "content": lesson["text"],
            "dimension": "coding",
            "version": "v1",
            "source": APPROVED_MEMCUBE_SOURCE,
            "model": None,
            "priority": 5,
            "sensitivity": "internal",
            "usage_count": 0,
            "extra_data": extra_data,
        }

    def _execution(self, conversation_id: int, execution_id: str) -> dict:
        record = self.memory.chats.get_external_execution(conversation_id, execution_id)
        if not record:
            raise ExperienceError("EXECUTION_NOT_FOUND", "Execution was not found in this conversation")
        if record.get("conversation_id") != conversation_id or record.get("execution_id") != execution_id:
            raise ExperienceError("PROVENANCE_MISMATCH", "Stored execution identity does not match the requested identity")
        return record


def format_experience_section(result: dict) -> str:
    """Render the frozen, bounded, labelled experience context section.

    This is the single source of truth for context formatting. STN06 hashes its
    output so the baseline and treatment arms cannot receive different framing.
    The text is explicitly labelled untrusted task context, never an instruction.
    """
    return (
        f"{EXPERIENCE_CONTEXT_LABEL}\n"
        f"Experience: {result.get('experience_id')}\n"
        f"Source PlanTrace: {(result.get('source') or {}).get('plan_trace_id', 'unknown')}\n"
        f"Candidate lesson hypothesis: {result.get('lesson_hypothesis')}\n"
        f"Limits: {'; '.join(result.get('limitations') or [])}"
    )


def _review_result(record: dict) -> dict:
    return {
        "execution_id": record.get("execution_id"),
        "execution_status": record.get("status"),
        "human_review": record.get("human_review"),
        "human_review_history": record.get("human_review_history", []),
        "human_review_status": record.get("human_review_status", "PENDING"),
        "experience_id": record.get("experience_id"),
        "experience_status": record.get("experience_status", "NOT_EXTRACTED"),
        "acceptance_id": record.get("acceptance_id"),
    }


def _experience_id(execution_id: str) -> str:
    digest = hashlib.sha256(f"stephanie-experience-v1:{execution_id}".encode()).hexdigest()[:32]
    return f"exp_{digest}"


def _candidate_hash(candidate: dict) -> str:
    stable = {key: value for key, value in candidate.items() if key not in {"candidate_hash", "created_at"}}
    raw = json.dumps(stable, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def _bounded(value: Any, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    return str(redact_secrets(value))[:limit]


def _string_list(values: list[str] | None, max_items: int, max_length: int) -> list[str]:
    if values is None:
        return []
    if not isinstance(values, list) or len(values) > max_items:
        raise ExperienceError("EXPERIENCE_INPUT_INVALID", "Evidence list is invalid or too large")
    result = []
    for value in values:
        if not isinstance(value, str) or not value.strip() or len(value) > max_length:
            raise ExperienceError("EXPERIENCE_INPUT_INVALID", "Evidence references must be bounded non-empty strings")
        if value not in result:
            result.append(value)
    return result


def _tokens(text: str) -> set[str]:
    return {item for item in _TOKEN.findall((text or "").lower()) if item not in _STOP_WORDS}
