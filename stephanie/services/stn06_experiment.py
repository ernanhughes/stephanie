"""STN06 deterministic preregistration and dry-run evaluation.

This service performs no model inference. It freezes the withdrawn-but-approved
STN05 experience as an explicitly authorized, experiment-scoped input without
reactivating the MemCube for ordinary agent execution, builds the held-out task
manifest and arm schedule, and runs a deterministic dry-run that exercises
dispatch construction, memory selection, the sealed oracle and the scoring path.

Ordinary execution keeps using ``ExecutionExperienceService.retrieve_for_task``,
which still requires an ACTIVE, reuse-eligible MemCube. The withdrawn record is
therefore unavailable outside the experiment.
"""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stephanie.experiments import stn06_manifest as manifest_mod
from stephanie.memory.experiment_store import _stable_variant_index
from stephanie.services.execution_experience import (
    EXPERIENCE_CONTEXT_LABEL,
    RETRIEVAL_POLICY,
    format_experience_section,
)

EXPERIMENT_NAME = "stn06-coding-transfer"
EXPERIMENT_DOMAIN = "python"
FROZEN_EXPERIENCE_ID = "exp_33fd63f133b44b21275681e5af0aabb7"
FROZEN_EXPERIENCE_VERSION = "v1"
FROZEN_EXPERIENCE_HASH = "d4d72077672fb481d7e955ccf249832921691d5ac268be4ca1f004184e12af79"

MODEL_ROUTE = "opencode-go/gpt-6-luna"
EXECUTOR_BACKEND = "opencode-docker"
PERMISSION_CLASS = "workspace_write"
WORKER_PROTOCOL = "worker.work-request.v1"

ARM_BASELINE = "A"
ARM_EXPERIENCE = "B"
ARM_NEUTRAL = "C"

PILOT_TASK_IDS = ["stn06-near-01", "stn06-near-02"]

_NEUTRAL_CONTEXT = (
    "Neutral task context (control; no retrieved experience):\n"
    "No prior experience is supplied for this task."
)


class Stn06Error(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def task_case_id(task_id: str) -> int:
    """Stable integer case id derived from a task id (cross-process safe)."""
    return int(hashlib.sha256(f"stn06-task:{task_id}".encode("utf-8")).hexdigest()[:12], 16)


_VOLATILE_KEYS = frozenset({"created_at", "frozen_at", "authorized_at"})


def _strip_volatile(value: Any) -> Any:
    """Remove wall-clock timestamps so the preregistration hash is reproducible."""
    if isinstance(value, dict):
        return {key: _strip_volatile(item) for key, item in value.items() if key not in _VOLATILE_KEYS}
    if isinstance(value, list):
        return [_strip_volatile(item) for item in value]
    return value


def build_schedule() -> list[dict]:
    """Frozen per-task arm order built from the stable SHA-256 assignment."""
    schedule = []
    for task in manifest_mod.HELD_OUT_TASKS:
        case_id = task_case_id(task["task_id"])
        idx = _stable_variant_index(EXPERIMENT_NAME, EXPERIMENT_DOMAIN, case_id, 2)
        order = [ARM_BASELINE, ARM_EXPERIENCE] if idx == 0 else [ARM_EXPERIENCE, ARM_BASELINE]
        schedule.append({
            "task_id": task["task_id"],
            "task_class": task["task_class"],
            "case_id": case_id,
            "run_order": order,
        })
    return schedule


class Stn06ExperimentService:
    """Freeze, authorize, and dry-run the STN06 experience-transfer experiment."""

    def __init__(self, *, memory, cfg, logger=None, experience_service=None):
        self.memory = memory
        self.cfg = cfg
        self.logger = logger
        self.experience_service = experience_service

    @property
    def name(self) -> str:
        return "sis-stn06-experiment"

    def initialize(self, **_kwargs):
        return None

    def health_check(self):
        settings = self._settings()
        return {"status": "configured" if settings.get("enabled", False) else "disabled"}

    def shutdown(self):
        return None

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    def _settings(self) -> dict:
        return dict(self.cfg.get("stn06_experiment", {}) or {})

    def _require_enabled(self):
        if not self._settings().get("enabled", False):
            raise Stn06Error("STN06_DISABLED", "STN06 experiment preparation is disabled")

    def _frozen_spec(self) -> dict:
        spec = self._settings().get("frozen_experience", {}) or {}
        return {
            "experience_id": spec.get("experience_id", FROZEN_EXPERIENCE_ID),
            "version": spec.get("version", FROZEN_EXPERIENCE_VERSION),
            "candidate_hash": spec.get("candidate_hash", FROZEN_EXPERIENCE_HASH),
        }

    # ------------------------------------------------------------------
    # 1. Freeze the approved experience (no global reactivation)
    # ------------------------------------------------------------------
    def freeze_experience(self) -> dict:
        self._require_enabled()
        spec = self._frozen_spec()
        row = self.memory.casebooks.get_execution_experience_candidate(spec["experience_id"])
        if not row:
            raise Stn06Error("FROZEN_EXPERIENCE_MISSING", "The frozen experience candidate is not in CaseBook")
        candidate = dict(row["candidate"])
        if candidate.get("candidate_hash") != spec["candidate_hash"]:
            raise Stn06Error("FROZEN_HASH_MISMATCH", "Persisted candidate hash does not match the frozen hash")
        if candidate.get("candidate_review_status") != "APPROVED":
            raise Stn06Error("FROZEN_NOT_APPROVED", "Only an approved candidate can be frozen")
        if candidate.get("execution_review_status") != "ACCEPT":
            raise Stn06Error("FROZEN_EXECUTION_NOT_ACCEPTED", "The source execution must remain accepted")

        memcube_id = f"{spec['experience_id']}:{spec['version']}"
        cube = self.memory.memcubes.get_by_id(memcube_id)
        extra = (cube.extra_data or {}) if cube is not None else {}
        # The withdrawal must survive freezing: no global reactivation.
        if extra.get("reuse_eligible") is True or extra.get("promotion_status") == "ACTIVE":
            raise Stn06Error(
                "MEMCUBE_GLOBALLY_ACTIVE",
                "The frozen experience is globally retrievable; experiment-scoped freezing requires it to stay withdrawn",
            )

        lesson = dict(candidate.get("lesson_hypothesis") or {})
        result_section = {
            "experience_id": spec["experience_id"],
            "memcube_id": memcube_id,
            "source": candidate.get("source"),
            "lesson_hypothesis": lesson.get("text"),
            "limitations": list(lesson.get("limitations") or []),
        }
        context_section = format_experience_section(result_section)
        return {
            "schema": "stephanie.stn06-frozen-experience.v1",
            "experience_id": spec["experience_id"],
            "version": spec["version"],
            "candidate_hash": candidate.get("candidate_hash"),
            "memcube_id": memcube_id,
            "lesson_hypothesis": lesson,
            "applicability": list(lesson.get("applicability") or []),
            "limitations": list(lesson.get("limitations") or []),
            "provenance": {
                "source": candidate.get("source"),
                "execution_review": candidate.get("execution_review"),
                "candidate_review": candidate.get("candidate_review"),
                "deactivation_history": list(candidate.get("deactivation_history") or []),
            },
            "memcube_state": {
                "found": cube is not None,
                "reuse_eligible": extra.get("reuse_eligible"),
                "promotion_status": extra.get("promotion_status"),
            },
            "withdrawal_preserved": True,
            "globally_retrievable": False,
            "context_section": context_section,
            "context_hash": hashlib.sha256(context_section.encode("utf-8")).hexdigest(),
            "frozen_at": datetime.now(timezone.utc).isoformat(),
        }

    # ------------------------------------------------------------------
    # 2. Explicit, auditable experiment-scoped authorization
    # ------------------------------------------------------------------
    def authorize_scope(
        self,
        *,
        reviewer: str,
        authorization_id: str,
        notes: str,
        retention: str = "experiment_only",
    ) -> dict:
        self._require_enabled()
        if not authorization_id or len(authorization_id) > 128:
            raise Stn06Error("AUTHORIZATION_ID_INVALID", "Authorization id must be a bounded identifier")
        if not reviewer or len(reviewer) > 160:
            raise Stn06Error("AUTHORIZATION_REVIEWER_INVALID", "Authorization reviewer must be identified")
        if not (notes or "").strip():
            raise Stn06Error("AUTHORIZATION_NOTES_REQUIRED", "Record a rationale for the experiment-scoped authorization")
        frozen = self.freeze_experience()
        return {
            "schema": "stephanie.stn06-authorization.v1",
            "authorization_id": authorization_id,
            "reviewer": reviewer.strip(),
            "notes": notes.strip()[:2000],
            "retention": retention,
            "scope": EXPERIMENT_NAME,
            "experience_id": frozen["experience_id"],
            "version": frozen["version"],
            "candidate_hash": frozen["candidate_hash"],
            "global_reactivation": False,
            "normal_execution_eligibility_unchanged": True,
            "authorized_at": datetime.now(timezone.utc).isoformat(),
        }

    # ------------------------------------------------------------------
    # 3. Manifest, schedule, budget, preregistration
    # ------------------------------------------------------------------
    def build_manifest(self) -> dict:
        return manifest_mod.manifest()

    def build_schedule(self) -> list[dict]:
        return build_schedule()

    def budget_policy(self) -> dict:
        full_executions = manifest_mod.TASK_CLASS_COUNTS
        return {
            "pilot": {
                "task_ids": list(PILOT_TASK_IDS),
                "arms": [ARM_BASELINE, ARM_EXPERIENCE],
                "planned_executions": len(PILOT_TASK_IDS) * 2,
                "purpose": "exercise the experiment machinery; not evidence of improvement",
            },
            "full": {
                "task_ids": [task["task_id"] for task in manifest_mod.HELD_OUT_TASKS],
                "arms": [ARM_BASELINE, ARM_EXPERIENCE],
                "planned_executions": int(full_executions.get("near_transfer", 0)
                                          + full_executions.get("complex", 0)
                                          + full_executions.get("unrelated", 0)) * 2,
            },
            "enforceable_limits": {
                "max_model_requests_per_execution": 1,
                "max_wall_seconds_per_execution": 600,
                "no_silent_retries": True,
                "no_model_substitution": True,
                "no_paid_escalation": True,
                "max_pilot_paid_executions": len(PILOT_TASK_IDS) * 2,
                "max_full_paid_executions": 16,
            },
            "monetary": {
                "target_usd": 2.0,
                "enforceable": False,
                "reason": (
                    "STN04 did not report token usage or cost and the runtime exposes no metering feed. "
                    "A dollar hard-stop cannot be enforced; bounded execution counts, wall-clock limits and "
                    "a single model request per execution are the enforceable controls. Report the dollar "
                    "target as a target, not a guaranteed cap."
                ),
            },
        }

    def evaluation_criteria(self) -> dict:
        return {
            "primary": (
                "Binary task success on the first bounded attempt: the sealed oracle exits with the "
                "expected code AND every modified path is within the expected scope AND no prohibited path changed."
            ),
            "secondary": [
                "oracle exit code and failure category",
                "out-of-scope or prohibited modifications",
                "unnecessary changes",
                "wall-clock duration",
                "attempts and tool errors",
                "reported token usage and cost only if the runtime reports them",
            ],
            "arm_semantics": {
                ARM_BASELINE: "GPT-6 Luna + OpenCode with the sealed objective only (no retrieved experience).",
                ARM_EXPERIENCE: "Identical objective plus exactly the frozen approved experience as bounded, lower-trust context.",
                ARM_NEUTRAL: "Optional deferred arm: identical but with a neutral, non-experience context block.",
            },
            "analysis": (
                "Paired per-task differences; exact McNemar for pass/fail; robust paired summaries for duration; "
                "missing usage/cost treated as unavailable, never estimated."
            ),
        }

    def preregistration(self) -> dict:
        self._require_enabled()
        frozen = self.freeze_experience()
        manifest = self.build_manifest()
        document = {
            "schema": "stephanie.stn06-preregistration.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "experiment": {"name": EXPERIMENT_NAME, "domain": EXPERIMENT_DOMAIN},
            "model_route": MODEL_ROUTE,
            "executor": {"backend": EXECUTOR_BACKEND, "permission_class": PERMISSION_CLASS, "protocol": WORKER_PROTOCOL},
            "frozen_experience": frozen,
            "manifest": manifest,
            "schedule": self.build_schedule(),
            "retrieval_policy": RETRIEVAL_POLICY,
            "context_label": EXPERIENCE_CONTEXT_LABEL,
            "context_hash": frozen["context_hash"],
            "evaluation": self.evaluation_criteria(),
            "budget": self.budget_policy(),
            "leakage_controls": [
                "The experience is frozen before any held-out task is disclosed to the model.",
                "Only the frozen approved experience enters the experience arm.",
                "Retrieved text is bounded and labelled untrusted task context, never a system instruction.",
                "The sealed oracle and reference solutions are never placed in an agent workspace.",
                "Each arm receives a fresh, isolated workspace and cannot observe the other arm.",
                "No automatic prompt mutation, skill promotion, model training or autonomous policy change.",
                "The withdrawn MemCube is not globally reactivated for ordinary agent execution.",
            ],
        }
        document["preregistration_hash"] = hashlib.sha256(
            json.dumps(
                _strip_volatile(document),
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return document

    def persist_preregistration(self, *, authorization: dict) -> dict:
        self._require_enabled()
        document = self.preregistration()
        experiment = self.memory.experiments.get_or_create_experiment(
            EXPERIMENT_NAME,
            domain=EXPERIMENT_DOMAIN,
            label="STN06 coding transfer (experience vs baseline)",
            config={},
        )
        self.memory.experiments.upsert_variant(
            experiment.id, ARM_BASELINE, is_control=True,
            payload={"arm": ARM_BASELINE, "experience_ids": []},
        )
        self.memory.experiments.upsert_variant(
            experiment.id, ARM_EXPERIENCE, is_control=False,
            payload={
                "arm": ARM_EXPERIENCE,
                "experience_ids": [document["frozen_experience"]["experience_id"]],
                "candidate_hash": document["frozen_experience"]["candidate_hash"],
            },
        )
        self.memory.experiments.set_config(experiment.id, {
            "stn06_preregistration": document,
            "authorization": authorization,
        })
        return {
            "experiment_id": experiment.id,
            "name": EXPERIMENT_NAME,
            "domain": EXPERIMENT_DOMAIN,
            "preregistration_hash": document["preregistration_hash"],
            "frozen_experience_id": document["frozen_experience"]["experience_id"],
            "candidate_hash": document["frozen_experience"]["candidate_hash"],
        }

    # ------------------------------------------------------------------
    # 4. Experiment-scoped memory selection
    # ------------------------------------------------------------------
    def select_experience(self, arm: str, task_id: str | None = None) -> dict:
        self._require_enabled()
        frozen = self.freeze_experience()
        if arm == ARM_BASELINE:
            return {
                "arm": ARM_BASELINE,
                "task_id": task_id,
                "experience_ids": [],
                "selected_experience_ids": [],
                "results": [],
                "context": None,
                "context_chars": 0,
                "context_hash": None,
            }
        if arm == ARM_NEUTRAL:
            context = _NEUTRAL_CONTEXT
            return {
                "arm": ARM_NEUTRAL,
                "task_id": task_id,
                "experience_ids": [],
                "selected_experience_ids": [],
                "results": [],
                "context": context,
                "context_chars": len(context),
                "context_hash": hashlib.sha256(context.encode("utf-8")).hexdigest(),
            }
        if arm != ARM_EXPERIENCE:
            raise Stn06Error("ARM_INVALID", "Arm must be one of A (baseline), B (experience) or C (neutral)")
        candidate = dict(self.memory.casebooks.get_execution_experience_candidate(frozen["experience_id"])["candidate"])
        result = {
            "experience_id": frozen["experience_id"],
            "memcube_id": frozen["memcube_id"],
            "score": 1.0,
            "source": candidate.get("source"),
            "lesson_hypothesis": (candidate.get("lesson_hypothesis") or {}).get("text"),
            "limitations": list((candidate.get("lesson_hypothesis") or {}).get("limitations") or []),
        }
        context = format_experience_section(result)
        if hashlib.sha256(context.encode("utf-8")).hexdigest() != frozen["context_hash"]:
            raise Stn06Error("CONTEXT_DRIFT", "Rendered experience context does not match the frozen context hash")
        return {
            "arm": ARM_EXPERIENCE,
            "task_id": task_id,
            "experience_ids": [frozen["experience_id"]],
            "selected_experience_ids": [frozen["experience_id"]],
            "candidate_hash": frozen["candidate_hash"],
            "results": [result],
            "context": context,
            "context_chars": len(context),
            "context_hash": frozen["context_hash"],
            "retrieval_policy": RETRIEVAL_POLICY,
        }

    def normal_retrieval(self, task: dict) -> dict:
        """Return the ordinary-execution retrieval result for a held-out task."""
        if self.experience_service is None:
            raise Stn06Error("EXPERIENCE_SERVICE_REQUIRED", "An ExecutionExperienceService is required for this check")
        return self.experience_service.retrieve_for_task(task["objective"])

    # ------------------------------------------------------------------
    # 5. Deterministic fixture / oracle verification
    # ------------------------------------------------------------------
    @staticmethod
    def _run_oracle(command: str, cwd: str) -> subprocess.CompletedProcess:
        parts = shlex.split(command)
        if parts and parts[0] in {"python3", "python"}:
            parts[0] = sys.executable
        return subprocess.run(parts, cwd=cwd, capture_output=True, text=True, timeout=120)

    def _evaluate_workspace(self, task: dict, workspace: dict[str, str]) -> dict:
        """Run the sealed oracle against a workspace in an isolated temp dir."""
        with tempfile.TemporaryDirectory(prefix="stn06-oracle-") as tmp:
            root = Path(tmp)
            for path, content in workspace.items():
                target = root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
            for path, content in task["oracle_files"].items():
                (root / path).write_text(content, encoding="utf-8")
            completed = self._run_oracle(task["oracle_command"], str(root))
            return {
                "exit_code": completed.returncode,
                "expected_exit_code": task["expected_exit_code"],
                "passed": completed.returncode == task["expected_exit_code"],
                "stdout_tail": (completed.stdout or "")[-600:],
                "stderr_tail": (completed.stderr or "")[-600:],
            }

    def verify_fixtures(self, task_ids: list[str] | None = None) -> dict:
        """Prove each oracle fails on the defective workspace and passes on the reference solution."""
        selected = task_ids or [task["task_id"] for task in manifest_mod.HELD_OUT_TASKS]
        results = []
        for task_id in selected:
            task = manifest_mod.task_by_id(task_id)
            if task is None:
                raise Stn06Error("TASK_NOT_FOUND", f"Unknown held-out task {task_id}")
            broken = self._evaluate_workspace(task, task["workspace_files"])
            fixed_workspace = dict(task["workspace_files"])
            fixed_workspace.update(task["reference_solution"])
            fixed = self._evaluate_workspace(task, fixed_workspace)
            results.append({
                "task_id": task_id,
                "defect_fails_oracle": not broken["passed"],
                "reference_passes_oracle": fixed["passed"],
                "broken_exit_code": broken["exit_code"],
                "reference_exit_code": fixed["exit_code"],
                "verified": (not broken["passed"]) and fixed["passed"],
            })
        return {
            "tasks": results,
            "all_verified": all(item["verified"] for item in results),
        }

    # ------------------------------------------------------------------
    # 6. Scoring
    # ------------------------------------------------------------------
    @staticmethod
    def score_outcome(task: dict, *, oracle_exit_code: int, modified_paths: list[str]) -> dict:
        def _is_within(path: str, scope: str) -> bool:
            return path == scope or path.startswith(scope.rstrip("/") + "/")

        prohibited_hit = sorted(
            path for path in modified_paths
            if any(_is_within(path, blocked) for blocked in task["prohibited_paths"])
        )
        out_of_scope = sorted(
            path for path in modified_paths
            if not any(_is_within(path, allowed) for allowed in task["expected_modified_paths"])
            and path not in prohibited_hit
        )
        oracle_ok = oracle_exit_code == task["expected_exit_code"]
        success = bool(oracle_ok and not prohibited_hit and not out_of_scope and modified_paths)
        return {
            "task_id": task["task_id"],
            "oracle_ok": oracle_ok,
            "oracle_exit_code": oracle_exit_code,
            "modified_paths": sorted(modified_paths),
            "out_of_scope": out_of_scope,
            "prohibited_hits": prohibited_hit,
            "unnecessary_changes": len(out_of_scope),
            "success": success,
        }

    # ------------------------------------------------------------------
    # 7. Dispatch construction (no model call)
    # ------------------------------------------------------------------
    def build_dispatch(self, *, task: dict, arm: str, selection: dict, operation_id: str) -> dict:
        objective = task["objective"]
        context = selection.get("context")
        worker_objective = objective if not context else f"{objective}\n\n{context}"
        if len(worker_objective) > 16_000:
            raise Stn06Error("DISPATCH_TOO_LARGE", "Task plus context exceeds the execution prompt bound")
        return {
            "schema": WORKER_PROTOCOL,
            "app": "stephanie",
            "objective": worker_objective,
            "workspace": {"root": f"<{EXPERIMENT_NAME}:{task['task_id']}:{arm}>", "project_id": task["task_id"]},
            "execution": {"backend": EXECUTOR_BACKEND, "model": MODEL_ROUTE},
            "options": {"acceptance": "human", "dry_run": False},
            "metadata": {
                "source": "sis",
                "operation_id": operation_id,
                "stage_id": "OPENCODE",
                "experience_ids": list(selection.get("selected_experience_ids") or []),
                "experiment": {
                    "name": EXPERIMENT_NAME,
                    "task_id": task["task_id"],
                    "arm": arm,
                    "candidate_hash": selection.get("candidate_hash"),
                },
            },
        }

    # ------------------------------------------------------------------
    # 8. Dry-run evaluation (no model invocation)
    # ------------------------------------------------------------------
    def dry_run(self, *, scope: str = "pilot", verify_oracles: bool = True) -> dict:
        self._require_enabled()
        frozen = self.freeze_experience()
        if scope == "pilot":
            task_ids = list(PILOT_TASK_IDS)
        elif scope == "full":
            task_ids = [task["task_id"] for task in manifest_mod.HELD_OUT_TASKS]
        else:
            raise Stn06Error("SCOPE_INVALID", "Dry-run scope must be 'pilot' or 'full'")

        manifest = self.build_manifest()
        schedule = {entry["task_id"]: entry for entry in self.build_schedule()}
        per_task = []
        for task_id in task_ids:
            task = manifest_mod.task_by_id(task_id)
            baseline = self.select_experience(ARM_BASELINE, task_id)
            treatment = self.select_experience(ARM_EXPERIENCE, task_id)
            # Selection invariants
            if baseline["context"] is not None or baseline["selected_experience_ids"]:
                raise Stn06Error("BASELINE_CONTAMINATED", "The baseline arm received experience context")
            if treatment["selected_experience_ids"] != [frozen["experience_id"]]:
                raise Stn06Error("TREATMENT_MISMATCH", "The treatment arm did not receive exactly the frozen experience")
            if treatment["candidate_hash"] != frozen["candidate_hash"]:
                raise Stn06Error("TREATMENT_HASH_MISMATCH", "The treatment arm used a different candidate hash")
            # Dispatch construction for both arms
            baseline_dispatch = self.build_dispatch(
                task=task, arm=ARM_BASELINE, selection=baseline,
                operation_id=f"stn06-dryrun:{task_id}:A",
            )
            treatment_dispatch = self.build_dispatch(
                task=task, arm=ARM_EXPERIENCE, selection=treatment,
                operation_id=f"stn06-dryrun:{task_id}:B",
            )
            if baseline_dispatch["metadata"]["experience_ids"]:
                raise Stn06Error("BASELINE_DISPATCH_CONTAMINATED", "Baseline dispatch carried experience ids")
            if treatment_dispatch["metadata"]["experience_ids"] != [frozen["experience_id"]]:
                raise Stn06Error("TREATMENT_DISPATCH_MISMATCH", "Treatment dispatch did not carry the frozen experience id")
            # Scoring path exercised with clearly-labelled simulated outcomes.
            simulated_fail = self.score_outcome(task, oracle_exit_code=1, modified_paths=[])
            simulated_pass = self.score_outcome(
                task, oracle_exit_code=task["expected_exit_code"],
                modified_paths=list(task["expected_modified_paths"]),
            )
            if simulated_fail["success"] or not simulated_pass["success"]:
                raise Stn06Error("SCORING_PATH_INVALID", "Scoring did not distinguish clearly simulated failure from success")
            per_task.append({
                "task_id": task_id,
                "task_class": task["task_class"],
                "run_order": schedule[task_id]["run_order"],
                "baseline": {
                    "experience_ids": baseline["experience_ids"],
                    "context_present": baseline["context"] is not None,
                    "dispatch_objective_chars": len(baseline_dispatch["objective"]),
                },
                "treatment": {
                    "experience_ids": treatment["experience_ids"],
                    "candidate_hash": treatment["candidate_hash"],
                    "context_chars": treatment["context_chars"],
                    "context_hash": treatment["context_hash"],
                    "dispatch_objective_chars": len(treatment_dispatch["objective"]),
                },
                "scoring_path": {"simulated_fail": simulated_fail, "simulated_pass": simulated_pass},
            })

        normal_retrieval = None
        if self.experience_service is not None:
            samples = []
            for task_id in task_ids:
                task = manifest_mod.task_by_id(task_id)
                retrieved = self.normal_retrieval(task)
                samples.append({"task_id": task_id, "results": len(retrieved.get("results") or [])})
            normal_retrieval = {
                "samples": samples,
                "withdrawn_unavailable": all(sample["results"] == 0 for sample in samples),
            }

        verification = self.verify_fixtures(task_ids) if verify_oracles else None

        return {
            "scope": scope,
            "dry_run": True,
            "model_calls_made": 0,
            "trials_persisted": 0,
            "manifest_hash": manifest["manifest_hash"],
            "preregistration_hash": self.preregistration()["preregistration_hash"],
            "frozen_experience": {
                "experience_id": frozen["experience_id"],
                "version": frozen["version"],
                "candidate_hash": frozen["candidate_hash"],
                "context_hash": frozen["context_hash"],
                "globally_retrievable": frozen["globally_retrievable"],
                "memcube_state": frozen["memcube_state"],
            },
            "normal_retrieval": normal_retrieval,
            "oracle_verification": verification,
            "tasks": per_task,
            "planned_executions": len(task_ids) * 2,
        }
