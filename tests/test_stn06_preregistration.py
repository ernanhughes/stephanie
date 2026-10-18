"""Deterministic STN06 preregistration, leak-control and dry-run tests.

These tests never call a model. They exercise the frozen manifest hashes, the
stable arm schedule, the experiment-scoped freeze/select path, the sealed oracle
and the scoring function against an isolated SQLite database.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from sqlalchemy import JSON, create_engine
from sqlalchemy.orm import configure_mappers, sessionmaker
from hydra import compose, initialize_config_dir

# Import the normal MemoryTool store registry before configuring relationships.
from stephanie.memory.memory_tool import MemoryTool as _RegisterMemoryStores
from stephanie.memory.casebook_store import CaseBookStore
from stephanie.memory.experiment_store import ExperimentStore
from stephanie.memory.goal_store import GoalStore
from stephanie.memory.memcube_store import MemCubeStore
from stephanie.orm.base import Base
from stephanie.orm.casebook import CaseBookORM, CaseORM, CaseScorableORM
from stephanie.orm.experiment import ExperimentORM, VariantORM
from stephanie.orm.goal import GoalORM
from stephanie.orm.memcube import MemCubeORM
from stephanie.services.execution_experience import ExecutionExperienceService
from stephanie.services.stn06_experiment import (
    ARM_BASELINE,
    ARM_EXPERIENCE,
    EXPERIMENT_NAME,
    Stn06Error,
    Stn06ExperimentService,
    build_schedule,
)
from stephanie.experiments import stn06_manifest as manifest_mod

CONFIG_DIR = str((Path(__file__).resolve().parents[1] / "config").resolve())


class ChatFixture:
    def __init__(self, record):
        self.record = dict(record)
        self.messages = [
            SimpleNamespace(id=101, role="user"),
            SimpleNamespace(id=102, role="assistant"),
        ]

    def get_external_execution(self, conversation_id, execution_id):
        if self.record["conversation_id"] != conversation_id or self.record["execution_id"] != execution_id:
            return None
        return dict(self.record)

    def get_messages(self, conversation_id):
        return self.messages

    def update_external_execution(self, conversation_id, execution_id, changes):
        self.record.update(changes)
        return dict(self.record)


class TraceFixture:
    def __init__(self, meta):
        self.meta = meta
        self.execution_steps = [
            SimpleNamespace(step_id="worker_submission"),
            SimpleNamespace(step_id="worker_result"),
        ]


class TraceStoreFixture:
    def __init__(self, trace):
        self.trace = trace

    def get_by_trace_id(self, trace_id):
        return self.trace if trace_id == self.trace.meta["trace_id"] else None


def test_manifest_is_frozen_and_class_balanced():
    manifest = manifest_mod.manifest()
    assert manifest["task_count"] == 8
    assert manifest["task_class_counts"] == {"near_transfer": 4, "complex": 2, "unrelated": 2}
    # Hashing is deterministic.
    assert manifest_mod.manifest()["manifest_hash"] == manifest["manifest_hash"]
    # Every task has a distinct identity and sealed oracle.
    hashes = {entry["task_hash"] for entry in manifest["tasks"]}
    assert len(hashes) == 8
    for entry in manifest["tasks"]:
        assert entry["workspace_hash"] != entry["oracle_hash"]
        assert entry["reference_hash"] != entry["oracle_hash"]
        # Sealed artifacts are never part of the agent-facing workspace.
        assert not set(entry["oracle_file_paths"]) & set(entry["workspace_file_paths"])


def test_schedule_is_stable_across_python_hash_seeds():
    code = (
        "import json;"
        "from stephanie.services.stn06_experiment import build_schedule;"
        "print(json.dumps([entry['run_order'] for entry in build_schedule()]))"
    )
    outputs = []
    for seed in ("0", "12345", "random"):
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = seed
        completed = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, env=env, check=True,
        )
        outputs.append(completed.stdout.strip())
    assert len(set(outputs)) == 1
    assert json.loads(outputs[0]) == [entry["run_order"] for entry in build_schedule()]


def test_stn06_preparation_disabled_in_checked_in_defaults():
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="sis")
    assert cfg.stn06_experiment.enabled is False
    assert cfg.experience_learning.enabled is False


def _sqlite_stack(tmp_path, monkeypatch):
    configure_mappers()
    original_tags_type = CaseBookORM.__table__.c.tags.type
    monkeypatch.setattr(CaseBookORM.__table__.c.tags, "type", JSON())
    engine = create_engine(f"sqlite:///{tmp_path / 'stn06.sqlite'}")
    Base.metadata.create_all(
        engine,
        tables=[
            GoalORM.__table__,
            CaseBookORM.__table__,
            CaseORM.__table__,
            CaseScorableORM.__table__,
            MemCubeORM.__table__,
            ExperimentORM.__table__,
            VariantORM.__table__,
        ],
    )
    sessions = sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)
    return engine, sessions, original_tags_type


def _approved_candidate(memory, service):
    return service.extract_candidate(conversation_id=44, execution_id=memory.chats.record["execution_id"])


def _fixture_record():
    return {
        "conversation_id": 44,
        "execution_id": "sex_" + "a" * 32,
        "originating_message_id": 101,
        "assistant_message_id": 102,
        "worker_request_id": "wrq_" + "b" * 32,
        "trace_id": "worker:sex_" + "a" * 32,
        "status": "COMPLETED_UNVERIFIED",
        "backend": "opencode-docker",
        "requested_model": "opencode-go/gpt-6-luna",
        "provider_request_model": "gpt-6-luna",
        "provider_response_status": "200",
        "opencode_session_id": "ses_fixture",
        "container_id": "c" * 64,
        "image_digest": "sha256:" + "d" * 64,
        "project_id": "fixture-project",
        "workspace_id": "fixture-workspace",
        "objective": "Fix a small Python arithmetic defect and run the tests.",
        "verification_command": "python3 -m unittest",
        "verification_status": "passed",
        "verification_exit_code": "0",
        "artifacts": [{"id": "calculator.py", "kind": "modified_file"}],
        "result_summary": "Worker reported the arithmetic correction; two tests passed.",
        "acceptance_id": None,
        "human_review": {
            "review_id": "review-fixture",
            "reviewer": "synthetic-reviewer",
            "decision": "ACCEPT",
            "reviewed_at": "2026-10-17T00:00:00+00:00",
            "accepted_scope": ["calculator.py"],
            "evidence_refs": ["synthetic-fixture-evidence"],
        },
    }


def _build_memory(sessions):
    record = _fixture_record()
    trace = TraceFixture({
        "trace_id": record["trace_id"],
        "execution_id": record["execution_id"],
        "conversation_id": record["conversation_id"],
        "user_message_id": record["originating_message_id"],
        "worker_request_id": record["worker_request_id"],
        "backend": record["backend"],
        "requested_model": record["requested_model"],
    })
    return SimpleNamespace(
        chats=ChatFixture(record),
        plan_traces=TraceStoreFixture(trace),
        goals=GoalStore(sessions),
        casebooks=CaseBookStore(sessions),
        memcubes=MemCubeStore(sessions),
        experiments=ExperimentStore(sessions),
    )


def _experience_cfg(experience_id, candidate_hash, *, retrieval=True):
    return {
        "experience_learning": {
            "enabled": True,
            "model_assisted_extraction": False,
            "retrieval": {
                "enabled": retrieval,
                "max_results": 3,
                "candidate_pool": 20,
                "max_context_chars": 1200,
            },
        },
        "stn06_experiment": {
            "enabled": True,
            "frozen_experience": {
                "experience_id": experience_id,
                "version": "v1",
                "candidate_hash": candidate_hash,
            },
        },
    }


def test_freeze_select_preregister_and_dry_run(tmp_path, monkeypatch):
    engine, sessions, original_tags_type = _sqlite_stack(tmp_path, monkeypatch)
    try:
        memory = _build_memory(sessions)
        experience_service = ExecutionExperienceService(
            memory=memory,
            cfg=_experience_cfg("placeholder", "placeholder"),
        )
        candidate = _approved_candidate(memory, experience_service)
        experience_id = candidate["experience_id"]
        candidate_hash = candidate["candidate"]["candidate_hash"]
        # Approve so the candidate is eligible to be frozen (still not promoted).
        experience_service.review_candidate(
            conversation_id=44,
            execution_id=memory.chats.record["execution_id"],
            reviewer="synthetic-reviewer",
            decision="APPROVE",
            review_id="candidate-review-fixture",
            notes="Synthetic fixture approval.",
            evidence_refs=[
                f"experience:{experience_id}",
                f"plan_trace:{memory.chats.record['trace_id']}",
                "verification:python3 -m unittest:passed:0",
            ],
        )
        # Withdraw it again so the frozen input must stay non-global.
        approved = memory.casebooks.get_execution_experience_candidate(experience_id)
        memcube_id = approved["candidate"]["memcube_id"]
        experience_service.deactivate_experience(
            conversation_id=44,
            experience_id=experience_id,
            reviewer="synthetic-reviewer",
            reason="Withdraw for the experiment-scoped fixture.",
            review_id="withdraw-fixture",
        )
        assert memory.memcubes.get_by_id(memcube_id).extra_data["reuse_eligible"] is False
        # The withdrawal label is now consistent on the source execution.
        assert memory.chats.record["experience_status"] == "WITHDRAWN"

        cfg = _experience_cfg(experience_id, candidate_hash)
        experience_service = ExecutionExperienceService(memory=memory, cfg=cfg)
        service = Stn06ExperimentService(
            memory=memory, cfg=cfg, experience_service=experience_service,
        )

        frozen = service.freeze_experience()
        assert frozen["experience_id"] == experience_id
        assert frozen["candidate_hash"] == candidate_hash
        assert frozen["globally_retrievable"] is False
        assert frozen["withdrawal_preserved"] is True

        baseline = service.select_experience(ARM_BASELINE, "stn06-near-01")
        treatment = service.select_experience(ARM_EXPERIENCE, "stn06-near-01")
        assert baseline["context"] is None and baseline["selected_experience_ids"] == []
        assert treatment["selected_experience_ids"] == [experience_id]
        assert treatment["candidate_hash"] == candidate_hash
        assert "untrusted task context" in treatment["context"]

        authorization = service.authorize_scope(
            reviewer="operator",
            authorization_id="stn06-auth-fixture",
            notes="Authorize the frozen experience for the experiment scope only.",
        )
        assert authorization["global_reactivation"] is False

        persisted = service.persist_preregistration(authorization=authorization)
        assert persisted["preregistration_hash"]
        variants = memory.experiments.list_variants(persisted["experiment_id"])
        assert {variant["name"] for variant in variants} == {"A", "B"}
        stored = memory.experiments.get_by_name(EXPERIMENT_NAME, domain="python")
        assert stored["config"]["stn06_preregistration"]["preregistration_hash"] == persisted["preregistration_hash"]

        # The withdrawn experience stays unavailable to ordinary retrieval.
        assert experience_service.retrieve_for_task("Fix a Python arithmetic defect.")["results"] == []

        # The preregistration hash is reproducible despite volatile timestamps.
        assert service.preregistration()["preregistration_hash"] == service.preregistration()["preregistration_hash"]

        report = service.dry_run(scope="pilot", verify_oracles=True)
        assert report["model_calls_made"] == 0
        assert report["trials_persisted"] == 0
        assert report["planned_executions"] == 4
        assert report["oracle_verification"]["all_verified"] is True
        assert report["normal_retrieval"]["withdrawn_unavailable"] is True
        for task in report["tasks"]:
            assert task["baseline"]["context_present"] is False
            assert task["baseline"]["experience_ids"] == []
            assert task["treatment"]["experience_ids"] == [experience_id]
            assert task["scoring_path"]["simulated_fail"]["success"] is False
            assert task["scoring_path"]["simulated_pass"]["success"] is True
    finally:
        engine.dispose()
        monkeypatch.setattr(CaseBookORM.__table__.c.tags, "type", original_tags_type)


def test_freeze_rejects_globally_active_memcube(tmp_path, monkeypatch):
    engine, sessions, original_tags_type = _sqlite_stack(tmp_path, monkeypatch)
    try:
        memory = _build_memory(sessions)
        experience_service = ExecutionExperienceService(memory=memory, cfg=_experience_cfg("placeholder", "placeholder"))
        candidate = _approved_candidate(memory, experience_service)
        experience_id = candidate["experience_id"]
        candidate_hash = candidate["candidate"]["candidate_hash"]
        experience_service.review_candidate(
            conversation_id=44,
            execution_id=memory.chats.record["execution_id"],
            reviewer="synthetic-reviewer",
            decision="APPROVE",
            review_id="candidate-review-fixture",
            notes="Synthetic fixture approval.",
            evidence_refs=[
                f"experience:{experience_id}",
                f"plan_trace:{memory.chats.record['trace_id']}",
                "verification:python3 -m unittest:passed:0",
            ],
        )
        service = Stn06ExperimentService(
            memory=memory,
            cfg=_experience_cfg(experience_id, candidate_hash),
            experience_service=experience_service,
        )
        try:
            service.freeze_experience()
        except Stn06Error as exc:
            assert exc.code == "MEMCUBE_GLOBALLY_ACTIVE"
        else:
            raise AssertionError("Freezing a globally active MemCube must fail")
    finally:
        engine.dispose()
        monkeypatch.setattr(CaseBookORM.__table__.c.tags, "type", original_tags_type)


def test_score_outcome_enforces_scope_and_prohibition():
    task = manifest_mod.task_by_id("stn06-near-01")
    in_scope = Stn06ExperimentService.score_outcome(
        task, oracle_exit_code=0, modified_paths=["mathops.py"],
    )
    assert in_scope["success"] is True
    prohibited = Stn06ExperimentService.score_outcome(
        task, oracle_exit_code=0, modified_paths=["mathops.py", "tests/test_mathops.py"],
    )
    assert prohibited["success"] is False
    assert prohibited["prohibited_hits"] == ["tests/test_mathops.py"]
    out_of_scope = Stn06ExperimentService.score_outcome(
        task, oracle_exit_code=0, modified_paths=["mathops.py", "extra.py"],
    )
    assert out_of_scope["success"] is False
    assert out_of_scope["out_of_scope"] == ["extra.py"]
    no_change = Stn06ExperimentService.score_outcome(task, oracle_exit_code=0, modified_paths=[])
    assert no_change["success"] is False
