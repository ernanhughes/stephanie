from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace

from sqlalchemy import JSON, create_engine
from sqlalchemy.orm import configure_mappers, sessionmaker
from hydra import compose, initialize_config_dir

# Import the normal MemoryTool store registry before configuring relationships.
from stephanie.memory.memory_tool import MemoryTool as _RegisterMemoryStores
from stephanie.memory.casebook_store import CaseBookStore
from stephanie.memory.goal_store import GoalStore
from stephanie.memory.memcube_store import MemCubeStore
from stephanie.orm.base import Base
from stephanie.orm.casebook import CaseBookORM, CaseORM, CaseScorableORM
from stephanie.orm.goal import GoalORM
from stephanie.orm.memcube import MemCubeORM
from stephanie.orm.training_stats import TrainingStatsORM
from stephanie.services.execution_experience import ExecutionExperienceService
from stephanie.memory.experiment_store import _stable_variant_index


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
        assert conversation_id == self.record["conversation_id"]
        return self.messages

    def update_external_execution(self, conversation_id, execution_id, changes):
        assert conversation_id == self.record["conversation_id"]
        assert execution_id == self.record["execution_id"]
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


def test_normal_store_registry_resolves_training_stats_and_experience_mappers():
    configure_mappers()
    assert TrainingStatsORM.__mapper__.configured
    assert CaseBookORM.__mapper__.configured
    assert MemCubeORM.__mapper__.configured


def test_experience_and_retrieval_stay_disabled_in_checked_in_sis_defaults():
    from pathlib import Path

    config_dir = str((Path(__file__).resolve().parents[1] / "config").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="sis")
    assert cfg.experience_learning.enabled is False
    assert cfg.experience_learning.model_assisted_extraction is False
    assert cfg.experience_learning.retrieval.enabled is False


def test_experiment_assignment_is_stable_across_python_hash_seeds():
    expected = _stable_variant_index("stn06-coding-transfer", "python", 7301, 2)
    code = (
        "from stephanie.memory.experiment_store import _stable_variant_index; "
        "print(_stable_variant_index('stn06-coding-transfer', 'python', 7301, 2))"
    )
    outputs = []
    for seed in ("1", "177", "random"):
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = seed
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        outputs.append(int(result.stdout.strip()))
    assert outputs == [expected, expected, expected]


def test_casebook_candidate_and_approved_memcube_persist_and_retrieve_in_isolated_sqlite(tmp_path, monkeypatch):
    configure_mappers()
    original_tags_type = CaseBookORM.__table__.c.tags.type
    # CaseBook's production tags use PostgreSQL JSONB. Use JSON for this
    # isolated SQLite integration fixture; production ORM metadata is restored.
    monkeypatch.setattr(CaseBookORM.__table__.c.tags, "type", JSON())
    engine = create_engine(f"sqlite:///{tmp_path / 'stn05-learning.sqlite'}")
    try:
        Base.metadata.create_all(
            engine,
            tables=[
                GoalORM.__table__,
                CaseBookORM.__table__,
                CaseORM.__table__,
                CaseScorableORM.__table__,
                MemCubeORM.__table__,
            ],
        )
        sessions = sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)
        record = {
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
        trace = TraceFixture({
            "trace_id": record["trace_id"],
            "execution_id": record["execution_id"],
            "conversation_id": record["conversation_id"],
            "user_message_id": record["originating_message_id"],
            "worker_request_id": record["worker_request_id"],
            "backend": record["backend"],
            "requested_model": record["requested_model"],
        })
        memory = SimpleNamespace(
            chats=ChatFixture(record),
            plan_traces=TraceStoreFixture(trace),
            goals=GoalStore(sessions),
            casebooks=CaseBookStore(sessions),
            memcubes=MemCubeStore(sessions),
        )
        service = ExecutionExperienceService(
            memory=memory,
            cfg={"experience_learning": {
                "enabled": True,
                "model_assisted_extraction": False,
                "retrieval": {
                    "enabled": True,
                    "max_results": 3,
                    "candidate_pool": 20,
                    "max_context_chars": 1200,
                },
            }},
        )

        candidate = service.extract_candidate(conversation_id=44, execution_id=record["execution_id"])
        assert candidate["created"] is True
        assert candidate["reuse_eligible"] is False
        assert memory.memcubes.search(scorable_type="execution_experience", limit=10) == []
        casebook = memory.casebooks.get_execution_experience_candidate(candidate["experience_id"])
        assert casebook is not None
        assert casebook["candidate"]["source"]["worker_request_id"] == record["worker_request_id"]
        assert casebook["candidate"]["lesson_hypothesis"]["validation_status"] == "untested_for_transfer"

        experience_id = candidate["experience_id"]
        approved = service.review_candidate(
            conversation_id=44,
            execution_id=record["execution_id"],
            reviewer="synthetic-reviewer",
            decision="APPROVE",
            review_id="candidate-review-fixture",
            notes="Synthetic integration fixture only.",
            evidence_refs=[
                f"experience:{experience_id}",
                f"plan_trace:{record['trace_id']}",
                "verification:python3 -m unittest:passed:0",
            ],
        )
        assert approved["reuse_eligible"] is True
        assert approved["promotion_status"] == "ACTIVE"
        assert memory.memcubes.get_by_id(approved["memcube_id"]) is not None
        assert memory.casebooks.get_execution_experience_candidate(experience_id)["candidate"]["promotion_status"] == "ACTIVE"

        retrieved = service.retrieve_for_task("Fix a Python arithmetic function and run unit tests.")
        assert [row["experience_id"] for row in retrieved["results"]] == [experience_id]
        assert "untrusted task context" in retrieved["context"]
        assert retrieved["context_chars"] <= 1200
        assert service.retrieve_for_task("unrelated marine biology topic")["results"] == []

        withdrawn = service.deactivate_experience(
            conversation_id=44,
            experience_id=experience_id,
            reviewer="synthetic-reviewer",
            reason="Exercise withdrawal in an isolated test database.",
            review_id="withdraw-fixture",
        )
        assert withdrawn["candidate"]["reuse_eligible"] is False
        assert service.retrieve_for_task("Fix a Python arithmetic function and run unit tests.")["results"] == []
    finally:
        engine.dispose()
        monkeypatch.setattr(CaseBookORM.__table__.c.tags, "type", original_tags_type)
