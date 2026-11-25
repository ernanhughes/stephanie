from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import configure_mappers, sessionmaker

from stephanie.improvement.engine import EVALUATOR_DIGEST, ImprovementEngine
from stephanie.improvement.models import Candidate, Evidence, Hypothesis, Opportunity

FIXTURES = Path(__file__).parents[1] / "stephanie" / "improvement" / "harness_fixtures.json"


def seeded(tmp_path):
    engine = ImprovementEngine(tmp_path)
    engine.import_json(FIXTURES)
    opportunity = engine.opportunities()[0]
    candidate = engine.construct_candidate(opportunity)
    return engine, opportunity, candidate


def test_harness_import_and_candidate_identity_preserve_provenance(tmp_path):
    engine, opportunity, candidate = seeded(tmp_path)
    assert len(opportunity.evidence_ids) == 4
    assert candidate.candidate_id == engine.construct_candidate(opportunity).candidate_id
    assert all(row["cost_usd"] is None and row["cost_status"] == "unknown" for row in engine._read(engine.evidence_path, []))
    assert all(row["source_digest"] for row in engine._read(engine.evidence_path, []))
    assert all("oracle_verified" not in row for row in engine._read(engine.evidence_path, []))
    assert len(candidate.evidence_digests) == len(candidate.evidence_ids)


def test_candidate_rejects_evidence_digest_drift(tmp_path):
    engine, _, candidate = seeded(tmp_path)
    rows = engine._read(engine.evidence_path, [])
    rows[0]["evaluation_status"] = "FAIL"
    engine.evidence_path.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError, match="source digests"):
        engine.validate_candidate(candidate, authorized=True)


def test_p1_candidate_resolves_known_false_verified_defect_against_frozen_oracle(tmp_path):
    engine, _, candidate = seeded(tmp_path)
    evaluation = engine.evaluate_fixtures(candidate)
    assert evaluation.status == "PASS_FIXTURE_ONLY"
    assert evaluation.baseline_score == 0.0
    assert evaluation.candidate_score == 1.0


def test_p2_no_measurable_improvement_is_inconclusive(tmp_path):
    engine = ImprovementEngine(tmp_path)
    rows = json.loads(FIXTURES.read_text(encoding="utf-8"))
    for row in rows:
        row["evaluation_status"] = "PASS"
        row["acceptance_status"] = "ACCEPTED"
        row["execution_completed"] = True
    engine.import_evidence(rows)
    opportunity = Opportunity(opportunity_id="controlled", title="control", evidence_ids=tuple(r["evidence_id"] for r in rows), observed_gap="controlled", confidence="medium")
    candidate = engine.construct_candidate(opportunity)
    result = engine.evaluate_fixtures(candidate)
    assert result.status == "INCONCLUSIVE"
    assert result.baseline_score == result.candidate_score == 1.0


def test_p3_regression_and_p7_evaluator_tampering_are_rejected(tmp_path):
    engine, _, candidate = seeded(tmp_path)
    tampered = candidate.model_copy(update={"protected_evaluator_digest": "0" * 64})
    with pytest.raises(ValueError, match="weaken"):
        engine.validate_candidate(tampered, authorized=True)
    # A changed stored artifact with the same candidate identity fails immutability.
    path = engine.candidates_path
    rows = engine._read(path, []); rows[0]["intervention"] = "weaken evaluator"; path.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError, match="immutable"):
        engine.validate_candidate(candidate, authorized=True)


def test_p3_candidate_regression_is_measured_against_protected_oracle(tmp_path, monkeypatch):
    engine, opportunity, _ = seeded(tmp_path)
    records = engine._read(engine.evidence_path, [])
    for row in records:
        row["evaluation_status"] = "PASS"
        row["acceptance_status"] = "ACCEPTED"
    engine.evidence_path.write_text(json.dumps(records), encoding="utf-8")
    candidate = engine.construct_candidate(opportunity)
    original = ImprovementEngine.classify
    monkeypatch.setattr(ImprovementEngine, "classify", staticmethod(lambda row, item=None: "UNVERIFIED" if item is not None else original(row, item)))
    result = engine.evaluate_fixtures(candidate)
    assert result.status == "REJECTED"
    assert result.candidate_score == 0.0


def test_p4_insufficient_evidence_abstains_and_p5_unauthorized_rejects(tmp_path):
    engine = ImprovementEngine(tmp_path)
    engine.import_evidence(json.loads(FIXTURES.read_text(encoding="utf-8"))[:1])
    opp = engine.opportunities()[0]
    assert opp.abstained
    with pytest.raises(ValueError, match="abstained"):
        engine.construct_candidate(opp)
    engine, _, candidate = seeded(tmp_path / "authorized")
    with pytest.raises(PermissionError):
        engine.validate_candidate(candidate, authorized=False)


def test_p6_compute_requirement_must_be_named_and_is_retained(tmp_path):
    engine, opportunity, candidate = seeded(tmp_path)
    needs_compute = engine.construct_candidate(opportunity, additional_compute="2x model inference token budget")
    missing = candidate.model_copy(update={"requires_additional_compute": True, "additional_compute": None})
    with pytest.raises(ValueError, match="must be identified"):
        engine.validate_candidate(missing, authorized=True)
    result = engine.evaluate_fixtures(needs_compute)
    assert result.status == "INCONCLUSIVE"
    assert "compute-matched" in result.reason
    assert result.additional_compute == "2x model inference token budget"
    assert result.additional_compute_identified is True


def test_p8_relevant_experience_retrieves_and_p9_irrelevant_does_not(tmp_path):
    engine, _, candidate = seeded(tmp_path)
    engine.evaluate_fixtures(candidate)
    assert engine.retrieve_experience("Harness evidence gate acceptance")
    assert engine.retrieve_experience("unrelated astrophysics theorem") == []


@pytest.mark.long
@pytest.mark.long
def test_persistence_uses_existing_experiment_and_casebook_stores_in_isolated_sqlite(tmp_path):
    # Register Stephanie's existing ORM mappers, then use only an isolated file DB.
    from stephanie.memory.memory_tool import MemoryTool  # noqa: F401
    from stephanie.memory.casebook_store import CaseBookStore
    from stephanie.memory.experiment_store import ExperimentStore
    from stephanie.orm.base import Base
    from stephanie.orm.casebook import CaseBookORM, CaseORM, CaseScorableORM
    from stephanie.orm.experiment import ExperimentORM, VariantORM, TrialORM, TrialMetricORM

    configure_mappers()
    sql_engine = create_engine(f"sqlite:///{tmp_path / 'isolated.sqlite3'}")
    tables = [CaseBookORM.__table__, CaseORM.__table__, CaseScorableORM.__table__, ExperimentORM.__table__,
              VariantORM.__table__, TrialORM.__table__, TrialMetricORM.__table__]
    Base.metadata.create_all(sql_engine, tables=tables)
    Session = sessionmaker(bind=sql_engine)
    stores = ImprovementEngine(tmp_path / "artifacts", experiment_store=ExperimentStore(Session), casebook_store=CaseBookStore(Session))
    stores.import_json(FIXTURES)
    candidate = stores.construct_candidate(stores.opportunities()[0])
    stores.evaluate_fixtures(candidate)
    assert stores.experiment_store.list(limit=5)
    assert stores.casebook_store.get_by_name("stephanie-evolution-harness") is not None
    assert stores.retrieve_experience("Harness evidence acceptance")
    assert stores.retrieve_experience("unrelated astrophysics") == []
