"""Stage 2 tests (§34): invariants, repository, legacy comparison, migration."""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from datetime import datetime


@contextmanager
def raises(exc_type):
    try:
        yield
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__}")

from stephanie.evaluation import (
    CANONICAL_ONLY,
    LEGACY_ONLY,
    MATCH,
    UNEXPECTED_DIVERGENCE,
    ComparisonOutcome,
    Criterion,
    Evaluation,
    EvaluationAttribute,
    EvaluationObservation,
    EvaluationProvenance,
    EvaluatorRef,
    EvidenceRef,
    FusedScore,
    FusionSpec,
    InMemoryEvaluationRepository,
    Interpretation,
    Score,
    ScoreAttribute,
    ScoreScale,
    SubjectRef,
    compare_evaluations,
    fuse_weighted_mean,
    validate_confidence,
)
from stephanie.evaluation.adapters import stephanie_legacy, writer_legacy
from stephanie.evaluation.diagnostics import LEGACY_SNAPSHOT_DIVERGENCE, VERDICT_AMBIGUOUS
from stephanie.evaluation.reader import latest_score_for_dimension
from stephanie.evaluation.writer import append_observation, supersede
from stephanie.services.evaluation_runtime import EvaluationRuntime


def _run(coro):
    return asyncio.run(coro)


def _subject():
    return SubjectRef(subject_type="model.response", subject_id="req_1")


def _criterion():
    return Criterion(name="technical_correctness", version="v1", scale=ScoreScale(0.0, 1.0))


def _evaluator():
    return EvaluatorRef(name="judge")


def _eval(score_value=0.7, confidence=0.8, eid="eval_1"):
    return Evaluation(
        evaluation_id=eid,
        subject=_subject(),
        criterion=_criterion(),
        evaluator=_evaluator(),
        created_at=datetime(2026, 1, 1),
        confidence=confidence,
    )


def _score(value=0.7, dim="correctness", eid="eval_1", sid="score_1"):
    return Score(score_id=sid, evaluation_id=eid, dimension=dim, value=value)


# -- Type invariants ----------------------------------------------------


def test_subject_ref_coerces_int_ids_to_strings():
    subject = SubjectRef(subject_type="document", subject_id=123)
    assert subject.subject_id == "123"
    assert subject.key == ("document", "123")


def test_missing_confidence_stays_missing():
    evaluation = _eval(confidence=None)
    assert evaluation.confidence is None
    assert validate_confidence(None) is None


def test_confidence_out_of_range_rejected():
    with raises(ValueError):
        _eval(confidence=1.5)
    with raises(ValueError):
        validate_confidence(-0.1)


def test_score_value_never_silently_normalized():
    score = Score(score_id="s", evaluation_id="e", dimension="q", value=71.0,
                  scale=ScoreScale(0.0, 100.0))
    assert score.value == 71.0  # raw stays raw; no /100+clamp


def test_interpretation_is_not_decision():
    interp = Interpretation(namespace="claim_support", value="partially_supported")
    assert str(interp) == "claim_support:partially_supported"
    assert interp.value not in {"KEEP", "REVERT"}


def test_fusion_recomputable_and_versioned():
    scores = [_score(0.5, dim="a", sid="a"), _score(1.0, dim="b", sid="b")]
    spec = FusionSpec(fusion_id="f1", version="v1", method="weighted_mean",
                      weights={"a": 1.0, "b": 3.0})
    fused = fuse_weighted_mean(scores, spec)
    assert abs(fused.value - 0.875) < 1e-9
    assert fused.fusion_spec_id == "f1@v1"
    assert fused.component_score_ids == ("a", "b")
    # Recompute gives the same result.
    assert fuse_weighted_mean(scores, spec) == fused


# -- Repository ----------------------------------------------------------


def test_append_read_latest_and_dimension_lookup():
    from dataclasses import replace
    repo = InMemoryEvaluationRepository()
    _run(repo.append(_eval(eid="e1"), [_score(eid="e1")]))
    e2 = replace(_eval(eid="e2"), created_at=datetime(2026, 1, 2))
    _run(repo.append(e2, [_score(0.9, eid="e2")]))
    latest = _run(repo.latest(_subject(), "technical_correctness"))
    assert latest.evaluation_id == "e2"
    assert _run(latest_score_for_dimension(repo, _subject(), "technical_correctness", "correctness")).value == 0.9
    assert _run(latest_score_for_dimension(repo, _subject(), "technical_correctness", "absent")) is None


def test_supersession_deactivates_old():
    repo = InMemoryEvaluationRepository()
    old = _eval(eid="old")
    _run(repo.append(old, [_score(eid="old")]))
    obs = EvaluationObservation(subject=_subject(), criterion=_criterion(),
                                evaluator=_evaluator(), scores=[_score(0.95, eid="", sid="")])
    new = _run(supersede(repo, old, obs))
    assert new.supersedes_id == "old"
    assert _run(repo.get("old")).is_active is False
    assert _run(repo.latest(_subject(), "technical_correctness")).evaluation_id == new.evaluation_id


def test_duplicate_append_rejected_idempotent_retry_safe():
    repo = InMemoryEvaluationRepository()
    _run(repo.append(_eval(eid="dup"), [_score(eid="dup")]))
    with raises(ValueError):
        _run(repo.append(_eval(eid="dup"), [_score(eid="dup")]))


def test_purge_cascades_score_attributes():
    repo = InMemoryEvaluationRepository()
    _run(repo.append(_eval(eid="e1"), [_score(eid="e1", sid="s1")]))
    _run(repo.add_score_attributes([ScoreAttribute(score_id="s1", namespace="judge", name="raw", value="x")]))
    _run(repo.purge("e1"))
    assert _run(repo.get("e1")) is None
    assert _run(repo.scores("e1")) == []
    assert "s1" not in repo._score_attrs  # no orphans


def test_evidence_linking():
    repo = InMemoryEvaluationRepository()
    _run(repo.append(_eval(eid="e1"), [_score(eid="e1")]))
    from stephanie.evaluation.evidence import EvaluationEvidenceLink

    _run(repo.link_evidence(EvaluationEvidenceLink(evaluation_id="e1", evidence_id="ev1", relationship="supports")))
    assert repo.evidence_links[0].evidence_id == "ev1"


# -- Legacy comparison ----------------------------------------------------


def test_stephanie_json_only_row_readable():
    class Row:
        id = 7
        scorable_type = "hypothesis"
        scorable_id = "42"
        strategy = "relevance"
        evaluator_name = "llm"
        created_at = datetime(2026, 1, 2)
        pipeline_run_id = None
        model_name = "qwen3"
        source = "llm"
        agent_name = "llm"
        scores = {"relevance": {"score": 0.6, "source": "llm"}}

    evaluation, scores, diags = stephanie_legacy.evaluation_orm_to_canonical(Row(), None)
    assert evaluation.evaluation_id == "legacy:7"
    assert evaluation.is_active is True  # legacy rows treated as active
    assert scores[0].value == 0.6
    assert scores[0].source == "legacy_snapshot"
    assert diags == []


def test_stephanie_snapshot_row_divergence_flagged():
    class ScoreRow:
        id = 1
        dimension = "relevance"
        score = 0.9
        weight = 1.0
        source = "llm"
        rationale = ""

    class Row:
        id = 8
        scorable_type = "hypothesis"
        scorable_id = "43"
        strategy = "relevance"
        evaluator_name = "llm"
        created_at = datetime(2026, 1, 2)
        pipeline_run_id = None
        model_name = "qwen3"
        source = "llm"
        agent_name = "llm"
        scores = {"relevance": {"score": 0.2}}

    _, _, diags = stephanie_legacy.evaluation_orm_to_canonical(Row(), [ScoreRow()])
    assert any(d.code == LEGACY_SNAPSHOT_DIVERGENCE for d in diags)


def test_writer_verdict_mapping_and_experiment_quarantine():
    interp, diag = writer_legacy.verdict_to_interpretation("needs_review")
    assert interp.value == "needs_review" and diag is None
    interp2, diag2 = writer_legacy.verdict_to_interpretation("REVERT")
    assert interp2 is None and diag2.code == VERDICT_AMBIGUOUS


def test_compare_harness_match_and_divergence():
    subject = _subject()
    legacy = (_eval(eid="l1"), [_score(0.7, eid="l1", sid="ls1")])
    canonical = (_eval(eid="c1"), [_score(0.7, eid="c1", sid="cs1")])
    assert compare_evaluations(subject, "technical_correctness", legacy, canonical).verdict == MATCH
    other = (_eval(eid="c2"), [_score(0.1, eid="c2", sid="cs2")])
    assert compare_evaluations(subject, "technical_correctness", legacy, other).verdict == UNEXPECTED_DIVERGENCE
    assert compare_evaluations(subject, "technical_correctness", legacy, None).verdict == LEGACY_ONLY
    assert compare_evaluations(subject, "technical_correctness", None, canonical).verdict == CANONICAL_ONLY


# -- Runtime + migration ---------------------------------------------------


def test_runtime_record_with_model_provenance():
    runtime = EvaluationRuntime()
    obs = EvaluationObservation(
        subject=_subject(), criterion=_criterion(), evaluator=_evaluator(),
        scores=[_score(eid="", sid="")], confidence=0.8, confidence_source="judge_self_report",
        evidence=[EvidenceRef(evidence_id="ev1", evidence_type="document")],
        attributes=[EvaluationAttribute(evaluation_id="", namespace="sicql", name="q_value", value=0.5)],
    )
    evaluation = _run(
        runtime.record_from_model_invocation(
            obs, model_id="ollama:qwen3", request_id="req_1",
            trace_id="t1", task_type="research.claim.verify", provider="ollama",
        )
    )
    assert evaluation.model_id == "ollama:qwen3"
    provenance = evaluation.metadata["provenance"]
    assert provenance["metadata"]["model_request_id"] == "req_1"
    assert provenance["metadata"]["task_type"] == "research.claim.verify"
    assert provenance["trace_id"] == "t1"
    assert len(runtime.repository.evidence_links) == 1


def test_dual_write_tracks_successes_separately():
    runtime = EvaluationRuntime()
    obs = EvaluationObservation(subject=_subject(), criterion=_criterion(),
                                evaluator=_evaluator(), scores=[_score(eid="", sid="")])

    async def good_legacy(observation):
        return True

    evaluation, legacy_ok, canonical_ok = _run(runtime.dual_write(obs, good_legacy))
    assert legacy_ok and canonical_ok and evaluation is not None

    async def bad_legacy(observation):
        raise RuntimeError("legacy down")

    # Legacy failure must not fail production (canonical still records).
    evaluation2, legacy_ok2, canonical_ok2 = _run(runtime.dual_write(obs, bad_legacy))
    assert legacy_ok2 is False and canonical_ok2 is True
