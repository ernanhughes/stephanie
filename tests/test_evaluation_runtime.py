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


# -- Stage 2.5: attribute persistence round-trip (sqlite) --------------------


def _sqlite_repo():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from stephanie.evaluation.persistence.orm import CanonicalBase
    from stephanie.evaluation.persistence.repository import SqlAlchemyEvaluationRepository

    engine = create_engine("sqlite:///:memory:")
    CanonicalBase.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    return SqlAlchemyEvaluationRepository(factory)


def test_sqlite_canonical_round_trip_with_attributes():
    repo = _sqlite_repo()
    obs = EvaluationObservation(
        subject=_subject(), criterion=_criterion(), evaluator=_evaluator(),
        scores=[Score(score_id="s1", evaluation_id="", dimension="correctness", value=0.71,
                       scale=ScoreScale(0.0, 1.0), weight=2.0, confidence=0.82,
                       confidence_source="judge_self_report", scorer="judge",
                       rationale="solid")],
        confidence=0.82, confidence_source="judge_self_report",
        task_type="research.claim.verify", model_id="ollama:qwen3", run_id="run_7",
    )
    obs.attributes = [EvaluationAttribute(evaluation_id="", namespace="sicql", name="q_value", value=0.5)]
    obs.score_attributes = [ScoreAttribute(score_id="s1", namespace="judge", name="raw", value="x")]
    evaluation = _run(append_observation_sql(repo, obs))
    fetched = _run(repo.get(evaluation.evaluation_id))
    assert fetched.task_type == "research.claim.verify"
    assert fetched.model_id == "ollama:qwen3"
    assert fetched.confidence == 0.82
    scores = _run(repo.scores(evaluation.evaluation_id))
    assert scores[0].value == 0.71 and scores[0].scale.maximum == 1.0
    eval_attrs = _run(repo.evaluation_attributes(evaluation.evaluation_id))
    assert eval_attrs[0].qualified_name == "sicql.q_value"
    score_attrs = _run(repo.score_attributes("s1"))
    assert score_attrs[0].qualified_name == "judge.raw"


def append_observation_sql(repo, observation):
    from stephanie.evaluation.writer import append_observation as _append

    async def _go():
        evaluation = await _append(repo, observation)
        # Re-target attribute rows at the assigned evaluation id.
        from dataclasses import replace

        fixed_eval_attrs = [
            replace(a, evaluation_id=evaluation.evaluation_id) for a in observation.attributes
        ]
        if fixed_eval_attrs:
            await repo.add_evaluation_attributes(fixed_eval_attrs)
        if observation.score_attributes:
            await repo.add_score_attributes(observation.score_attributes)
        return evaluation

    return _go()


# -- Stage 2.5: save_bundle shadow hook --------------------------------------


def test_shadow_hook_off_by_default():
    import os

    from stephanie.evaluation.shadow import agent_family, maybe_shadow_bundle

    os.environ.pop("STEPHANIE_CANONICAL_SHADOW", None)
    assert agent_family("llm") == "A"
    assert agent_family("scorable_ranker") == "?"
    attempted, ok, err = maybe_shadow_bundle(bundle=None, scorable=None, agent_name="llm")
    assert (attempted, ok, err) == (False, True, None)


def test_shadow_hook_family_gate():
    import os

    from stephanie.evaluation.shadow import maybe_shadow_bundle

    os.environ["STEPHANIE_CANONICAL_SHADOW"] = "1"
    os.environ["STEPHANIE_SHADOW_FAMILIES"] = "A"
    try:
        attempted, ok, _ = maybe_shadow_bundle(bundle=None, scorable=None, agent_name="scorable_ranker")
        assert attempted is False  # family B not enabled
    finally:
        os.environ.pop("STEPHANIE_CANONICAL_SHADOW", None)
        os.environ.pop("STEPHANIE_SHADOW_FAMILIES", None)


def test_shadow_hook_writes_canonical_rows_sqlite():
    import os
    import tempfile

    import stephanie.evaluation.shadow as shadow_mod
    from stephanie.evaluation.shadow import maybe_shadow_bundle

    os.environ["STEPHANIE_CANONICAL_SHADOW"] = "1"
    os.environ["STEPHANIE_SHADOW_FAMILIES"] = "A"
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    try:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        from stephanie.evaluation.persistence.orm import CanonicalBase
        from stephanie.evaluation.persistence.repository import SqlAlchemyEvaluationRepository

        # File-backed sqlite: shared across the shadow worker thread
        # (":memory:" gives each connection a fresh empty DB).
        engine = create_engine(f"sqlite:///{tmp.name}")
        CanonicalBase.metadata.create_all(engine)
        shadow_mod._repo = SqlAlchemyEvaluationRepository(
            sessionmaker(bind=engine, expire_on_commit=False)
        )

        class FakeResult:
            dimension = "relevance"
            score = 0.66
            weight = 1.0
            source = "llm"
            rationale = "r"

        class FakeBundle:
            results = {"relevance": FakeResult()}

        class FakeScorable:
            target_type = "hypothesis"
            id = "h9"
            text = "t"

        attempted, ok, err = maybe_shadow_bundle(
            bundle=FakeBundle(), scorable=FakeScorable(), agent_name="llm",
            evaluator="llm", model_name="ollama:qwen3", source="llm",
            strategy="relevance",
        )
        assert attempted and ok, err
        history = _run(shadow_mod._repo.performance_history(
            model_id="ollama:qwen3", criterion="relevance"))
        assert len(history) == 1
        assert history[0].subject.key == ("hypothesis", "h9")
    finally:
        shadow_mod._repo = None
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        os.environ.pop("STEPHANIE_CANONICAL_SHADOW", None)
        os.environ.pop("STEPHANIE_SHADOW_FAMILIES", None)


def test_performance_history_gate_query():
    repo = InMemoryEvaluationRepository()
    base = _eval(eid="h1")
    from dataclasses import replace

    _run(repo.append(base, [_score(eid="h1", sid="hs1")]))
    _run(repo.append(
        replace(base, evaluation_id="h2", model_id="ollama:qwen3",
                task_type="research.claim.verify",
                created_at=datetime(2026, 2, 1)),
        [_score(0.9, eid="h2", sid="hs2")],
    ))
    history = _run(repo.performance_history(
        model_id="ollama:qwen3", task_type="research.claim.verify",
        criterion="technical_correctness"))
    assert [e.evaluation_id for e in history] == ["h2"]
    assert _run(repo.performance_history(model_id="nope")) == []


# -- Stage 2.5: causal chain -------------------------------------------------


def test_invocation_to_evaluation_causal_chain():
    from stephanie.evaluation import EvaluationContext
    from stephanie.models import ModelRequest
    from stephanie.models import Model as RuntimeModel
    from stephanie.services.model_runtime import ModelRuntime
    from stephanie.models import StubProvider

    model = RuntimeModel.from_ref("stub:cheap")
    runtime = ModelRuntime()
    runtime.register_model(model)
    runtime.register_provider("stub", StubProvider())
    request = ModelRequest(model="stub:cheap", prompt="verify this claim",
                           task_type="research.claim.verify", trace_id="trace_9")
    response = _run(runtime.invoke(request))

    context = EvaluationContext.from_model_response(
        task_type=request.task_type, request_id=response.request_id,
        trace_id=request.trace_id, model_id=response.model_id, provider=response.provider,
    )
    obs = EvaluationObservation(
        subject=_subject(), criterion=_criterion(), evaluator=_evaluator(),
        scores=[Score(score_id="", evaluation_id="", dimension="correctness", value=0.8)],
    )
    context.apply_to(obs, "judge")
    evaluation_runtime = EvaluationRuntime()
    evaluation = _run(evaluation_runtime.record(obs))

    summary = context.chain_summary(evaluation_id=evaluation.evaluation_id)
    assert summary["request_id"] == response.request_id
    assert summary["model_id"] == "stub:cheap"
    assert evaluation.task_type == "research.claim.verify"
    assert evaluation.model_id == "stub:cheap"
    assert evaluation.metadata["provenance"]["trace_id"] == "trace_9"
    # Gate: performance history answers the portfolio question.
    history = _run(evaluation_runtime.repository.performance_history(
        model_id="stub:cheap", task_type="research.claim.verify",
        criterion="technical_correctness"))
    assert [e.evaluation_id for e in history] == [evaluation.evaluation_id]
