"""Stage 3 tests: independence-first portfolio behavior (§26, partial)."""
from __future__ import annotations

import asyncio

from stephanie.evaluation import Criterion, EvaluatorRef, ScoreScale
from stephanie.models import Model as RuntimeModel
from stephanie.models import ModelRequest, StubProvider
from stephanie.portfolio import (
    INDEPENDENCE_VIOLATION,
    DisagreementAnalyzer,
    PortfolioBudget,
    PortfolioEvaluator,
    PortfolioExecutor,
    PortfolioPlanner,
    PortfolioPolicy,
    PortfolioRole,
    PortfolioSynthesizer,
    VerifierRegistry,
    conditional_failure_rate,
    exact_match_verifier,
    failure_overlap,
    joint_failure_rate,
    role_performance,
    unique_detection_rate,
    FailureObservation,
    MarginalValueComponents,
)
from stephanie.services.evaluation_runtime import EvaluationRuntime
from stephanie.services.model_runtime import ModelRuntime
from stephanie.services.portfolio_runtime import PortfolioRuntime


def _run(coro):
    return asyncio.run(coro)


def _models():
    return [
        RuntimeModel.from_ref("openai:strong", local=False),
        RuntimeModel.from_ref("ollama:breadth-cheap", local=True),
    ]


def _runtime():
    runtime = ModelRuntime()
    for model in _models():
        runtime.register_model(model)
    runtime.register_provider("openai", StubProvider(name="openai", echo_prefix="OPENAI:"))
    runtime.register_provider("ollama", StubProvider(name="ollama", echo_prefix="OLLAMA:"))
    return runtime


def _request():
    return ModelRequest(model="openai:strong", prompt="Summarize the claim.",
                        task_type="research.claim.verify", trace_id="t-port")


def _policy(**overrides):
    base = {"task_type": "research.claim.verify"}
    base.update(overrides)
    return PortfolioPolicy(**base)


def _criterion():
    return Criterion(name="correctness", scale=ScoreScale(0.0, 1.0))


def test_plan_replayable_and_explained():
    planner = PortfolioPlanner(_models())
    plan1 = _run(planner.plan(_request(), _policy()))
    plan2 = _run(planner.plan(_request(), _policy()))
    roles1 = [c.role for c in plan1.candidates]
    roles2 = [c.role for c in plan2.candidates]
    assert roles1 == roles2 == [PortfolioRole.PRIMARY, PortfolioRole.INDEPENDENT_REVIEWER,
                                PortfolioRole.BREADTH, PortfolioRole.SYNTHESIZER]
    assert all("reason" in c.metadata for c in plan1.candidates)
    assert len(plan1.rationale()) == 4


def test_plan_budget_trims_with_diagnostic():
    planner = PortfolioPlanner(_models())
    plan = _run(planner.plan(_request(), _policy(budget=PortfolioBudget(max_models=2))))
    assert [c.role for c in plan.candidates] == [PortfolioRole.PRIMARY, PortfolioRole.INDEPENDENT_REVIEWER]
    assert any("PORTFOLIO_BUDGET_EXCEEDED" in note for note in plan.metadata["diagnostics"])


def test_plan_no_eligible_model_raises():
    from stephanie.portfolio import PortfolioDiagnostic

    planner = PortfolioPlanner([])
    try:
        _run(planner.plan(_request(), _policy()))
    except PortfolioDiagnostic as exc:
        assert exc.code == "PORTFOLIO_NO_ELIGIBLE_MODEL"
    else:
        raise AssertionError("expected PortfolioDiagnostic")


def test_reviewer_uses_different_provider():
    planner = PortfolioPlanner(_models())
    plan = _run(planner.plan(_request(), _policy()))
    primary = next(c for c in plan.candidates if c.role == PortfolioRole.PRIMARY)
    reviewer = next(c for c in plan.candidates if c.role == PortfolioRole.INDEPENDENT_REVIEWER)
    assert primary.model_id != reviewer.model_id
    assert reviewer.metadata["provider"] != primary.metadata["provider"]


def test_independent_reviewer_never_sees_candidate_text():
    seen: dict[str, str] = {}

    class RecordingRuntime:
        def __init__(self, inner):
            self.inner = inner

        async def invoke(self, request):
            seen[request.model.id if hasattr(request.model, "id") else request.model] = request.prompt or ""
            return await self.inner.invoke(request)

    planner = PortfolioPlanner(_models())
    plan = _run(planner.plan(_request(), _policy()))
    executor = PortfolioExecutor(RecordingRuntime(_runtime()))
    executions = _run(executor.execute(plan))
    assert len(executions) == 4 and all(e.success for e in executions)
    reviewer_prompt = seen["ollama:breadth-cheap"]
    assert "OPENAI:" not in reviewer_prompt and "OLLAMA:" not in reviewer_prompt
    assert reviewer_prompt == "Summarize the claim."


def test_anchor_independent_role_raises():
    from stephanie.portfolio import PortfolioDiagnostic

    planner = PortfolioPlanner(_models())
    plan = _run(planner.plan(_request(), _policy()))
    executor = PortfolioExecutor(_runtime())
    reviewer = next(c for c in plan.candidates if c.role == PortfolioRole.INDEPENDENT_REVIEWER)
    try:
        _run(executor.execute_anchored(reviewer, [], ["some_candidate"]))
    except PortfolioDiagnostic as exc:
        assert exc.code == INDEPENDENCE_VIOLATION
    else:
        raise AssertionError("expected INDEPENDENCE_VIOLATION")


def test_critic_receives_anchors_explicitly():
    from stephanie.portfolio import PortfolioCandidate

    runtime = _runtime()
    critic_model = RuntimeModel.from_ref("openai:strong", local=False)

    critic = PortfolioCandidate(
        candidate_id="critic_1", model_id="openai:strong", role=PortfolioRole.CRITIC,
        request=ModelRequest(model=critic_model, prompt="Review this."),
        independence_group="anchored_critique",
    )
    executor = PortfolioExecutor(runtime)
    from stephanie.portfolio.executor import PortfolioExecution

    prior = [PortfolioExecution(execution_id="e1", candidate_id="primary_x", model_id="openai:strong",
                                role=PortfolioRole.PRIMARY, output_text="ANSWER-42",
                                request_id="r1", trace_id="t", latency_ms=1.0)]
    result = _run(executor.execute_anchored(critic, prior, ["primary_x"]))
    assert result.success and "ANSWER-42" in result.output_text
    assert result.anchored_on == ("primary_x",)


def test_execution_inherits_trace_and_usage():
    planner = PortfolioPlanner(_models())
    plan = _run(planner.plan(_request(), _policy()))
    executions = _run(PortfolioExecutor(_runtime()).execute(plan))
    for execution in executions:
        assert execution.trace_id == "t-port"
        assert execution.request_id
        assert execution.usage is not None and execution.latency_ms is not None


def test_evaluation_linkage_and_history():
    planner = PortfolioPlanner(_models())
    plan = _run(planner.plan(_request(), _policy()))
    executions = _run(PortfolioExecutor(_runtime()).execute(plan))
    eval_runtime = EvaluationRuntime()
    evaluator = PortfolioEvaluator(eval_runtime)
    evaluations = _run(evaluator.evaluate(
        executions, _criterion(), EvaluatorRef(name="judge"),
        lambda execution: [("correctness", 0.8, "ok")],
    ))
    assert len(evaluations) == 4
    assert all(e.model_id for e in evaluations)
    history = _run(eval_runtime.repository.performance_history(
        task_type="research.claim.verify", criterion="correctness"))
    assert len(history) == 4
    perf = _run(role_performance(eval_runtime.repository, task_type="research.claim.verify",
                                 criterion="correctness", role=PortfolioRole.PRIMARY))
    assert sum(p.observations for p in perf) >= 1


def test_disagreement_gap_and_unique_claims():
    from stephanie.evaluation import Evaluation
    from stephanie.portfolio.executor import PortfolioExecution

    execs = [
        PortfolioExecution("e1", "c1", "m1", PortfolioRole.PRIMARY, "The sky is blue. Water is wet.",
                           "r1", "t", 1.0, success=True),
        PortfolioExecution("e2", "c2", "m2", PortfolioRole.BREADTH, "The sky is blue. Markets crashed today.",
                           "r2", "t", 1.0, success=True),
    ]
    analyzer = DisagreementAnalyzer(gap_threshold=0.15)
    from stephanie.evaluation import EvaluatorRef as ER
    scored = {"e1": [type("S", (), {"dimension": "correctness", "value": 0.9})()],
              "e2": [type("S", (), {"dimension": "correctness", "value": 0.5})()]}
    evals = [
        Evaluation("e1", execs[0].output_text and __import__(
            "stephanie.evaluation.subject", fromlist=["SubjectRef"]).SubjectRef("s", "1"),
            _criterion(), ER("j"), __import__("datetime").datetime.utcnow(),
            metadata={"portfolio_candidate_id": "c1"}),
        Evaluation("e2", __import__("stephanie.evaluation.subject", fromlist=["SubjectRef"]).SubjectRef("s", "2"),
            _criterion(), ER("j"), __import__("datetime").datetime.utcnow(),
            metadata={"portfolio_candidate_id": "c2"}),
    ]
    disagreements = analyzer.analyze(execs, evals, scored)
    assert any(d.dimension == "correctness" and abs(d.severity - 0.4) < 1e-9 for d in disagreements)
    assert any(d.disagreement_type == "MISSING_INFORMATION" for d in disagreements)


def test_verification_overrides_consensus():
    from stephanie.portfolio.executor import PortfolioExecution
    from stephanie.portfolio.synthesis import PortfolioSynthesizer

    execs = [
        PortfolioExecution("e1", "c-high-1", "m1", PortfolioRole.PRIMARY, "The answer is 41.", "r1", "t", 1.0, success=True),
        PortfolioExecution("e2", "c-high-2", "m2", PortfolioRole.BREADTH, "The answer is 41 indeed.", "r2", "t", 1.0, success=True),
        PortfolioExecution("e3", "c-low", "m3", PortfolioRole.INDEPENDENT_REVIEWER, "The answer is 42.", "r3", "t", 1.0, success=True),
    ]
    registry = VerifierRegistry()
    registry.register(exact_match_verifier("The answer is 42."))
    verifications = registry.verify(execs)
    assert sum(1 for v in verifications if v.passed) >= 1

    from stephanie.evaluation import EvaluatorRef as ER
    import datetime as _dt
    from stephanie.evaluation.subject import SubjectRef as SR

    def _ev(cid, eid):
        return __import__("stephanie.evaluation.evaluation", fromlist=["Evaluation"]).Evaluation(
            eid, SR("s", cid), _criterion(), ER("j"), _dt.datetime.utcnow(),
            metadata={"portfolio_candidate_id": cid})

    evals = [_ev("c-high-1", "e1"), _ev("c-high-2", "e2"), _ev("c-low", "e3")]
    from stephanie.evaluation.score import Score as _S

    scores = {"e1": [_S("s1", "e1", "correctness", 0.95)], "e2": [_S("s2", "e2", "correctness", 0.93)],
              "e3": [_S("s3", "e3", "correctness", 0.60)]}
    outcome = PortfolioSynthesizer().select(execs, evals, [], verifications, scores)
    assert outcome.selected_candidate_id == "c-low"  # verified beats 2x higher-scored consensus


def test_selection_counts_quality_not_votes():
    from stephanie.portfolio.executor import PortfolioExecution
    from stephanie.portfolio.synthesis import PortfolioSynthesizer

    execs = [
        PortfolioExecution(f"e{i}", f"c{i}", "m", PortfolioRole.BREADTH, f"text {i}", f"r{i}", "t", 1.0, success=True)
        for i in range(3)
    ]
    from stephanie.evaluation import EvaluatorRef as ER
    import datetime as _dt
    from stephanie.evaluation.evaluation import Evaluation as _E
    from stephanie.evaluation.subject import SubjectRef as SR
    from stephanie.evaluation.score import Score as _S

    evals = [_E(f"e{i}", SR("s", f"c{i}"), _criterion(), ER("j"), _dt.datetime.utcnow(),
                metadata={"portfolio_candidate_id": f"c{i}"}) for i in range(3)]
    scores = {f"e{i}": [_S(f"s{i}", f"e{i}", "correctness", v)] for i, v in [(0, 0.6), (1, 0.61), (2, 0.9)]}
    outcome = PortfolioSynthesizer().select(execs, evals, [], [], scores)
    assert outcome.selected_candidate_id == "c2"


def test_synthesis_brief_cites_evidence():
    from stephanie.portfolio.executor import PortfolioExecution
    from stephanie.portfolio.synthesis import PortfolioSynthesizer

    execs = [PortfolioExecution("e1", "c1", "m", PortfolioRole.PRIMARY, "output", "r1", "t", 5.0, success=True)]
    seen: dict[str, str] = {}

    def _capture(brief: str) -> str:
        seen["brief"] = brief
        return "SYNTH"

    synth = PortfolioSynthesizer(synthesize_fn=_capture)
    outcome = synth.synthesize(execs, [], [], [])
    assert outcome.synthesized_text == "SYNTH"
    assert "candidate c1" in seen["brief"]


def test_independence_metrics_math():
    obs = [
        FailureObservation("t1", "A", PortfolioRole.PRIMARY, "c", True),
        FailureObservation("t1", "B", PortfolioRole.BREADTH, "c", True),
        FailureObservation("t1", "C", PortfolioRole.BREADTH, "c", False),
        FailureObservation("t2", "A", PortfolioRole.PRIMARY, "c", True),
        FailureObservation("t2", "B", PortfolioRole.BREADTH, "c", False),
        FailureObservation("t2", "C", PortfolioRole.BREADTH, "c", False),
        FailureObservation("t3", "A", PortfolioRole.PRIMARY, "c", False),
        FailureObservation("t3", "B", PortfolioRole.BREADTH, "c", False),
        FailureObservation("t3", "C", PortfolioRole.BREADTH, "c", True),
    ]
    assert joint_failure_rate(obs, "A", "B") == 1 / 3
    assert conditional_failure_rate(obs, "A", "B") == 0.5  # B repeats half of A's mistakes
    assert conditional_failure_rate(obs, "A", "C") == 0.0  # C catches all of A's failures
    assert failure_overlap(obs, "A", "B") == 0.5
    assert unique_detection_rate(obs, "C", ["A", "B"]) == 1 / 3  # C saves t1


def test_unique_detection_rate_counts_saves():
    obs = [
        FailureObservation("t1", "A", PortfolioRole.PRIMARY, "c", True),
        FailureObservation("t1", "B", PortfolioRole.BREADTH, "c", True),
        FailureObservation("t1", "C", PortfolioRole.BREADTH, "c", False),
        FailureObservation("t2", "A", PortfolioRole.PRIMARY, "c", False),
        FailureObservation("t2", "B", PortfolioRole.BREADTH, "c", False),
        FailureObservation("t2", "C", PortfolioRole.BREADTH, "c", False),
    ]
    assert unique_detection_rate(obs, "C", ["A", "B"]) == 0.5  # saves t1 of 2 eligible


def test_miv_transparent_components():
    miv = MarginalValueComponents(
        model_id="ollama:cheap", expected_quality_gain=0.05,
        expected_verification_gain=0.2, expected_unique_detection=0.27,
        expected_latency_ms=2000.0, correlated_failure_penalty=0.18,
    )
    value = miv.marginal_intelligence_value()
    assert value is not None and value > 0
    assert MarginalValueComponents(model_id="x").marginal_intelligence_value() is None


def test_full_portfolio_run_end_to_end():
    runtime = PortfolioRuntime(
        planner=PortfolioPlanner(_models()),
        executor=PortfolioExecutor(_runtime()),
        evaluator=PortfolioEvaluator(EvaluationRuntime()),
    )
    result = _run(runtime.run(
        _request(), _policy(), _criterion(), EvaluatorRef(name="judge"),
        lambda execution: [("correctness", 0.75, "stubbed")],
    ))
    assert len(result.executions) == 4
    assert len(result.evaluations) == 4
    assert result.outcome.selected_candidate_id is not None
    assert result.outcome.total_latency_ms is not None
    assert result.plan.rationale()
