"""ELM validation tests (pytest-compatible).

Replaces the legacy top-level validation script which aborted full pytest
collection via ``sys.exit(1)`` on import failure. All checks are now real
test functions with no module-level side effects.
"""

from stephanie.components.elm import (
    ContextPack,
    CalibratedThresholds,
    RetentionTracker,
    CollapseDetector,
    GovernanceSignalExtractor,
    DominanceChecker,
    RegimeController,
    PerturbationInjector,
    SystemInterface,
    MockSystem,
)


def _make_thresholds() -> CalibratedThresholds:
    return CalibratedThresholds(
        energy_max=0.5,
        energy_warning=0.4,
        hrm_min=0.6,
        margin_min=0.5,
        variance_min=0.3,
        collapse_index_max=10.0,
        drift_max=0.15,
    )


def test_elm_imports():
    """All core ELM symbols are importable."""
    assert ContextPack is not None
    assert CalibratedThresholds is not None
    assert RetentionTracker is not None
    assert CollapseDetector is not None
    assert GovernanceSignalExtractor is not None
    assert DominanceChecker is not None
    assert RegimeController is not None
    assert PerturbationInjector is not None
    assert SystemInterface is not None
    assert MockSystem is not None


def test_core_objects_created():
    thresholds = _make_thresholds()
    tracker = RetentionTracker()
    detector = CollapseDetector(thresholds)

    assert thresholds.energy_max == 0.5
    assert tracker is not None
    assert detector is not None


def test_mock_system_satisfies_interface():
    mock = MockSystem()
    assert isinstance(mock, SystemInterface)


def test_experiment_harness_created():
    from stephanie.components.elm.experiment.experiment import (
        ScoreBundleExperiment,
    )

    thresholds = _make_thresholds()
    mock = MockSystem()
    experiment = ScoreBundleExperiment(
        system=mock,
        queries=["test query 1", "test query 2"],
        thresholds=thresholds,
        extractor=GovernanceSignalExtractor(),
    )
    assert experiment is not None
    assert experiment.system is mock


def test_governance_dominance_roundtrip():
    """Signal extractor + dominance checker agree on a simple improvement."""
    from stephanie.data.score_bundle import ScoreBundle
    from stephanie.data.score_result import ScoreResult

    def _bundle(alignment: float, energy: float, margin: float) -> ScoreBundle:
        return ScoreBundle(
            results={
                "alignment": ScoreResult(
                    dimension="alignment",
                    score=alignment,
                    source="test",
                    rationale="test",
                    weight=1.0,
                    attributes={},
                ),
                "energy": ScoreResult(
                    dimension="energy",
                    score=energy,
                    source="test",
                    rationale="test",
                    weight=1.0,
                    attributes={"raw_energy": energy / 100.0},
                ),
                "margin": ScoreResult(
                    dimension="margin",
                    score=margin,
                    source="test",
                    rationale="test",
                    weight=1.0,
                    attributes={},
                ),
            }
        )

    before = _bundle(alignment=70.0, energy=60.0, margin=60.0)
    # Higher alignment/margin is better; lower energy score is better.
    after = _bundle(alignment=80.0, energy=50.0, margin=70.0)

    extractor = GovernanceSignalExtractor()
    assert extractor.dominates(before, after) is True
    assert extractor.compute_dominance(before, after) is True

    checker = DominanceChecker()
    result = checker.check(before, after)
    assert result.is_dominant is True

    metrics = extractor.extract_metrics(after)
    assert metrics["energy_raw"] == 0.5
