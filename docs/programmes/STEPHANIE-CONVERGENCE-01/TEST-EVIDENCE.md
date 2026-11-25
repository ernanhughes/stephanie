# Test and verification evidence

## Checks performed for this readiness review

- Read-only Git state/HEAD comparison for Stephanie, AI, Worker, Praxis, Mission, Writer, Inspect, AITest and Workbench. No tests were run in external repositories.
- Read-only live GET checks: Mission `/api/health`, `/api/harness/catalogue`; Worker `/health`, `/v1/strategies`, `/v1/models/catalogue`. These were observations only; no POST, model call, service restart or database mutation.
- AITest SQLite opened read-only. Its Stephanie project row contains 129 tests from an older inventory; none of the new SE2Q/convergence test filenames was present. Recent AITest runs include incomplete/cancelled collection records and older test-specific timing, not the focused SE2Q run.

## Previous focused run timing review

The previous Stephanie Python 3.13 focused command (`test_worker_harness.py`, `test_evolution02.py`, `test_improvement_engine.py`, `test_stn06_preregistration.py`, `-m 'not long'`) reported 27 passed, 2 deselected, 1 warning, pytest duration 15.67 seconds; wall elapsed was about 20.3 seconds. It was invoked without `--durations`, so this run cannot attribute wall time to individual tests. Its aggregate duration exceeded the repository's approximate two-second long threshold; the individual test nodes have not therefore been labelled long without per-test evidence.

Local AITest files show a prior bounded Stephanie discovery run over 124 tests taking 11.67 seconds with zero tests executed (`TARGETED_PASS`); another canceled inventory took 18.14 seconds. This suggests collection/startup is a material component, but these are not timing evidence for the four focused modules and the registry is stale. No new suite was run to chase the timing. Python `pytest.ini` now keeps AITest plugin out of ordinary pytest by default because the prior auto-supervision caused collection to exceed its 15-second foreground budget. For the next targeted run, use one module at a time with `--durations=10`, record wall and pytest time, and only then mark any individual >2s test `long`. Do not remove tests to lower wall time.

## Tests used as compatibility evidence

- Existing Stephanie `tests/test_worker_harness.py` exercises a fake loopback HTTP server and asserts the emitted Work request fields and independent output/evidence validation. This is a fixture contract test, not proof of a successful operational Worker request.
- Existing `tests/test_improvement_engine.py` checks candidate immutability, evidence digest drift, fixture evaluator boundaries, experience retrieval and isolated SQLite CaseBook/ExperimentStore persistence. The persistence test is marked `long`; it was not rerun here.
- H6 Mission and Worker tests in their own repositories and their reports establish interface/source/test evidence, but their trees are dirty and no external tests were executed for this commission.
- No new integration tests were added because the only actual cross-repository tests would require executing mutable external adapters/services or duplicating unversioned DTOs. The synthetic and current live observations are clearly separated above.

## Deliberately not run

No full backend/frontend suite, long corpus, SE3, model experiment, cross-repository service test, AITest project activation, or operational database test was run. This is intentional; long suites remain deferred.
