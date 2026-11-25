# Test Evidence

Focused verification:

- `pytest -q tests/test_sc3_integration.py tests/test_sc2_integrations.py --durations=10` — 11 passed in 1.37s; slowest test 0.53s. This final combined run includes the once-only claim test, SC3 controls, and SC2 AI/read-adapter tests.
- Shared AI focused selection — 10 passed in 1.1s (SC2 final verification; no AI changes during SC3).
- Python compile check and `git diff --check` passed.

No long suite was run. The previously identified SQLite persistence test remains marked long and was deliberately deferred; broad suites were not run. A pytest AnyIO deprecation warning was observed in prior focused runs.
