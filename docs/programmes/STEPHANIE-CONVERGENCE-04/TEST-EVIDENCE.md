# SC4 test evidence

Focused suites rerun for the approved-handoff correction on 2026-11-25:

- Mission backend `backend/tests/test_improvement_api.py` and `backend/tests/test_work_envelope.py`: **12 passed**. Native pytest mode; two existing warnings (pytest `CallSpec2` deprecation and unrecognized configured `timeout` option).
- Frontend `npm exec vitest run src/test/ImprovementPage.test.tsx`: **4 passed**.
- Frontend `npm run build`: passed; Vite emitted the existing advisory for bundles over 500 kB.
- `git diff --check`: passed; only line-ending normalization warnings were emitted for edited Mission files.

Explicit SC4B acceptance run:

- `backend/tests/test_improvement_integration.py::test_sc4_isolated_mission_to_stephanie_worker_praxis_and_retrieval`, marked `@pytest.mark.long`: **1 passed in 61.07 s**.
- This run used Mission Work Outcome services and a temporary SQLite DB, Stephanie Python 3.13, the ephemeral actual Worker HTTP service with deterministic scripted responses, and Praxis. It exercised approved scope followed by manual handoff, matching Work Envelope registration, dispatch/result/retrieval, zero model calls, source isolation, repeated dispatch reconciliation, simulated interrupted SC3 completion-claim recovery without additional Worker requests, ledger verification, Work Outcome linking, and the post-result decision API. The long run was required because this change directly targeted the SC4B acceptance gate.

Browser qualification:

- The current Mission frontend was exercised against an isolated temporary service at `127.0.0.1:8791`, with its own SQLite DB and Stephanie run root under the system temporary directory. The browser verified approval, manual handoff, envelope registration, one dispatch, live CaseBook records, C1–C5 comparisons, and post-result decision controls. Read-only API reconciliation also verified C6/C7, zero model calls, four retained records, and a `reported_complete` Work Outcome link. The post-result decision controls were left unused. See `SC4B-END-TO-END-EVIDENCE.md` for identifiers and exact scope.
- The isolated service remains available for inspection. The standard service on `:8790` was not restarted or modified.

The related `test_mission_invokes_real_stephanie_cli_with_isolated_root_and_idempotency` test in `test_improvement_integration.py` is also marked `long`; it was deliberately deferred in this loop. Broad Mission/Stephanie suites, the complete negative-control matrix, concurrent operational dispatch, and deployed E2E were not run. SC4 operational activation remains off.
