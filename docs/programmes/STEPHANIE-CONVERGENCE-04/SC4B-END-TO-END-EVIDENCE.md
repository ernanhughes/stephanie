# SC4B — isolated end-to-end evidence

**Status: bounded cross-service API qualification and the current isolated Mission browser flow through live retained-experience display and post-result decision controls passed. The full negative-control matrix and concurrent operational dispatch qualification remain incomplete.**

Acceptance test: `backend/tests/test_improvement_integration.py::test_sc4_isolated_mission_to_stephanie_worker_praxis_and_retrieval`, marked `long` and run explicitly on 2026-11-25.

- Result: 1 passed in 61.23 seconds on the 2026-11-25 continuation; pytest elapsed time for the final run. The detail response was asserted to expose live retained CaseBook matches. A post-result accept decision was bound to source/candidate/protocol/result/proposal digests, repeated idempotently, and a conflicting reject was refused.
- Components: Mission's real FastAPI/Work Outcome services, Stephanie Python 3.13 CLI/SC3, an ephemeral in-process actual Worker HTTP server, the scripted deterministic Worker provider, and actual Praxis comparison code.
- Isolation: temporary Mission SQLite DB and per-run Stephanie artifact roots; isolated shared CaseBook/ExperimentStore SQLite under the configured run root; ephemeral Worker HTTP port and execution ledger; no operational service or storage writes.
- Governance: human-confirmed investigation proposal; experiment scope revision returned the same proposal to draft and required renewed human approval; active Work Envelope pinned the approved proposal; a missing-approval dispatch was refused; mismatched-investigation dispatch was refused.
- Results: Worker fixture and Praxis C1–C5 comparisons retained; C6 failure and C7 invalid candidate visible; zero model invocations; retained experience was retrieved by a subsequent Mission investigation.
- Recovery: repeated dispatch returned the existing Work Outcome link without a second execution. A simulated interrupted SC3 claim was reconciled from complete saved Worker results and the actual nested Worker ledger records; the Worker ledger record count did not increase. Work Outcome refreshed its content-addressed result reference.
- Recovery negative control: changing a saved response while retaining the real Worker ledger caused recovery to be rejected; restoring the saved artifact allowed reconciliation without another Worker request.
- Result interpretation: deterministic fixture integration only. Praxis decisions do not establish general or operational improvement; formal acceptance and production promotion remain false/not authorized.

The test does not demonstrate the complete negative-control set required by the commission (missing Stephanie/Worker services, stale or substituted candidate/protocol authorization, and all quota/unknown states). Its post-result human decision is API-tested, idempotent, and conflict-checked. Existing SC3 unit tests cover Praxis insufficient evidence, no-benefit, regression, additional-compute withholding, Worker failure, missing evaluator, invalid-candidate rejection, and candidate mutation; this is not a substitute for the complete SC4 negative matrix. Concurrent dispatch has not been qualified against the operational database.

## Browser qualification — isolated local runtime

On 2026-11-25, a second end-to-end run exercised the built Mission frontend against a separate temporary Mission process on `127.0.0.1:8791`. Its SQLite database, data paths, and Stephanie run root were under `C:\Users\ernan\AppData\Local\Temp\mission-sc4-browser-7bb6d6282b5e4c31842a826fc6d1301b`; no operational Mission database or service was used. The service was stopped after the run. The isolated database was seeded with one synthetic open Worker observation and matching objective; subsequent actions used the browser UI.

The UI sequence selected the observation, created and approved its Work Corrective investigation request, ran Stephanie's source-bound investigation, inspected the hypothesis/candidate/frozen experiment identities, requested a separate experiment decision, reapproved the version-pinned scope in Work Corrective, registered the matching Work Envelope, dispatched once, and displayed the resulting C1–C5 Praxis comparisons and resource observations. The run used the current Worker HTTP/service implementation with its deterministic scripted provider, 27 isolated Worker ledger records, actual Praxis comparison code, and zero model calls. Mission `HEAD` was `35b19c7c7ffc0b549f0816ca30c969b3ce3b96ff`, Stephanie `HEAD` was `a1605c3b08b3f34b79b098a1faa18e72c321faed`, and Worker `HEAD` was `a2e881c810c388fa2ce286a0b5169d4c93e791c5`; these are source checkout identities, not a clean release package, and the worktrees contain other concurrent changes.

That earlier run remains partial browser qualification: it predates the live CaseBook detail panel and post-result decision controls. The current source was requalified in the isolated browser flow below. No objective-completion or post-result human decision was recorded. The API-level and unit negative controls also remain incomplete as a single SC4 matrix.

The per-control evidence and remaining gaps are listed in `NEGATIVE-CONTROL-MATRIX.md`.

## Real CLI isolation test — 2026-11-25

The separately deferred test was run as an isolated qualification, not as another SC4B experiment:

```powershell
# Working directory: C:\Projects\mission\backend
..\.venv\Scripts\python.exe -m pytest tests/test_improvement_integration.py::test_mission_invokes_real_stephanie_cli_with_isolated_root_and_idempotency -q
```

It passed twice (20.18s, then 22.55s under process observation). The second run wrote artifacts under `C:\Users\ernan\AppData\Local\Temp\pytest-of-ernan\pytest-20278\test_mission_invokes_real_step0\stephanie-sc4-isolated`. The only SQLite file created by this CLI path was `memory\stephanie-sc3-isolated.sqlite3` (81,920 bytes) under that temporary root. Claims and each investigation's source/evidence/response files also remained under that root; the two investigation IDs had distinct roots. The second run's JUnit receipt (`exe_3fb838281d774c66`) records exactly one test case, passed in 22.308s (pytest wall time 22.55s).

The test ran from Mission's `.venv`; its adapter launched `C:\Projects\stephanie\venv\Scripts\python.exe -m stephanie.improvement.cli` with the temporary root and one synthetic finding JSON. The captured launcher PID was 46592. The process observer saw the pytest runner PIDs 33900, 47984 and 5664, and transient CLI command PIDs 46188/35256 for investigation `mission-64aac8ca1fbbd8363a7115f8` and 26228/34948 for `mission-adf6d9bcb7037d7852e4c626`. The AITest job-object wrapper exposed duplicate transient process observations; its JUnit receipt reports one executed test, but the watcher did not preserve parent PID mapping to disambiguate those duplicate CLI PIDs. This limitation is retained rather than inferring a one-to-one process mapping.

The test has no Mission `TestClient`, operational database connection, Worker service, or dispatch call. It proves the adapter's isolated per-investigation storage, deterministic source idempotency, distinct roots, and no Worker-results file at the run root. It does **not** prove preservation by before/after hash of an operational database or Worker ledger, nor does it qualify backup/restore. The child environment is built from the adapter's allowlist (`PATH`, Windows system/temp variables and `USERPROFILE`) and sets only the isolated Stephanie memory root plus Python path/encoding values. No production database, Worker ledger, or CLI process remained as an output of this test.

## Current browser qualification — 2026-11-25

The current Mission source was rebuilt and run on `127.0.0.1:8791` with a temporary SQLite database and an isolated Stephanie run root under `C:\Users\ernan\AppData\Local\Temp\mission-sc4-browser-isolated-3603321356074ee39618b33ba8d9f9a2`. `MC_DSN` and the legacy database URL were removed; startup reported `backend=sqlite`. No production service or database was used.

The browser showed proposal `wp:1ee64e8049444480` v2 approved, then `queued` after the already-authorized manual handoff. This exposed a lifecycle mismatch in the SC4 integration, which previously filtered for `approved` only. The UI/backend were corrected to preserve the handoff state as dispatchable while retaining scope-hash and active-envelope checks. The isolated acceptance test now exercises this approval → manual handoff → envelope → dispatch sequence.

The browser registered envelope `we:08f10a4a732848c5`, bound to objective `ob:sc4-browser`, finding `finding:sc4-browser`, and authorization `proposal:wp:1ee64e8049444480@v2:2a1a037c9a0fab25851910d3`. One dispatch completed for experiment `7cc37cb2e95dd58d098c3d80`. Mission's read-back reported zero model calls, C1–C5 comparisons, C6 provider failure recorded in the Worker ledger, C7 rejected before execution, four live retained CaseBook records, a `reported_complete` Work Outcome proposal, and no human accept/reject decision. The UI displayed live CaseBook evidence and post-result decision controls; those controls were left unused. No production promotion was authorized.

The isolated Mission process remains available on `:8791` for inspection. This is browser qualification of the bounded workflow, not SC4C activation evidence. Operational packaging, backup/rollback, full compatibility, independent activation review, and the complete negative-control matrix remain gates; activation stays OFF.

Do not use this qualification as activation evidence by itself. The normal Mission backend on `:8790` remains pre-SC4, and operational packaging, backup, rollback, compatibility, and independent activation review are still outstanding.
