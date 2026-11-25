# SC4 negative-control evidence

**Verdict: partial; activation gate not met.** Evidence below distinguishes deterministic source tests from live service observations. The SC4B result is immutable; this record does not rerun or accept it.

| Control | Evidence | Result |
|---|---|---|
| Stephanie missing or unconfigured | `test_cli_adapter_has_no_implicit_writable_run_root`; `StephanieCli.available` is false without an isolated run root. | Partial: no live missing-process test |
| Worker unavailable | No SC4 test with Worker stopped/unreachable. | Open |
| Missing authorization | SC4 isolated API integration rejects dispatch before approval with 409. | Passed in isolated fixture |
| Authorization for another candidate | `test_experiment_dispatch_checks_frozen_digests_and_claims_once` refuses changed candidate digest before invoking CLI. | Passed deterministically |
| Stale proposal version | The integrated fixture versions changed scope back to draft and requires reapproval; no direct stale-version dispatch race test. | Partial |
| Stale protocol digest | Same focused adapter test refuses a mismatched frozen protocol digest. | Passed deterministically |
| Revoked or invalid authority | Exact proposal/envelope binding is checked; no dedicated revocation transition matrix. | Partial |
| Repeated submission | Integrated test and browser evidence reconcile a sequential duplicate to the existing Work Outcome with one execution. | Passed sequentially |
| Concurrent dispatch | No multi-process or operational-database race qualification. | Open |
| Unknown execution state | `test_uncertain_investigation_claim_is_never_retried`; integrated run also refuses a tampered recovery artifact. | Passed for covered states |
| Interrupted dispatch | SC4B integration reconciles validated saved Worker results without an additional ledger record. | Passed in isolated fixture |
| Missing evaluation | Existing SC3 evaluator-absence fail-closed tests; no new SC4 live service test. | Passed at SC3 unit scope |
| Praxis insufficient evidence | C4 deterministic comparison in SC4B returns `insufficient_evidence`. | Passed for fixture |
| Candidate regression | C3 deterministic Praxis comparison rejects the regression. | Passed for fixture |
| Additional compute | C5 comparison records added invocations and withholds an unconfounded improvement recommendation. | Passed for fixture |
| Rejected candidate | C7 invalid candidate is rejected before Worker execution. | Passed for fixture |
| Experience retention failure | Successful retention and subsequent retrieval are exercised; forced retention-store failure is not. | Open |
| Human post-result decision | API tests cover result-bound accept, idempotent repeat, and contradictory decision refusal; browser controls were left untouched. | Partial; owner decision remains undecided |

The marked-long `test_sc4_isolated_mission_to_stephanie_worker_praxis_and_retrieval` passed in 61.23 seconds (recorded in `SC4B-END-TO-END-EVIDENCE.md`). The marked-long real CLI isolation test passed first in 20.18 seconds, then once more under process observation in 22.55 seconds because the first run had not emitted process IDs. No other long suites were run for this commission.

Routing tests are deterministic and do not make provider calls. They establish configured route choice, quota fallback, privacy/capability filtering, explicit pin behavior, cooldown, bounded attempts, and preservation of the supplied conversation message list. They do not establish live API compatibility, quota state, nor production concurrency.

**Activation stays OFF** until missing-service and concurrent dispatch controls, retention-failure behavior, independent review, package provenance, backup/restore, and compatibility gates are closed.
