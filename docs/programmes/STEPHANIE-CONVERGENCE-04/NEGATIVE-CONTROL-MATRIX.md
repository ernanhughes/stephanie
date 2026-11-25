# SC4 negative-control qualification matrix

This record maps the commission's controls to focused tests and identifies what remains unqualified. Passing a deterministic fixture control does not establish general operational effectiveness.

| Required control | Evidence | Status |
|---|---|---|
| Stephanie service missing | Adapter fails closed when CLI is unavailable; no dedicated missing-executable endpoint test | Partial |
| Worker service missing | No isolated E2E with Worker unavailable | Open |
| Invalid candidate / rejected candidate | SC3 C7 rejects an unallowlisted strategy before Worker request; asserted in the integrated run | Passed for fixture |
| Missing authorization | Isolated Mission E2E dispatch before renewed approval returns 409 | Passed |
| Authorization for a different candidate | CLI adapter rejects candidate digest mismatch; full API flow with altered approved candidate not separately exercised | Partial |
| Stale protocol digest | CLI adapter rejects stale digest before dispatch | Passed |
| Duplicate dispatch | Repeated API dispatch reconciles to the existing Work Outcome link without another Worker request | Passed sequentially |
| Unknown execution state | Tampered Worker ledger blocks result retrieval; incomplete run claim remains non-retryable and is reconciled only from validated records | Passed for covered recovery cases |
| Worker execution failure | C6 fixture records an actual Worker failure and preserves failure/resource evidence | Passed for fixture |
| Missing evaluator | Frozen Praxis evaluator absence fails closed in SC3 tests | Passed |
| Praxis insufficient evidence | C4 fixture returns `insufficient_evidence` | Passed for fixture |
| Candidate regression | C3 fixture returns a rejection for regression | Passed for fixture |
| Additional-compute confound | C5 records extra invocations and withholds an unconfounded positive recommendation | Passed for fixture |
| Recovery from tampered saved response | Recovery rejects a changed Worker response when the authoritative ledger is unchanged | Passed |
| Provider quota, privacy, and routing controls | SC4 uses no genuine model inference and does not modify free-first routing; routing-specific deterministic tests were not implemented here | Open / not exercised |
| Concurrent duplicate requests | Tests use sequential requests and isolated SQLite; operational database locking/concurrency is unqualified | Open |
| Post-result human acceptance or rejection | API accepts an exact result-bound human decision; idempotence and conflicting-decision rejection are tested. Current isolated browser displays both decision controls, which remain unused pending a separate human decision | Partial |

SC4B browser workflow passes for the bounded fixture path, but this matrix remains partial because missing-service, candidate-substitution, quota/privacy/routing, and concurrent-dispatch controls are not fully exercised. Do not use this matrix as activation approval. SC4C remains OFF until the remaining negative controls, clean package provenance, backup/rollback, compatibility, and activation review gates pass.
