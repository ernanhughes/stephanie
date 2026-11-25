# Worker execution issue — Stephanie WorkResult observability

**Status:** observed in SE2Q; causes unresolved. **Owner:** Worker. This record is suitable for a future Worker engineering commission; it does not prescribe an implementation or claim current defects have not since changed.

## Archived observations

Use `STEPHANIE-EVOLUTION-02Q/DIAGNOSTIC-RESULTS.md`, `data/stephanie_evolution02q/readiness-results.json`, `worker-1024-results.json`, `worker-executions.jsonl`, `worker-executions-budget.jsonl`, and the run-local H3 ledgers as source. Preserve their manifests/hashes and the documented follow-up hash gap.

| Outcome | Count | Evidence and limit |
|---|---:|---|
| Worker result body structurally/citation usable | 1/5 Work requests | One 1,024-token result valid, 533 output tokens. |
| Malformed/incomplete body | 1 | 512-token task body ended mid-JSON at exactly the requested output ceiling. Supports budget interaction on that one task; provider `finish_reason` was not retained. |
| Dispatched request with empty response | 2 | Work request dispatched; body absent; inference execution, timeout/cancel, extraction and usage are UNKNOWN. |
| Blocked before provider dispatch | 1 | Worker route temporarily unavailable; no provider inference is claimed. |
| Corrective invocation observed | 0 | No correction call occurred in retained results. |

Direct AI is a separate lane (3/3 structured/citation-valid, semantic rubric 1/3, plus one passing exact API probe); do not combine it with Worker reliability counts.

## Questions for Worker engineering

For each empty-output case, distinguish request admission, executor dispatch, provider invocation, provider response, response extraction, persistence, timeout/cancellation, and caller transport. Preserve request/execution/invocation identities at each boundary; retain raw provider response when policy permits; record provider termination/usage or UNKNOWN; make empty response distinct from pre-dispatch block and from error response. Verify ledger/artifact durability after restart. Do not infer dispatch from HTTP acceptance or `AWAITING_ACCEPTANCE` alone.

The 1,024-token outcome shows that one task can fit at 533 output tokens. Increasing the cap cannot explain or resolve two empty outputs without observing their lifecycle. Do not add retries or silently change executor/strategy; bounded correction is for an observed structural mismatch and is not an unknown-outcome retry.

## Current status

Worker's current local service reports version 0.2.0 and has `structured.v1`, `corrective.v1`, and `adaptive.v1`; adaptive allows up to two model invocations and reports requested time/token controls while provider enforcement/termination can remain unknown. Its `opencode` executor was unavailable in the live health response, while `opencode-work` and the model executor reported available. No model call was made to retest the issue.

Current dirty Worker source at HEAD `a2e881c` has made a material observability change: model runtime classifies empty output as `empty_output`, records a failed attempt with `request_dispatched=true`, `request_admitted=UNKNOWN`, `inference_executed=UNKNOWN`, and the Worker execution ledger can archive terminal Work/H3 summary evidence (including pre-dispatch executor-health failure). This can distinguish a returned-empty adapter outcome from a known pre-dispatch block more clearly. It still cannot establish that inference executed, and the provider adapter's decoded body/response extraction can be lost when empty text is raised as a failure. H3 summaries retain digests/status rather than arbitrary raw response bodies; raw/normalized output belongs to WorkResult/artifact retention. The two historical empty attempts cannot be retroactively reclassified, and current dirty source/live health are not a verified fix or reliability qualification.

Requested future evidence: a fresh bounded test under a separately authorized Worker commission, per-stage lifecycle events for one synthetic task, preserved raw/empty outcomes and usage/termination metadata, restart durability, and independent review. Do not modify Worker from Stephanie.
