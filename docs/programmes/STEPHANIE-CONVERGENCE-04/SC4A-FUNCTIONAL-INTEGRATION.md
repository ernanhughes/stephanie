# SC4A — functional integration

**Status: IMPLEMENTED FOR THE BOUNDED SC3 FIXTURE PROFILE; broader operational qualification remains gated.**

Mission now calls Stephanie through a fixed local CLI adapter. Each investigation receives a separate root for evidence, candidate, and frozen protocol. An explicitly configured SC4 run root owns the shared isolated CaseBook/ExperimentStore database so a later investigation can retrieve retained SC3 experience without sharing the operational SIS store.

## Implemented path

- Mission builds a bounded source snapshot from its authoritative episode and open finding records. Stephanie receives the snapshot only after a Work Corrective proposal covering that episode/finding is approved.
- The general path returns a source-bound deterministic hypothesis and request for additional evidence. Only the exact `worker_json_fence_normalization` finding that cites the current Worker source digest can produce the existing frozen SC3 fixture candidate/protocol. Investigation does not execute Worker.
- A separate experiment decision revises the same approved Work Corrective proposal, bumps its version, and returns it to draft. The renewed human approval pins canonical scope containing the source, candidate, experiment, protocol, resource, and Work Objective identities/digests. No overlapping approval store is added.
- Dispatch requires that exact approved scope and an active Work Envelope bound to the proposal ID/version/scope hash and same objective/finding. Stephanie validates the on-disk candidate/protocol and calls only the existing `sc3-run` command.
- Mission atomically reserves the approved proposal under a row lock, advances its existing Work Corrective lifecycle to `queued`/`running`, and creates a Work Outcome execution link before invoking Stephanie. Scope edits and cancellation are blocked while that claim is running.
- A durable exclusive Mission claim and Stephanie's SC3 claim prevent duplicate dispatch. `sc3-reconcile` accepts an interrupted run only when frozen requests match exactly, saved results are complete and ordered, and each result embeds the exact unique record present in the isolated Worker ledger. Recovery verifies the Worker ledger digest, normalized request input digest, requested execution summary, resolved strategy identity, and public output digest. Normal result retrieval also checks Stephanie's SC3 claim, integrated-result hash, Worker ledger hash, and Mission's outer claim. Timeout/unknown state is never retried.
- Mission updates the existing Work Outcome execution link and lifecycle to `reported_complete` through a dedicated result linker. Execution completion remains separate from objective completion, acceptance, and promotion.
- The investigation detail API retrieves bounded matching retained experience through Stephanie's existing `sc3-retrieve` command under the isolated run root. The Improvement page displays it separately from the historical SC3 artifact snapshot and explicitly shows retrieval unavailability.
- After a verified result, Mission can append a distinct accept/reject decision event to the existing Work Corrective proposal event history. The event includes exact investigation/source/candidate/protocol/result and authorization proposal digests, a human actor and optional rationale. It does not change the proposal lifecycle status, complete the objective, or authorize production promotion. Repeating the same decision is idempotent; a contradictory second decision is rejected.
- The Improvement page exposes investigation, exact authorization, envelope registration and dispatch controls, saved Praxis/resource results, live retained experience, and post-result accept/reject controls. Historical SC3 remains labeled as an isolated artifact snapshot.

## Boundaries

Only the existing deterministic Worker JSON-fence fixture profile is executable through this path. It uses an ephemeral actual Worker HTTP service and scripted provider, with zero model calls. Generic observations remain investigation-only. The post-result decision API and UI controls passed isolated browser qualification; no post-result decision was recorded. This work does not provide general Stephanie conversational integration, free-first model routing, production promotion, or a live SIS service endpoint. Row-lock guarantees depend on database locking semantics; concurrent duplicate requests have not been separately qualified.

The ordinary Mission service on `127.0.0.1:8790` still predates SC4. Do not activate the feature until SC4B/SC4C gates pass.
