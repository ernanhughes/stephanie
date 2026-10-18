# STN05 — Implementation Results

## Summary

The STN05 review, deterministic candidate, CaseBook/MemCube lifecycle, and optional retrieval path are implemented behind a Hydra feature gate. The operator explicitly accepted STN04 for the `calculator.py` scope. One deterministic CaseBook candidate is now persisted with a separate lesson review pending. No STN04 MemCube was promoted, and no model inference or A/B task was run for STN05.

## Reused components and changes

- `stephanie.memory.ChatStore`: atomically records SIS human-review history in the existing execution metadata, with idempotency and no change to Worker status/acceptance ID.
- `stephanie.memory.CaseBookStore`: atomically upserts a provenance-linked candidate CaseBook/Case/Scorable, records the second review, promotion state, and withdrawal/supersession/stale actions.
- `stephanie.memory.MemCubeStore`: approved candidates are inserted with deterministic identity, internal sensitivity, approval provenance, and active eligibility. Retrieval filters by the exact approved STN05 source and status flags.
- `stephanie.services.ServiceContainer`: registers `ExecutionExperienceService`; it uses the existing `MemoryTool` stores and Hydra config.
- SIS existing auth/session/CSRF/ownership middleware protects review, extraction, candidate review, deactivation, and retrieval endpoints.
- SIS execution form supports an optional approved-experience preview and explicit experience IDs. If selected, SIS re-resolves them server-side, adds a bounded, labeled untrusted-context section to the Worker objective, and records the selected IDs/scores/context separately from the original objective and in PlanTrace metadata. No Worker source or WorkRequest schema changed.
- Checked-in `config/sis.yaml` keeps experience and retrieval disabled. Local ignored `config/sis_stn04_local.yaml` enables explicit review and read-only retrieval preview; real review and candidate decisions remain visible as separate states.

No model-assisted extraction is implemented or invoked. Deterministic extraction separates observed Worker/verifier facts from the lesson hypothesis. A separate human approval is required before MemCube promotion.

## STN04 review state

- The authenticated SIS review record is `ACCEPT`, reviewer `sis`, idempotency/review ID `stn05-human-review-stn04-accept-001`, scope `calculator.py`. Worker status remains `AWAITING_ACCEPTANCE` and no Worker acceptance ID was created.
- The actual SIS evidence page shows the recorded decision, passing verifier output, and a clear warning that no raw before/after diff artifact was persisted.
- A reconstructed one-line comparison is in `stn05-learning-inventory.md`, explicitly labeled as reconstructed from the recorded fixture baseline, not the original patch.
- The authenticated review controls require evidence references and a file scope for ACCEPT. Live readback records review `stn05-human-review-stn04-accept-001`, reviewer `sis`, and scope `calculator.py`. This does not create a Worker acceptance ID.
- Before the human decision, a no-review candidate-extraction request returned `HUMAN_ACCEPTANCE_REQUIRED` (HTTP 409), with no candidate write. After the explicit SIS ACCEPT, the deterministic extraction endpoint created the candidate. Approved-experience retrieval still returns zero because the lesson itself remains pending review.
- STN04 remains Worker `AWAITING_ACCEPTANCE` / SIS `COMPLETED_UNVERIFIED`. Candidate `exp_33fd63f133b44b21275681e5af0aabb7` is in CaseBook `1744`, Case `21680`, Goal `2986`, Scorable role `experience_candidate`; status `PENDING_REVIEW`, `reuse_eligible=false`, and `model_assisted_extraction=false`. Fresh normal MemoryTool readback verified the CaseBook, Scorable, Goal, ChatStore execution and two-step PlanTrace links. No MemCube is active and no model-assisted lesson was generated.

## Learning-system audit

- A fresh normal `MemoryTool` instance successfully queried CaseBook, MemCube, Experiment, Metric, ChatStore, and PlanTrace stores.
- The minimal ORM-only `TrainingStatsORM` mapper error is caused by incomplete model registration when only a subset of ORM modules is imported. Normal `MemoryTool` registers `TrainingStatsStore`/`TrainingStatsORM` and the queried services work. No schema or ORM mapping change was needed for STN05.
- STN05 CaseBook/MemCube create, review, retrieval, and withdrawal were tested against an isolated SQLite database. The accepted STN04 candidate was read back through a fresh normal `MemoryTool` process; it links to the original message/execution/PlanTrace. The approved-only retrieval preview returned zero real experiences because the candidate lesson is still pending review. PostgreSQL-only CaseBook tag type was adapted only inside the test fixture and restored afterward.
- The legacy `MemCubeClient` and broad `ChatToCaseBookAgent`/CBR paths are not used: the former has an unset store/legacy method mismatch, and the latter lacks the STN05 human-acceptance gate. The dedicated flow uses existing stores directly and keeps pending candidates out of MemCube retrieval.
- `ExperimentStore.assign_variant` used process-randomized Python `hash()`. STN05 repairs this narrow defect with a stable SHA-256 assignment key and cross-`PYTHONHASHSEED` regression test. No live experiment/trial was created.

## Verification

- `venv\Scripts\python.exe -m pytest tests/test_stn05_experience.py tests/test_sis_worker_execution.py tests/test_sis_chat_execution.py tests/test_sis_security.py -q` — **68 passed**, one existing Starlette deprecation warning.
- `venv\Scripts\python.exe -m py_compile sis/routes/chats.py sis/main.py sis/services/worker_execution.py stephanie/services/execution_experience.py stephanie/memory/chat_store.py stephanie/memory/casebook_store.py stephanie/memory/experiment_store.py tests/test_stn05_experience.py tests/test_sis_worker_execution.py` — passed.
- `git diff --check` — passed; only existing LF/CRLF warnings were emitted.
- No paid model calls were made; no experimental trials were written to ExperimentStore/MetricStore.

## Remaining gate

The STN04 execution review is recorded as ACCEPT for the limited artifact scope. The **experience candidate itself still requires a separate human APPROVE/REJECT/NEEDS_REVISION decision**. Until it is approved, it remains retrieval-ineligible; no MemCube promotion or improvement claim exists.
