# STN05 — Verified Experience Design

## Scope

STN05 implements a deterministic, human-gated lifecycle around existing ChatStore, PlanTrace, CaseBookStore, MemCubeStore, ServiceContainer, Hydra, and SIS routes. It adds no table, embedding index, model call, training job, or execution backend.

The real STN04 execution review is now explicitly ACCEPT for the scoped `calculator.py` change. Its separate lesson-candidate review remains pending; no Worker acceptance ID exists and no MemCube is active.

## Trust states

### Execution review

```text
PENDING ── human ACCEPT ──> ACCEPTED_FOR_EXPERIENCE_REVIEW
   │                             │
   ├── human REJECT ─────────> REJECTED
   └── NEEDS_REVISION ───────> NEEDS_REVISION (review may be repeated)
```

`ACCEPTED_FOR_EXPERIENCE_REVIEW` is stored in SIS review metadata only. It does not change `Worker.status`, create a Worker `acceptance_id`, or claim general correctness. ACCEPT is rejected unless the persisted SIS execution is `COMPLETED_UNVERIFIED`, the independent verifier reports `passed` with exit `0`, evidence references match the execution/trace/verifier, and the accepted file scope is a subset of Worker-reported artifacts.

Human review entries include reviewer, decision, timestamp, rationale, evidence references, accepted scope, and an idempotency key. The ChatStore update is row-locked and appends an audit history. A final ACCEPT/REJECT cannot be changed by reusing a different decision; `NEEDS_REVISION` can be reviewed again.

### Experience review and promotion

```text
accepted execution
        │ deterministic extraction
        ▼
CaseBook candidate: PENDING_REVIEW, reuse_eligible=false
        │ separate human lesson review
        ├── REJECT / NEEDS_REVISION ──> remains inactive
        └── APPROVE ──> insert approved MemCube ──> ACTIVE
```

Candidate review is separate from accepting the underlying execution. Only a second explicit APPROVE creates a MemCube row with source `stephanie.stn05.approved-execution-experience`. Retrieval requires that source plus `execution_review_decision=ACCEPT`, `candidate_review_decision=APPROVE`, `reuse_eligible=true`, and `promotion_status=ACTIVE`.

Withdraw, supersede, or stale actions preserve the historical record but set the MemCube's `reuse_eligible` false. A superseding experience must already be approved and active in the same conversation scope.

## Experience candidate v1

The deterministic candidate is stored under the existing `CaseBook.meta["stn05_experience"]`, with a related Case and `CaseScorable` role `experience_candidate`. The candidate identity is stable for the source execution; a canonical candidate hash prevents silently replacing content under the same identity.

```json
{
  "schema": "stephanie.execution-experience.v1",
  "experience_id": "exp_<stable-id>",
  "candidate_hash": "<canonical-sha256>",
  "task": {"objective": "...", "category": "coding"},
  "observations": [
    {"kind": "worker_report", "claim": "Worker reported the listed artifact change."},
    {"kind": "independent_verification", "claim": "The configured independent test command passed."}
  ],
  "reported_agent_summary": "...",
  "artifacts": [{"path": "calculator.py", "kind": "modified_file"}],
  "verification": {"command": "python3 -m unittest", "status": "passed", "exit_code": "0"},
  "lesson_hypothesis": {
    "text": "...",
    "classification": "hypothesis",
    "validation_status": "untested_for_transfer",
    "authoring": "deterministic_template",
    "applicability": ["small localized coding defects"],
    "limitations": ["derived from one small task", "transfer has not been measured"]
  },
  "source": {
    "conversation_id": 0,
    "originating_message_id": 0,
    "execution_id": "sex_...",
    "worker_request_id": "wrq_...",
    "plan_trace_id": "worker:sex_..."
  },
  "execution_review_status": "ACCEPT",
  "candidate_review_status": "PENDING_REVIEW",
  "reuse_eligible": false,
  "promotion_status": "NOT_PROMOTED",
  "model_assisted_extraction": false
}
```

Facts are structured records from SIS/Worker/PlanTrace. The agent's summary is retained separately as a report, not elevated to independent evidence. A reusable lesson is explicitly a hypothesis. No confidence score is invented.

## Retrieval and optional reuse

- Retrieval uses the existing `MemCubeStore.search` with exact source, type, and sensitivity filters, then a deterministic lexical-overlap ranking. It does not call an embedding model or LLM.
- Only approved active MemCube rows are returned. An empty result is normal.
- Results include experience ID, MemCube ID, relevance score, source PlanTrace, and a bounded lesson/limitations view.
- Context is bounded by Hydra config and clearly labeled as untrusted reference material, not an instruction.
- Preview is read-only and never invokes a model or automatically changes a task.
- A future SIS task may explicitly select approved experience IDs. The server re-resolves them through the approved-only retriever; client-supplied context text is not accepted. When selected, the bounded context is appended to the Worker objective and recorded separately from the original user objective, with IDs/scores in ChatStore and PlanTrace. No Worker schema or runtime change is required.
- The retrieval preview and task-context composition are tested with synthetic approved MemCube fixtures. No real STN04 experience is eligible until the operator explicitly accepts STN04 and separately approves its lesson.

## Hydra and existing service architecture

`experience_learning` is a Hydra node. Checked-in `config/sis.yaml` disables it and retrieval. The local `sis_stn04_local` profile enables the explicit review interface for this controlled test. `ExecutionExperienceService` is registered through the existing ServiceContainer; it performs no startup extraction and no model call. It consumes the already-registered `memory.casebooks` and `memory.memcubes` stores.

## Deliberate non-goals

- No real STN04 acceptance is written by code.
- No candidate is extracted or promoted from STN04 until a human explicitly accepts the execution.
- No experience is included in a Worker task without explicit SIS selection.
- No prompt, model, Worker authentication policy, embedding index, or model weights are changed.
- No claim of improved performance is made. Transfer is the subject of STN06.
