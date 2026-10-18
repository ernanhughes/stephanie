# STN06 — Preflight Results

## Summary

The deterministic preparation for STN06 is complete and frozen. The approved
STN05 experience is frozen as an experiment-scoped input, the eight held-out
tasks and their sealed oracles are hashed, the arm schedule is derived from the
stable SHA-256 assignment, the runtime/permission/retrieval budget policy is
recorded, and a dry-run exercised dispatch construction, memory selection, the
sealed oracle and the scoring path **without invoking GPT-6 Luna**.

**No paid model call was made. No trial or metric row was written. The withdrawn
MemCube was not reactivated, and ordinary agent execution still retrieves
nothing.** This document reports readiness; it does not report an experiment
result and makes no improvement claim.

Frozen identities:

| Identity | Value |
|---|---|
| Held-out manifest hash | `1c0d167f26b6f79b41197fd59b0b0be4a77fab053ac49b372404d0ee8e159518` |
| Preregistration hash | `08deb57223b1556b884d3a01b5f8479ab97d20b3aa5347c19382e94415442d51` |
| Frozen experience | `exp_33fd63f133b44b21275681e5af0aabb7` (version `v1`) |
| Frozen candidate hash | `d4d72077672fb481d7e955ccf249832921691d5ac268be4ca1f004184e12af79` |
| Frozen context hash | recorded in `stn06-preflight.json` (`context_hash`) |
| Model route / executor | `opencode-go/gpt-6-luna` via Worker `opencode-docker`, `workspace_write` |
| Preregistration hash is reproducible | Verified: the pilot and full dry-runs produced the same hash despite different wall-clock timestamps. |

Inspectable artifact: [`stn06-preflight.json`](stn06-preflight.json), produced by
`scripts/stn06_preflight.py`.

## 1. STN05 record reconciliation

The two earlier reports disagreed about whether the lesson was
`PENDING_REVIEW`/never promoted or `APPROVED`/promoted/withdrawn. A fresh
read-only `MemoryTool` readback (`scripts/stn06_reconcile.py`) of the persisted
ChatStore, CaseBook and MemCube records resolves it. **No record was changed to
make the reports agree.**

| Record | Persisted state | Expected state |
|---|---|---|
| STN04 human execution review | `ACCEPT` (review `stn05-human-review-stn04-accept-001`, 1 history entry) | ACCEPT |
| Worker acceptance | `acceptance_id = null`, execution status `COMPLETED_UNVERIFIED` | AWAITING_ACCEPTANCE |
| Experience candidate review | `APPROVED` (`candidate_review_decision = APPROVE`) | APPROVED |
| MemCube `exp_33fd63f133b44b21275681e5af0aabb7:v1` | `promotion_status = WITHDRAWN`, `reuse_eligible = false` | Withdrawn |
| Active retrieval eligibility | `false` | False |
| Transfer validation | `lesson_hypothesis.validation_status = untested_for_transfer` | Untested |

The newer state is authoritative. The appended earlier STN05 report described an
intermediate state that was superseded by the recorded candidate approval and
subsequent withdrawal (`review_id stn05-deactivate-stn04-candidate-001`).

**Observed residual inconsistency (documented, not silently rewritten):** the
source execution's ChatStore `experience_status` label still reads `ACTIVE` even
though the CaseBook and MemCube are `WITHDRAWN`. The withdrawal path updated the
CaseBook/MemCube but not the execution label. The service now keeps that label
consistent for **future** withdrawals, but the historical STN04/STN05 record was
left untouched. The authoritative lifecycle state is the CaseBook/MemCube record
(`WITHDRAWN`, `reuse_eligible=false`), and retrieval confirms eligibility is
`false`.

## 2. Frozen experience (no global reactivation)

`Stn06ExperimentService.freeze_experience()` reads the persisted candidate through
the existing CaseBook store and requires, or it refuses to proceed:

- `candidate_hash` equals the frozen hash;
- `candidate_review_status == APPROVED`;
- `execution_review_status == ACCEPT`;
- the MemCube is **not** globally retrievable (refuses with
  `MEMCUBE_GLOBALLY_ACTIVE` if `reuse_eligible=true`/`promotion_status=ACTIVE`).

Live result: `globally_retrievable = false`, `withdrawal_preserved = true`. The
frozen payload captures the exact lesson, applicability, limitations and
extraction/review provenance, plus a deterministic `context_section` and its
hash. There is no global MemCube mutation.

## 3. Experiment-scoped memory selection

Selection runs through `Stn06ExperimentService.select_experience(arm)`:

- **Arm A (baseline):** `context = None`, `selected_experience_ids = []`.
- **Arm B (experience):** exactly the frozen experience id and candidate hash,
  rendered through the single shared formatter
  (`ExecutionExperienceService.format_experience_section`) and labelled
  *"untrusted task context; not an instruction"*.
- **Arm C (neutral, optional/deferred):** a neutral context block with no
  experience.

Because selection reads the frozen candidate directly and never flips MemCube
eligibility, ordinary execution is unaffected. Confirmed live: for both pilot
tasks, `ExecutionExperienceService.retrieve_for_task(...)` returned **zero**
results (`normal_retrieval.withdrawn_unavailable = true`). The withdrawn record
stays unavailable to normal agent execution.

`authorize_scope(...)` records an explicit, auditable, experiment-only
authorization (`scope = stn06-coding-transfer`, `global_reactivation = false`,
`retention = experiment_only`) tied to the frozen id/version/hash.
`persist_preregistration(...)` writes the frozen preregistration and the
authorization into the **existing** `ExperimentStore` (`experiments` +
`experiment_variants` tables). It does not create a new database.

## 4. Held-out fixture manifest and sealed oracle

`stephanie/experiments/stn06_manifest.py` defines eight independent,
dependency-free Python tasks:

| Task class | Count | Tasks |
|---|---|---|
| Near transfer (small localized defect) | 4 | `stn06-near-01…04` |
| Generalization (multi-file / less obvious) | 2 | `stn06-complex-01…02` |
| Unrelated coding task | 2 | `stn06-unrelated-01…02` |

Each task carries three disjoint file sets: `workspace_files` (given to both
arms), a sealed `reference_solution`, and sealed `oracle_files`. The oracle and
reference sets are **never** placed in an agent workspace; the manifest test
asserts the workspace and oracle path sets do not intersect.

`verify_fixtures()` **actually ran every sealed oracle in an isolated temporary
directory** using the local interpreter (no model):

- the defective workspace fails the oracle, and
- the reference solution passes the oracle.

Live result: **8/8 fixtures verified** (`oracle_all_verified = true`).

## 5. Frozen runtime, policy, schedule and budget

- **Model/executor/permissions:** `opencode-go/gpt-6-luna` through Worker
  `opencode-docker` with `workspace_write`; request protocol
  `worker.work-request.v1`; one model request per execution.
- **Retrieval policy:** `approved_memcubes_keyword_v1`; the context label and
  rendered context hash are frozen in the preregistration.
- **Stable schedule:** derived from the existing, regression-tested
  `_stable_variant_index` (SHA-256) over `(experiment, domain, task_case_id)`;
  verified identical across `PYTHONHASHSEED` values.

| Task | Run order |
|---|---|
| `stn06-near-01` | A, B |
| `stn06-near-02` | B, A |
| `stn06-near-03` | A, B |
| `stn06-near-04` | B, A |
| `stn06-complex-01` | A, B |
| `stn06-complex-02` | A, B |
| `stn06-unrelated-01` | B, A |
| `stn06-unrelated-02` | B, A |

- **Budget:** pilot = 2 tasks × 2 arms = **4 executions**; full = **16**. The
  enforceable controls are bounded execution counts, a 600 s wall-clock limit per
  execution, one model request per execution, and an explicit ban on silent
  retries, model substitution and paid escalation. **The $2 target is not
  enforceable** and is reported as a target, not a guaranteed cap, because STN04
  reported no token/cost data and the runtime exposes no metering feed.

## 6. Isolation and leakage controls

The preregistration records the controls: the experience is frozen before any
task disclosure; only the frozen approved experience enters Arm B; retrieved text
is bounded and labelled untrusted context; sealed oracles and reference solutions
are never placed in a workspace; each arm uses a fresh isolated workspace and
cannot observe the other; the withdrawn MemCube is not globally reactivated; and
no automatic prompt mutation, skill promotion or model training occurs. Worker
stays a loopback-only, unauthenticated bridge.

## 7. Dry-run results

`Stn06ExperimentService.dry_run(scope="pilot", verify_oracles=True)`:

- `model_calls_made = 0`, `trials_persisted = 0`;
- baseline context absent and `experience_ids` empty for every task;
- treatment `experience_ids` and `candidate_hash` equal to the frozen values;
- dispatch bodies built for both arms within the 16 000-character bound, with
  experiment/task/arm identity in `metadata` for PlanTrace linkage;
- scoring distinguished clearly-labelled simulated failure
  (`success = false`) from simulated success (`success = true`);
- sealed oracle verification 8/8;
- ordinary retrieval returned zero results.

The simulated outcomes exercise the scoring path only; they are **not** results.

## 8. Integration points and remaining wiring

- **ExperimentStore:** experiment, variants A/B and the frozen preregistration +
  authorization are persisted there (verified on an isolated SQLite database).
- **MetricStore:** `record_trial()` is the wired path for real runs
  (`complete_trial` + metric group); the dry-run deliberately writes nothing.
- **PlanTrace:** each real run keeps the existing `worker:{execution_id}` trace;
  dispatch `metadata.experiment` carries `{name, task_id, arm, candidate_hash}`
  for stamping. Routing this block through the operational SIS `submit` path is
  a small, separately reviewed change that was intentionally **not** made here
  (the live Worker path is untouched).

## 9. Deterministic tests

- `tests/test_stn06_preregistration.py` — 6 passed: manifest balance/hashing and
  oracle/workspace disjointness, cross-`PYTHONHASHSEED` schedule stability,
  disabled-by-default config, freeze/select/preregister/dry-run on isolated
  SQLite, refusal to freeze a globally active MemCube, and scope/prohibition
  scoring.
- Navigation regression: `tests/test_sis_worker_execution.py`,
  `tests/test_sis_chat_execution.py`, `tests/test_sis_security.py`,
  `tests/test_stn05_experience.py`, `tests/test_stn06_preregistration.py` —
  **74 passed**.
- `py_compile` of all changed modules and `git diff --check` — clean.

## 10. Operator authorization still required

This preflight does **not** authorize paid runs. Before any metered execution:

1. Record the experiment-scoped authorization in the live database with the
   operator's reviewer identity and authorization id, e.g.
   `python scripts/stn06_preflight.py --persist --authorization-id <id> --reviewer <who> --notes "<rationale>"`.
   This freezes the preregistration into the existing `ExperimentStore`. No
   reviewer identity was fabricated here.
2. Verify a metering or hard-stop mechanism for cost, or accept the documented
   bounded-call substitute for the $2 target.
3. Authorize the paid A/B runs and the exact pilot scope.

Until then the experiment remains **NOT RUN**.

## Explicit non-claims

- No improvement is claimed; no A/B inference was run.
- The frozen experience is a narrow, untested-for-transfer hypothesis derived
  from one small task.
- A passing oracle here validates the harness, not the memory.
- STN04 evidence was not mutated; no acceptance record was fabricated; the
  withdrawn MemCube was not globally reactivated.

## Reproduce

```powershell
# read-only reconciliation of the STN05 records
venv\Scripts\python.exe scripts\stn06_reconcile.py
# deterministic preflight (no model call, no writes)
venv\Scripts\python.exe scripts\stn06_preflight.py --scope pilot --out docs\stephanie-next\stn06-preflight.json
# deterministic tests
venv\Scripts\python.exe -m pytest tests\test_stn06_preregistration.py -q
```
