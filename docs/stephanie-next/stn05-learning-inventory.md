# STN05 — Learning Infrastructure Inventory and STN04 Review

**Review status:** STN04 was explicitly **ACCEPTED for the calculator.py scope** by the authenticated SIS reviewer. The separate lesson-candidate review remains **PENDING**; no MemCube promotion was inferred.

## STN04 evidence reviewed

Source records are summarized in `stn04-first-agent-execution.md` and were retrieved from ChatStore/PlanTrace after a fresh PostgreSQL connection:

- Conversation `1667`, user message `83557`, assistant message `83558`.
- SIS execution `sex_8991baa0c2724f35878292c83651588f`.
- Worker request `wrq_ab52b2dab127f5e3b532b563e898eea6`.
- PlanTrace `worker:sex_8991baa0c2724f35878292c83651588f`.
- Worker `AWAITING_ACCEPTANCE`; SIS `COMPLETED_UNVERIFIED`; no acceptance ID.
- Independent `python3 -m unittest` verification: 2 tests, exit 0.
- SIS human review: `ACCEPT`, reviewer `sis`, review `stn05-human-review-stn04-accept-001`, accepted scope `calculator.py`. This did not create a Worker acceptance ID.

The final disposable-workspace file contains:

```python
def add(a, b):
    return a + b
```

The STN04 record reports the initial defect as subtraction (`return a - b`) and records `calculator.py` as the only changed artifact. **No raw before/after patch was persisted.** The following is therefore only a reconstruction from the recorded fixture baseline and final source, not an original diff artifact:

```diff
-    return a - b
+    return a + b
```

The persisted result summary reports both tests passing, and the independent verifier output says `Ran 2 tests … OK`. This is narrow evidence of the calculator task, not evidence of general coding reliability. A reviewer should inspect the final file and test cases; if the missing original diff is material, request a rerun with diff capture rather than treating this reconstruction as source evidence.

**Review conclusion:** the authenticated human reviewer accepted only `calculator.py`, after reviewing the final source, both test cases, and the independent verifier. The raw patch artifact is still unavailable; the SIS evidence page discloses this limitation. The decision does not establish general coding reliability.

## Component inventory

| Component | Status | Findings and STN05 use |
|---|---|---|
| ChatStore | **VERIFIED** | Stores the conversation, user/assistant messages, and bounded Worker execution metadata. STN05 review history is appended atomically to the existing execution JSON; no new chat table. |
| PlanTrace / PlanTraceStore | **VERIFIED for the STN04 path** | STN04 has submission/result steps with execution, Worker, container, and verifier references. Experience extraction requires matching trace, conversation, user message, and Worker request identities. |
| MemoryTool / ServiceContainer | **VERIFIED for normal registration** | A fresh process initialized `MemoryTool` and read CaseBook, MemCube, Experiment, Metric, ChatStore, and PlanTrace stores. The STN05 service is registered through the existing SIS ServiceContainer and reads Hydra configuration. |
| CaseBookStore | **IMPLEMENTED; STN05 CRUD verified in isolated SQLite** | Existing CaseBook/Case/Scorable models can hold a provenance-backed draft candidate in JSON metadata and a candidate scorable. STN05 adds idempotent candidate upsert/review methods; it does not create tables. |
| MemCubeStore | **IMPLEMENTED; STN05 lifecycle verified in isolated SQLite** | Existing `insert`, `search`, and `set_refined_result` support approved experience storage, lookup, and withdrawal. STN05 uses the store directly and requires approval metadata and an explicit approved-source filter. |
| MemCube domain/client | **PARTIAL / legacy client broken** | `MemCube` and `MemCubeORM` model a scorable plus lifecycle metadata. `MemCubeClient` is not used by the STN05 path; inspection found its constructor does not initialize `self.store`, and its methods call `save_memcube`, which is absent from `MemCubeStore`. STN05 uses `memory.memcubes` instead. |
| ChatToCaseBookAgent | **SOURCE-PRESENT; unsafe for this purpose without its own gate** | It converts broad conversation history and uses an idempotency bus. It does not require accepted Worker execution or separate lesson approval. STN05 does not invoke it. |
| CBRService retrieval | **NOT USED for approved experience retrieval** | Its CaseBook path calls a `casebooks.search` method not present on the current store; the fallback searches global embeddings and does not enforce STN05 approval state. STN05 instead uses `MemCubeStore.search` with an exact approved source/type and checks approval flags before returning anything. |
| PACS skill extraction/validation | **SOURCE-PRESENT; not appropriate for this first capture** | The preparation/validation path builds RLVR/training artifacts and may load models. STN05 makes no training or skill promotion calls. |
| ExperimentStore | **IMPLEMENTED; assignment stability repaired/tested** | Existing experiment, variant, trial, and metric APIs can record experiments. STN05 replaces process-randomized `hash()` assignment with stable SHA-256 assignment and tests it under multiple `PYTHONHASHSEED` values. No live experiment or trial was created. |
| MetricStore | **SOURCE-PRESENT; not a generic STN05 experience store** | The inspected API is focused on critic runs/models. STN05 does not treat it as a general case-outcome store. |
| TrajectoryStore / trajectory retrieval | **SOURCE-PRESENT / not demonstrated for OpenCode execution experiences** | The current schemas and contracts do not provide the STN04 human-review and independent-verifier gates. Not used as a substitute for CaseBook/MemCube. |

## ORM finding

A minimal fresh import of only ChatStore/PlanTrace ORM modules can fail mapper configuration with an unresolved `TrainingStatsORM` relationship. The normal `MemoryTool` path imports and registers `TrainingStatsStore`/`TrainingStatsORM` before querying; a fresh process successfully initialized that normal service and read the required CaseBook, MemCube, Experiment, Metric, ChatStore, and PlanTrace stores. Thus the observed partial-import failure does **not** currently block the normal STN05 store path, and no schema or mapping change was made.

The CaseBook/MemCube write/read lifecycle is tested against an isolated SQLite database with the PostgreSQL-only CaseBook tag type adapted for that test. After the explicit execution acceptance, the real STN04 execution has one pending-review CaseBook candidate (CaseBook `1744`, Case `21680`, Goal `2986`). A fresh normal `MemoryTool` process read it back with the source PlanTrace; no MemCube was promoted.

## Review and learning gates

1. **Execution review:** ACCEPT recorded by the authenticated SIS reviewer for `calculator.py`; Worker status and `acceptance_id` were not changed.
2. **Candidate extraction:** passed after execution acceptance and verifier/provenance checks; candidate `exp_33fd63f133b44b21275681e5af0aabb7` is persisted as `PENDING_REVIEW`.
3. **Experience review:** pending; the candidate's lesson has not been human-approved.
4. **MemCube promotion:** not performed. Pending, rejected, withdrawn, stale, and superseded candidates remain excluded from retrieval.
5. **Measured improvement:** not established; reserved for the controlled STN06 evaluation.
