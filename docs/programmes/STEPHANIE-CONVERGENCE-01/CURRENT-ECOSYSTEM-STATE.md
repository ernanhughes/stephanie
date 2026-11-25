# Stephanie Convergence 01 — Current ecosystem state

**Snapshot:** 2026-11-25. Repository identities are local HEADs; remote refs were already current at inspection (`origin/main` equaled `HEAD`). Source trees below are not all clean. No external repository was changed.

| Repository | `main` HEAD | Tree | Current status relevant to convergence |
|---|---|---|---|
| Stephanie | `98388a4b06374f515ae4e84c3e87f0466e8e0fb7` | 32 status entries | Python 3.13 / SE1–SE2Q implementation and reports are substantial local WIP, including untracked `stephanie/improvement/` and `tests/`. Preserve; source identity is not yet a clean distributable. |
| AI | `cf43eddc1f8109ea2f575ac290b6c1af849f3459` | clean | Shared `ai` 0.2.0 source API; no tracked source changes. |
| Worker | `a2e881c810c388fa2ce286a0b5169d4c93e791c5` | 19 status entries | Strategy, execution ledger, provider and runtime work is uncommitted. H6/H3 Worker changes are still WIP, so a future integration must pin an accepted source artifact. |
| Praxis | `cc7996138afeb3ab72e417fe94242e6f82cfd868` | clean | Current local library and contracts exist; no Stephanie adapter or running Praxis service was observed. |
| Mission | `a1410fa8c1757adda5cf238e43490939ffc5f3e9` | 16 status entries | Work Outcome, H6 adapter/UI, Publication and H6 evidence work includes uncommitted changes. Live service is 0.1.0, schema 54. |
| Writer / ECK | `65a2683985904651824b1772eda11fd66974ac18` | 50 status entries | Significant concurrent Writer/ECK and Publication work is dirty/untracked. No stable standalone ECK contract for Stephanie was identified. |
| Inspect | `ea693ccc7255ecc744db796f35c57d0e2e8dcb7c` | clean | Repository capability inventories are its output; it remains descriptive, not an execution or acceptance authority. Current generated inventory was not refreshed for this review. |
| AITest | `433888b7f4390a191aeaff380af549ebcfb7ebf` | clean | Mission-integrated test governance is operational, but Stephanie's local AITest registry snapshot is stale relative to current Stephanie files. |
| Workbench | `55b794b2efa3ccd790fa191beb9cc1ab6ef2e1f1` | 2 untracked entries | Publication proposal work is in progress; Workbench remains an author-owned writing workflow, not the improvement lifecycle authority. |

All listed branches are `main`; local `origin/main` refs matched their respective HEADs. This does not establish that uncommitted work is approved or remotely published.

## Programmes and changes since H6A

- **Harness H0–H5:** owner-accepted within documented scope; H3 is deterministic offline only; H4 remains conditional and H5 was not authorized. See Mission `docs/programmes/HARNESS-EVOLUTION-01/H5-FINAL-REPORT.md` and stage ledger.
- **H6:** accepted for isolated integration only. Mission's `/api/harness/catalogue` answered 200 and reported the connected Worker 0.2.0 plus strategy catalogue. This is a read-only current observation. H6's recorded qualification remains one scripted execution with zero model inference. Its source is still dirty; no activation was performed here.
- **H6A / activation:** the old claim that Mission/Worker are simply unavailable is stale: both are running, Mission serves schema 54, and its Harness catalogue reaches Worker. The formal H6 activation gate is still not closed: concurrent Mission/Worker source is dirty, service-to-source identity is not bound, no provider execution was demonstrated by H6, and H6's operator-authentication/actor limitation remains. No activation is authorized by this assessment.
- **Work Outcome:** W9 final status is conditional acceptance/closure recommended. Mission is live at 0.1.0/schema 54. Existing contracts include Work Episodes, Requirements, Findings/Evaluations, Work Envelopes, corrective Proposals and explicit acceptance policy. W9 records remain the authority for deferred requirements and operational policies.
- **Publication:** Mission's Publication state/decision files are modified and `pub08/experiment/arm_b1`, `arm_c1`, evaluation scripts and human-review files are untracked. This is active concurrent work; it is not a verified prerequisite for the proposed generic vertical slice.
- **Inspect:** repository has a descriptive capability-inspection system and snapshots; no Stephanie execution adapter or current refreshed Stephanie inventory was established here.
- **AITest:** AITest-02E reports 28 project identities, 28,701 tests, 3,010 assessed, 13 failing, one quarantined, adaptive operation on mission/alive/evidence-repair, with its worker running. Those numbers are a dated report, not a fresh service query. Mission health now says schema 54 whereas AITest-02E's live report was schema 51; the AITest view needs reconciliation before it is used as a release gate.
- **Praxis:** Praxis's source tree is clean and contains current effectiveness, harness, verification and readiness modules. PRAXIS-13 readiness report remains `not_yet_evaluable` with no eligible new accepted outcomes; no comparative study is active for Stephanie.
- **ECK:** Writer's uncommitted status means its experimental modules cannot be treated as stable APIs. Prior ECK/Praxis ownership records say ECK owns Writer-specific procedural experience and has dependencies that block broad imports.

## Current operational observations (read-only)

- Mission `GET /api/health`: `status=ok`, app version `0.1.0`, database `ok`, schema version `54`.
- Mission `GET /api/harness/catalogue`: 200; Worker reachable, version `0.2.0`.
- Worker `GET /health`, `/v1/strategies`, and `/v1/models/catalogue`: 200. `opencode` executor unavailable (`uv_spawn` error); `opencode-work` and model executors report available. Availability/catalogue does not prove a model invocation succeeds.
- Ollama provider health was not established with its provider API during this assessment; no model call was made. No Praxis service was observed.
- Workbench process is present on port 7420; its two local changes remain untouched.

No operational database records were queried or mutated. No service was restarted, no model or paid call was made, and no external repository was modified.

## Preserved SE2Q findings

SE2 remains inconclusive: 24 parser-observed invalid JSON outputs, with no retained raw bodies or provider usage/termination data. SE2Q accepted the demonstrated direct AI structured path only; 3/3 direct outputs were structurally/citation-valid, but the post-run semantic checklist was complete on 1/3. Worker produced 1 usable result in 5 observed outcomes, including one blocked request. The exact `generate_structured()` probe passed once. Keep the original schema-minimum and Worker follow-up manifest hash limitations documented; do not repair or reinterpret historical artifacts. Original prompts are exposed and not fresh holdouts. No SE3 was run.
