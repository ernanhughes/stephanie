# SC4C activation package

**Status: package not built; activation OFF.** The five authorized repositories are not a clean, source-pinned release set. These checkout identities are the observed `HEAD` values on 2026-11-25; they are not claims that the working trees match those commits:

| Repository | HEAD | Worktree observation |
|---|---|---|
| Mission | `35b19c7c7ffc0b549f0816ca30c969b3ce3b96ff` | 11 porcelain status lines; SC4 work and other concurrent modifications coexist; untracked `node_modules` excluded |
| Stephanie | `a1605c3b08b3f34b79b098a1faa18e72c321faed` | 44 status lines before this routing change; Python 3.13, SE/SC stages, and this routing work are mixed |
| AI | `cf43eddc1f8109ea2f575ac290b6c1af849f3459` | 3 pre-existing modified files; not changed by this continuation |
| Worker | `a2e881c810c388fa2ce286a0b5169d4c93e791c5` | 19 pre-existing status lines; not changed by this continuation |
| Praxis | `cc7996138afeb3ab72e417fe94242e6f82cfd868` | clean at observation |

## Intended SC4 source set (not packaged)

- Mission: `backend/mission_control/api/improvement.py`, `backend/mission_control/work_observation/envelopes.py`, `backend/mission_control/work_observation/proposals.py`, `backend/mission_control/improvement/`, the SC4 API/E2E tests, and the Improvement frontend/page tests and source.
- Stephanie: `stephanie/improvement/`, SC4 integration tests and SC4 programme records. This continuation additionally changes `stephanie/services/llm_service.py`, adds `stephanie/services/free_first_chat.py` and `tests/test_free_first_chat.py`, and configures the gated SIS routing policy in `config/sis.yaml`.
- AI, Worker, and Praxis: dependencies used by the isolated integration, but no clean dependency package has been assembled. Their concurrent WIP is not included in any releasable artifact.

This intended list is not yet a safe allowlist: Mission `proposals.py`, Stephanie's Python/CI/dependency files, and other dirty files overlap unrelated work. No patch bundle, clean worktree, lockfile set, generated backend package, or signed manifest was made. Do not deploy from the current dirty checkouts.

## Handoff finding

The generic Mission proposal service still describes its `manual` route as a copyable commission handled outside Mission, because its general Worker route is not configured. SC4's dedicated integrated path does not need that manual transfer: its API accepts an exactly approved proposal, binds the matching active Work Envelope, and reserves dispatch while preserving the human approval gate. The existing Improvement UI test exercises approved → envelope → dispatch directly; the historical SC4B evidence used an additional manual handoff record and remains unchanged. No approval was automated, and no cross-application files need copying in the dedicated SC4 workflow. Only the stale UI source comment was corrected; the generic Mission routing policy was left intact.

## Runtime observations (read-only, 2026-11-25)

- `http://127.0.0.1:8790/api/health`: status `ok`, version `0.1.0`, database `ok`, schema `54`, AI and monitor enabled. This endpoint does not identify the exact source tree or database DSN.
- Listener PID 9324 runs the command `C:\Projects\mission\.venv\Scripts\python.exe -m mission_control --host 127.0.0.1 --port 8790`; its resolved executable is the uv-managed CPython 3.13.3. This identifies the process command, not the precise imported backend source snapshot.
- The served HTML references `index--VBqxldF.js` (SHA-256 `BF6724989E35E1A7E6D5B8860E70C4BA7CA576CF41C34C9FAC206652ADF3EA94`) and `index-ChtZL1FQ.css` (SHA-256 `9D9178FDAD1DFCDE8BE1A0F1F8267B631ABCE7BB3BC8F4F3F7CBA6491BD35D06`). These fingerprints identify the current static files, not a controlled deploy bundle.
- `GET /api/improvement/integration`: 200; Stephanie investigation is unavailable because the isolated run root is not configured; Worker execution advertises availability only after exact SC4 authorization; Praxis is Stephanie-owned after dispatch.
- `GET /api/improvement/records`: 200, source `isolated_sc3_artifacts`, formal acceptance `UNKNOWN`, zero worker model invocations, zero Mission writes in the record. Its experiment ID is `4205bddf27071a214ba99e45`; it is not the owner-review experiment `7cc37cb2e95dd58d098c3d80` and must not be conflated with it.
- The isolated review app at `http://127.0.0.1:8791/improvement` remains a separate process (PID 50148, Python 3.13) with its isolated runtime. It was not restarted or modified by these read-only checks.
- Read-only `:8791` proposal/investigation checks confirmed `wp:1ee64e8049444480` v2 is `reported_complete`; investigation `mission-eb1b88ab129917ed84c0e3c2` has experiment `7cc37cb2e95dd58d098c3d80`, candidate `034647ef6575a8a58461a4c8`, protocol digest `2373dd5ee2262af66b7a260da6ad27df62a1ce9f5a0c62d8e9be14016cd401c6`, four retained records and zero human decisions. Its `GET /api/improvement/records` response describes a separate `4205bddf27071a214ba99e45` artifact; use the investigation detail endpoint for the owner-review experiment.

The normal service is not declared activated by this report. No database write, migration, restart, deployment, or production promotion was performed. No exact database identity, runtime source provenance, migration plan, service startup manifest, required health-check matrix, or rollback-qualified frontend/backend artifact is available. Gate remains blocked.
