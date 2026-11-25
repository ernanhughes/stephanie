# Test and verification evidence

## Short deterministic suites

- `tests/test_conversation_policy.py tests/test_mission_project_context.py -q`: 11 passed in 0.71 s, including oversized Mission projection coverage.
- `tests/test_free_first_chat.py -m long -q`: 13 passed in 6.19 s; marked `long` and run because SA1 changes the routing gate.
- `tests/test_sis_chat_execution.py -m long -k "one_conversation_continues_after_model_change or provider_failure_preserves" -q`: 1 passed, 30 deselected in 26.56 s; marked `long` and run because continuity is an acceptance gate. The failure/retry name did not match a collected test, so this run verifies continuity only.
- Python 3.13 `compileall` passed for changed runtime modules.
- `git diff --check` passed before the final review fix; rerun before commit.

## Live bounded checks

- Local SIS startup reached `/healthz` ready on the isolated DB.
- `/chats/mission/projects` returned 81 current project summaries via Mission API.
- `/chats/1` returned HTTP 200 and the conversation form.
- Same-key POST replay returned HTTP 200 with the stored response and `replayed=true`.
- Fresh-process ChatStore read found the same conversation with 4 messages and 2 turns.

No full backend/frontend suite or unrelated long corpus was run. `uv run` began provisioning the new environment concurrently and hit shared `.venv` install races; it was stopped. Tests were run with the installed Python 3.13 interpreter instead. The unrelated capability-resolution long corpus was deliberately deferred.
