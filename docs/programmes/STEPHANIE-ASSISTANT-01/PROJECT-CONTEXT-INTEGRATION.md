# Project context integration

`sis/services/mission_project_context.py` calls only GET endpoints on the configured loopback Mission API:

- `GET /api/projects?include_archived=false` for the authenticated UI picker and a small project summary projection.
- `GET /api/projects/{project_id}` for a selected project, including bounded task/decision/checkpoint/recommendation summaries.

No Mission database connection or write endpoint is used. The picker returned 81 projects from `/api/projects` during verification. A natural query containing “harness” filters current summaries locally; a selected project uses the project-detail GET. Each context carries the local observation time and endpoint source, and the API record `updated_at` is preserved. Lists/record groups are capped and truncation is declared.

Any project projection is `private_project`, is scanned for protected markers, and is forced onto the local model path. If Mission is unavailable, project questions receive a deterministic answer saying current state cannot be verified. Previous reports are not used as current-state substitutes.
