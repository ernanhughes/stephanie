# Stephanie Next — Migration and Data Plan

**Status:** STN01 database alignment and STN02 chat persistence completed without a schema migration.

## Findings

- Chat models are `ChatConversationORM`, `ChatMessageORM`, and `ChatTurnORM` in `stephanie/orm/chat.py`; existing IDs are integer primary keys and messages/turns reference conversations/messages.
- Existing chat rows live in established PostgreSQL databases. STN01 read one legacy conversation through ChatStore to verify compatibility; no conversation text was printed, copied or modified.
- STN00 found Hydra database `co` and a hard-coded ORM target `verity`. STN01 replaced the core runtime path with an engine built from Hydra `db`; direct pgvector connection parameters derive from the same URL. Read-only checks report configured/ORM/direct identity `co`; `verity` was not modified.
- The database password is supplied through `STEPHANIE_DB_PASSWORD`; no module-level engine is created during import. Legacy aliases require explicit initialization from the composed Hydra `db` node and are not used by MemoryTool.
- `sis/db.py:init_db()` no longer creates tables by default; schema creation requires an explicit local/test opt-in. It is not a migration ledger or part of SIS startup.
- No Alembic configuration, migration directory, or ordered migration chain was found. `schema_raw.sql` and `scripts/schema.py` are schema dump/conversion artifacts, not a verified forward/rollback migration system.

## STN00 changes

No DDL/schema change is needed for the first code slice:

- Existing conversation/message/turn tables are reused.
- New SIS conversations use `provider="stephanie"`, `meta.source="sis_live"`.
- Assistant `meta` stores runtime/model/cache metadata.
- STN02 uses existing JSON metadata for the single local owner, request/idempotency ID, execution status and observed model provenance; it atomically writes the assistant message and existing turn relationship. No column/table/index was added.
- `ChatStore.add_messages()` appends at `max(order_index)+1`, locks the parent conversation row on PostgreSQL, and updates `updated_at`.
- Existing IDs and imported conversation metadata are untouched.

## STN01 changes and verification

- `stephanie/orm/base.py` now owns structured URL construction and engine creation. SQLAlchemy `URL.create` handles credentials without URL string concatenation.
- `MemoryTool` binds ORM stores and the raw PostgreSQL/pgvector connection to the same Hydra database node, exposes a redacted `database_identity()` diagnostic, and closes both resources.
- Store retries dispose the pool attached to the relevant sessionmaker rather than a separate module-global database.
- Standalone administrative/read-only scripts now use Hydra database configuration; data-changing tools default to dry-run. Optional shadow/visualization scripts accept explicit DSNs. The destructive Nexus benchmark requires a dedicated scratch DB and explicit confirmation.
- Read-only checks confirmed identity `co`, one legacy conversation with six messages, and existing PlanTrace data. No schema was changed.
- A controlled one-stage variant of `config/code.yaml` used existing `LoaderAgent` on a temporary JSONL fixture with bus/model scoring and embedding indexing disabled. Repeatable `scripts/stn01_smoke.py --execute` run/trace `10026` was queried after a fresh MemoryTool connection: trace status `completed`, one step.
- Earlier smoke trace `10021` is retained as `incomplete` with diagnostic metadata after the first trace-completion guard defect. Successful runs `10022`–`10026` remain intact.

## STN02 chat persistence and data safety

- An authenticated two-turn SIS request created conversation `1665` in configured `co`; messages `83552`–`83555` and turns `37314`/`37315` were committed through ChatStore.
- A separate Python process reopened MemoryTool and read the conversation, message ordering, turn, provider/model metadata, usage, and completion status.
- Test records `1659`–`1665` are explicitly titled STN02 real-model acceptance conversations. No pre-existing conversation/message/turn was updated or deleted.
- No schema DDL, `create_all`, migration, history copy, or PlanTrace write was performed for chat. Ordinary chat deliberately has no PlanTrace until a future attributable workflow/run link is specified in STN03.

## Required order before any schema change

1. Select the canonical database source and correct the configured-vs-hard-coded engine discrepancy. Verify every store the SIS profile uses against the same database. Do not use a new chat-only database as a workaround.
2. Record table/column/index/FK inventory from the selected database and compare it to ORM metadata. Preserve a sanitized schema snapshot and a real backup under the operator's existing policy; do not commit database dumps or private data.
3. Establish the migration tool/runner and ownership. Alembic is a candidate, not an assumption; no dependency or migration system was added in STN00.
4. If adding run tracking, create only new tables/columns with nullable/additive FKs to existing conversation IDs. Proposed run records need stable run ID, conversation FK, runtime/model/version, status/timestamps, workspace reference, bounded usage, and evidence/trace references. Proposed event records need sequence, event type/version, bounded/redacted JSON payload and retention.
5. Test forward migration on an isolated copy containing representative imported conversations, empty conversations, multi-message ordering, turns, annotations and legacy JSON metadata. Verify all old IDs and counts remain unchanged.
6. Verify downgrade/rollback limitations. Prefer rollback of the new application feature while retaining evidence; do not drop run history by default.
7. Apply only with explicit authorization and a verified backup. No startup-time schema mutations.

## Migration risks

| Risk | Control |
|---|---|
| Wrong DB gets migrated | Assert configured/ORM/direct identity before any migration |
| Legacy IDs or imported provenance change | Additive migrations only; immutable legacy PKs and `external_id` |
| `create_all` conceals drift | Do not use it as an upgrade mechanism; schema-diff and migration checks |
| JSON metadata grows without bounds | Keep new event/evidence data bounded and normalized where queryable; define retention |
| Chat turn pairs become ambiguous | Store explicit message/run references; preserve `order_index`; test old imports and live append |
| Duplicate concurrent message indices | Parent row lock for PostgreSQL; add database uniqueness only after legacy duplicate audit |
| Secret/model data leaks into history | Whitelist metadata fields; redact keys and avoid persisting raw provider config |
| External agent sessions are mistaken for conversations | Distinct run IDs/table with an FK to the existing conversation |

## Verification before considering migration complete

- Backup restore drill succeeded on an isolated database.
- Migration applied and rolled back as documented, with row counts and PK checks.
- Legacy SIS list/detail/score routes read existing data.
- Chat import pipeline still deduplicates and preserves source IDs.
- PlanTrace, CaseBook, MemCube and experiment reads/writes still target the canonical database.
- No unapproved schema writes occurred during SIS boot.
