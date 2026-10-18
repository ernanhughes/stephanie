# SWF01-DB — Paper-Section History and PostgreSQL Replacement Verification

**Status: COMPLETE.** Read-only inspection of `co`, followed by a replacement
suite against real PostgreSQL in an isolated scratch database.
Facts are labelled **VERIFIED** (observed in the database or executed) or
**UNVERIFIED** / **NOT ESTABLISHED**.

Scripts used (kept outside the repo, in the agent temp dir):
`swf01_db_inspect.py`, `swf01_db_detail.py`, `swf01/part_d.py`.

---

## Part A — Repository and database identity

| Item | Value | Status |
| --- | --- | --- |
| Agent checkout `C:\Project\stephanie` | `ae8d7ff1` (`security: remove hardcoded credentials…`), clean | VERIFIED |
| Operator checkout `C:\Projects\stephanie` | `e794eaed`, clean | VERIFIED |
| `ae8d7ff1` files | `.env.example`, `config/sis.yaml`, `scripts/generate_configs.py`, `sis/config/config.yaml` — DB config untouched | VERIFIED |
| DB configuration source | `config/db/postgres.yaml` → `${oc.env:STEPHANIE_DB_NAME,co}` etc. | VERIFIED |
| **Inspected database** | `co` @ `localhost:5432`, user `co`, schema `public`, PostgreSQL **16.8** | VERIFIED |
| Session | `transaction_read_only = on` for every query | VERIFIED |
| Other databases on the same server | 26, including a separate `stephanie` (the Compose DB) | VERIFIED |
| Compose `stephanie` DB | Exists on the **same** server; auth as user `stephanie` failed, so it was **not** inspected | VERIFIED / not inspected |

No credential was displayed (`password=<masked len=2>`).

## Part B — Existing section rows

```sql
SELECT count(*), count(DISTINCT paper_id) FROM paper_sections;            -- 713 | 1
SELECT paper_id, count(*) FROM paper_sections GROUP BY 1;                 -- {2506.21734: 713}
SELECT count(*) FROM paper_sections
  WHERE id LIKE '%::sec-%' AND split_part(id,'::',1) <> paper_id;         -- 680
```

1. **Rows:** 713. **2. Distinct ids:** 713. **3. `section_index`:** 0…77.
4. **Ownership:** all 713 claim `paper_id = 2506.21734`.
5. **Prefix/ownership mismatch:** **680** rows carry a *related* paper's id prefix but the root `paper_id`. This is the source of "related content presented as root".
6. **`run_id`:** NULL ×713. **7. Titles:** stored titles are truncated from the 16-token era (e.g. `…parallel-in-time self-att`, cut mid-word). **8. Timestamps:** 712 rows created `2025-12-15 11:27:44.834627+00:00` (one bulk insert); 1 row created `2026-01-03`, **updated `2026-10-18 19:04:07+01:00`**. **9. Missing metadata:** `path` ×713, `content_hash` ×713, `token_count` ×713, `level` ×712 are NULL.

### The decisive row

```
id=2506.21734 | paper_id=2506.21734 | section_index=77 | level=0 | path=NULL
token_count=NULL | content_hash=NULL | created_at=2026-01-03 02:37:26
updated_at=2026-10-18 19:04:07  title="There were several emergent behaviors…CTM"
```

One row whose `id` is the **root arxiv_id** rather than `<source>::sec-N`, with
the ORM's `level=0` default, updated **during the pipeline runs of this session**
(minutes before the run-10038 blog). `section_index=77` is the last section
iterated.

### Composition

18 source papers, all present in `papers` (19 rows). Root `2506.21734` has 32
sections (`sec-0..31`) plus the odd row; related papers range 13–78 sections
(`2505.05522` 78, `2504.20571` 69, `1911.01547` 60, …).

### Verdict on the "swallowed IntegrityError" hypothesis — **REFUTED**

| Earlier claim | Verdict |
| --- | --- |
| `persist()` stamps every section with `id = arxiv_id` → PK collision | **VERIFIED** |
| `PaperSectionORM.path` is NOT NULL, so `flush()` raised `IntegrityError`, swallowed at `sections_cache.py:60-62` | **REFUTED** — live column is `path text NULL` (nullable); no exception is needed |
| `persist()` has never successfully written | **REFUTED** — it wrote, and is still writing (row updated today) |
| Rows served by `maybe_load` are from an older generation | **VERIFIED** — 712 rows bulk-inserted 2025-12-15 with `::sec-N` ids |

**Corrected mechanism:** `persist()` works, but `upsert` keys on `id` and `id` is
always the root arxiv_id, so all sections collapse onto one row and overwrite each
other. The 712 real rows came from a 2025-12-15 bulk load that used the
`<source>::sec-N` convention (so the blog generator's id-prefix repair was reading
a real signal) but stamped `paper_id = root`.

**Headings:** the blog's truncated H2s come straight from the stored 16-token-era
titles (a cache hit never re-summarises). There is **no evidence of two
generations of the same section**. What the data does show is two *distinct*
sections sharing a truncated title:

```
2201.11903: sec-10, sec-38  -> "chain of thought generally outperforms ablations by a large amount."
2506.04761: sec-37, sec-38  -> "hodlr (hierarchically off-diagonal low"
```

## Part C — Replacement dependencies

```sql
-- incoming FKs to paper_sections
SELECT … FROM pg_constraint WHERE contype='f' AND confrelid='paper_sections'::regclass;  -- 0 rows
-- outgoing
--   fk_paper_section        paper_id -> papers(id) ON DELETE CASCADE
--   paper_sections_paper_fk paper_id -> papers(id) ON DELETE CASCADE  (duplicate)
--   paper_sections_pkey     PRIMARY KEY (id)
```

- **No incoming FK** → replacing/deleting sections **cannot cascade** into any
  other table. `parent_id` is `integer`, has **no FK**, and 0 rows are set; the
  ORM's self-referential FK (`String` → `varchar id`) is not representable in
  `co` (a probe raised `operator does not exist: character varying = integer`).
- **CaseBooks are soft references:** 712 casebooks named
  `paper:2506.21734:<source>::sec-N` with `meta->>'section_id'`; `cases` link to
  `casebooks.id` (21,679 cases). Deleting sections leaves them **orphaned but
  intact**. `paper_run_features` is empty; `plan_traces` has no section columns.
- **Cross-paper id delete (reviewer's flag):** real but bounded — it can only hit
  another paper's row whose *id* claims the rebuilt paper, i.e. misattribution.
- **New gap found:** the old secondary delete only removed *incoming* ids, so once
  data is correctly owned a rebuild yielding **fewer** sections would leave
  `source::sec-M` orphans that `maybe_load(source)` still serves.

## Part D — Replacement suite against real PostgreSQL

`swf01_scratch` was created on the same server and given the **exact DDL** of
`co`'s `papers` + `paper_sections` (via `pg_dump --schema-only`, then
`transaction_timeout` stripped for the PG16 target). It was then exercised with
the **real** `PaperStore` + `PaperSectionsCache` (a `sessionmaker` with
`expire_on_commit=False`, matching `orm/base.py:122`). **`co` was never touched.**

```
T1.1 persist wrote 4 rows                                  PASS
T1.2 ROOT owns exactly its 3 sections                      PASS
T1.3 RELATED owns its own row                              PASS
T1.4 no missing required metadata                          PASS
T1.5 path/token_count populated                            PASS
T2.1 maybe_load(ROOT) returns 3                            PASS
T2.2 provenance preserved                                  PASS
T2.3 content preserved                                     PASS
T2.4 maybe_load(RELATED) is isolated                       PASS
T3.1 old generation gone                                   PASS
T4.1 five rows                                             PASS
T4.2 shrunk to one, no stale higher-index rows             PASS
T4.3 maybe_load agrees                                     PASS
T5.1 RELATED untouched by ROOT replace                     PASS
T6.1 'a_b' replace did not delete 'axb'                    PASS
T6.2 'a_b' replaced itself                                 PASS
T7.1 section for a non-existent paper is rejected          PASS
T8.1 duplicate PK raised                                   PASS
T8.2 previous generation intact (rollback)                 PASS
T9.1 repeated replace is identical                         PASS
T10.1 persist returns 0 on failure                         PASS
T11.1 mis-attributed row removed by the RELATED rebuild     PASS
22/22 checks passed
```

**`swf01_scratch` is left in place** for inspection (drop it when done).

## Fixes applied

| # | Change | Where |
| --- | --- | --- |
| 1 | Replacement delete is scoped to `paper_id = :pid OR id LIKE :pid::%` (LIKE wildcards escaped). Repairs misattribution, removes shrinking-rebuild orphans, and never touches another paper's own rows. | `stephanie/memory/paper_store.py` `PaperSectionStore.replace_for_paper` |
| 2 | Persistence failure is visible: `persist()` logs at ERROR and returns the count written; `run()` records `paper_sections_persisted` / `paper_sections_persist_failed` on the context and logs `PaperPipelinePersistFailed`. | `sections_cache.py`, `pipeline_runner.py` |

Unit tests: `tests/test_sections_cache_lifecycle.py` now has **16 tests**
(incl. shrinking rebuild, cross-paper protection, wildcard escaping, failure
visibility). Full suite: **39 passed**.

## Verification gaps

1. **ORM/DB drift.** `PaperSectionORM` declares `path NOT NULL`, `parent_id`
   `String` FK, `uq_paper_run_path`, and lacks `image_paths`/`domains`/`entities`;
   `co` has `path` nullable, `parent_id integer` (no FK), **no** unique
   constraint, and those three array columns. The ORM is not the source of truth
   for `co`. This drift is why the NOT NULL premise was wrong.
2. Origin of the 2025-12-15 bulk load (script vs older agent) — **NOT ESTABLISHED**.
3. Compose `stephanie` DB contents — not inspected.
4. Cost/runtime of a full rebuild: `SectionBuildTask` summarises **every** section
   (~713), so a rebuild is ~700 summariser calls (tens of minutes), unlike the
   ~18 calls of a cached run.

## Prerequisites for a safe live rebuild

1. Back up `paper_sections` (713 rows) and the 712 affected `casebooks`.
2. Decide the CaseBook policy — rebuilt sections may number differently, so some
   of the 712 `paper:<root>:<id>` casebooks become orphans.
3. Fix #1 is already applied (prevents orphaned higher-index rows).
4. Flag: `STEPHANIE_FORCE_REBUILD_SECTIONS=1` or
   `+agents.paper_pipeline.force_rebuild_sections=true` (NOT
   `+paper_pipeline.force_rebuild_sections=true`, which nothing reads).
5. Authorise the spend and wall-clock time; run once against `swf01_scratch`
   before `co`.
