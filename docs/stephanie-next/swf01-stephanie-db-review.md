# SWF01 — `stephanie` Database Review and Repair

**Status: reviewed and repaired.** The live database is intact; the schema has
been brought up to what the ORM requires.

## 1. The database the pipeline was actually writing to

`C:\Projects\stephanie\.env` pointed at **`co`**, not the live database. The
templates (`.env.example`, `.env.docker.example`) and `.env` as it now stands both
say **`stephanie`**; `.env_example.txt` documents the older `co` values.

Consequence: the pipeline runs in this session (10039, 10040) wrote to the
**legacy `co` database**. All the "damage" reported earlier is confined to `co`.
`stephanie` was never written to.

Verification of the live database:

```
db=stephanie  user=stephanie  (credentials from .env — connect OK)
paper_sections 712 rows | papers 19 | paper_runs 811 | casebooks 1008 | cases 20823
paper_sections.updated_at: max = 2025-12-15 11:27:44+00   (a single bulk insert)
rows updated in the last 30 days: 0                        (no damage)
2506.21734::sec-0 = "The HRM was inspired by three principles of neural computation observed in"
```

The root paper's own sections are **intact and correct**.

## 2. Schema drift vs the ORM (repaired)

The live schema predates the current ORM — `schema_raw.sql` (2026-01-03) is a dump
that matches the live DB, and there is no migration tooling in the repo. Eleven
paper-pipeline tables were compared against their ORM models:

| Table | Problem found |
| --- | --- |
| `paper_sections` | **missing `role`** (ORM: NOT NULL, default `reference`) |
| `paper_run_features` | missing `paper_id`, `updated_at` (both NOT NULL) |
| `paper_docling_pages` | missing `payload` (NOT NULL) |
| `paper_run_comparisons` | missing `left_run_id`, `right_run_id`, `winner_run_id`, `judge_source`, `scores`, `meta` |
| `paper_reference_graph_snapshots` | **table missing entirely** |

`paper_sections.role` was the blocker: `PaperSectionORM` selects and inserts it, so
**any ORM read or write against `stephanie` would have failed** with
`UndefinedColumn`. `PaperStore.get_sections_for_paper()` would have raised and
`maybe_load` would have silently degraded to "no cache".

All five were empty tables except `paper_sections`, so the repair is purely
additive and cannot lose data:

* backup: `paper_sections_swf01_backup` (712 rows, created before any change);
* `ALTER TABLE ... ADD COLUMN IF NOT EXISTS ...` for the 10 missing columns;
* `paper_reference_graph_snapshots` created from the ORM definition.

After the repair, the ORM diff reports **problems: none** for all 11 tables, and a
real read works:

```
PaperStore.get_sections_for_paper('2506.21734') -> 712 rows
PaperSectionsCache.maybe_load('2506.21734')     -> 712 sections
```

## 3. The cache bug that would have corrupted `stephanie`

Verified on the live data:

```
section_id      = 2511.02360::sec-0
paper_arxiv_id  = 2506.21734        <- disagrees with the id prefix
```

Every row in the live DB has `paper_id` stamped with the ROOT paper (680 of 712
rows belong to 17 other papers). `PaperSectionsCache` derived a section's source
from that column, so on a cache-hit write it:

1. re-attributed every section to the root paper, and
2. rewrote the id as `2506.21734::sec-<index>`, colliding across source papers.

That is exactly what happened to `co` during run 10039: `2506.21734::sec-0` went
from HRM content to `Vision-Language-Action (VLA) models to "think`, and the blog
for run 10040 stopped being about the paper at all.

**Fixed** in `stephanie/components/information/paper/pipeline/sections_cache.py`:
the **id prefix is authoritative** for a section's source paper, in both
`_group_by_source` (writes) and `_sections_from_orm` (reads). Two regression tests
cover the mis-attributed shape — the case the earlier `swf01_scratch` suite
missed, because it seeded correct ownership.

## 4. Remaining problems in `stephanie` (not yet fixed)

* **`papers.title` is empty for all 19 papers** → the blog headline renders as
  `# Paper 2506.21734`. The generator falls back to a placeholder.
* **680 of 712 section rows are mis-attributed** (`paper_id` = root). The reads are
  now correct (id prefix wins), but the column itself is still wrong; a forced
  rebuild would re-home them properly.
* **All stored section titles are from the 16-token era** and truncated
  (`A halting mecha- nism…`); two title pairs collide within a source paper.
* `path`, `content_hash`, `token_count` are NULL for all 712 rows.
* `paper_sections.role` is now `reference` for every row (the DB held no role
  information); the blog generator does not depend on it (it uses the id prefix).

## 5. Legacy `co` database

`co` was written to by runs 10039/10040 and now holds 759 rows, 79 of them
root-prefixed with mixed content (the 32 original HRM rows were overwritten). It
is not the live database; no backup exists. Left untouched pending a decision —
it can be cleaned or ignored.
