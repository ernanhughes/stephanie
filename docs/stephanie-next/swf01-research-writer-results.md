# SWF01 — Research Integrity and First Stephanie → Writer Workflow

**Status: PHASE 1 IMPLEMENTED, UNIT-VERIFIED, NOT YET LIVE-VERIFIED.**
Phases 2–7 not started. Nothing committed or pushed (awaiting authorization).

Scope of this report: Phase 1 (paper-section lifecycle). Everything below is
labelled **VERIFIED** (I read the code / ran the tests and can point at the
evidence) or **UNVERIFIED** / **BLOCKED**.

---

## 1. Root cause of the stale/duplicated section generation

The review's three findings are confirmed, plus one it did not mention, which
turns out to be the true "why did run 10038 look like two generations" answer.

### 1.1 `force_rebuild_sections` was never read — VERIFIED

`grep -r force_rebuild` over `stephanie/` **and** `config/` at `c57a70e4`
returns **zero hits**. The flag:

* was not declared in `config/information.yaml`;
* was not read in `PaperPipelineRunner.run()`.

`run()` at `c57a70e4:95-108` was:

```python
sections = self.sections_cache.maybe_load(arxiv_id, role="reference")
if sections is None:                      # <-- only condition that rebuilds
    ...
```

Consequence: **the override I ran (`+paper_pipeline.force_rebuild_sections=true`)
was a no-op.** It created a top-level Hydra key that nothing consumed, and
`maybe_load` hit the cache. My earlier claim that run 10038 "rebuilt but kept
both generations" was wrong — run 10038 never rebuilt at all.

### 1.2 `PaperSectionsCache.persist()` destroyed identity and could not write — VERIFIED (code) / UNVERIFIED (DB effect)

At `c57a70e4`, `sections_cache.py:49-62`:

```python
for sec in sections:
    s = asdict(sec)
    s["id"] = arxiv_id          # every section gets the SAME primary key
    s["paper_id"] = arxiv_id    # every section claims the ROOT paper
    self.paper_store.upsert_section(s)
```

Three defects in four lines:

1. **Primary-key collision.** `PaperSectionORM.id` is the PK (`orm/paper.py:66`)
   and `PaperSectionStore.upsert` looks up by `id` (`paper_store.py:576-581`),
   so all sections of a paper collapse onto one row.
2. **Wrong ownership.** `paper_id` is forced to the *root* arXiv id
   (`orm/paper.py:67` is the FK), which is precisely how related papers' content
   ended up attributed to the root paper.
3. **The write cannot succeed.** `asdict(PaperSection)` yields the dataclass
   field names (`section_id`, `paper_arxiv_id`, …), not ORM columns, and the
   required `PaperSectionORM.path` (**`nullable=False`**, `orm/paper.py:73`) is
   never supplied. The `flush()` raises `IntegrityError`, which is swallowed by
   the blanket `except Exception` at `sections_cache.py:60-62`.

So `persist()` has, with high probability, **never successfully written a
section**. The rows currently being served by `maybe_load` are from an older
code generation, which is the real reason a "regenerated" heading appeared
alongside a stale one. **This is the claim I most want checked against the DB**
(see §6 BLOCKED-1) because it changes the migration story.

### 1.3 `replace_for_paper()` was not usable as the rebuild mechanism — VERIFIED

`paper_store.py:537-568`:

* it called `self.delete_by_paper_id(paper_id)` **from inside** its own
  `op(s)`, and `delete_by_paper_id` again calls `self._run(op)`
  (`paper_store.py:528-535`) → a **second `session_scope`** (`base_store.py:39-46`),
  so delete and insert are not in one transaction;
* it omitted `path` (NOT NULL), `role`, `level`, `content_hash`, `token_count`.

### 1.4 Bonus: a rebuild would have crashed anyway — VERIFIED

`SectionBuildTask` constructed the dataclass as
`PaperSection(id=..., paper_role=...)` (`section_build_task.py:202-218` and
`:272-283`), but `PaperSection`'s fields are `section_id`, `paper_arxiv_id`,
`role`, `section_index`, … (`data.py:358-403`). A forced rebuild would raise
`TypeError: unexpected keyword argument 'id'`. This bug was invisible because
the build path was never reached (see 1.1).

---

## 2. Corrections made

| File | Change |
| --- | --- |
| `paper/pipeline/sections_cache.py` | `maybe_load(..., force_rebuild=False)` bypasses the cache. `persist(..., replace=False)` now groups sections by their **own** `paper_arxiv_id`, emits stable ids `{source}::sec-{index}` (the convention the blog generator parses), and supplies every NOT-NULL ORM column (`path`, `role`, `level`, `section_index`, `content_hash`, `token_count`). Duplicate indices are dropped, never collided. `_sections_from_orm` now maps real ORM column names. Persist failures are logged with context instead of silently swallowed. |
| `memory/paper_store.py` | `PaperSectionStore.replace_for_paper` rewritten: delete + insert in **one** session (no nested `_run`), deletes both the paper's rows and any row squatting on an incoming id, and populates all required columns. |
| `paper/pipeline/pipeline_runner.py` | `run()` loads the cache **per source paper in the graph** (root first), honours a new `_force_rebuild_requested()` (context → stage cfg → `STEPHANIE_FORCE_REBUILD_SECTIONS`), and persists with `replace=True` whenever a rebuild happened, so old and new generations can never both be active. |
| `tasks/section_build_task.py` | Fixed the dataclass kwargs (`section_id=`/`role=`) at both construction sites. |
| `config/information.yaml` | Declared `sections_cache_enabled: true` and `force_rebuild_sections: false` on the `paper_pipeline` stage. |

Ownership decision: persisted rows now carry the **true source paper** in
`paper_id`, so `maybe_load` is per paper and the loader must iterate the graph
(which it now does). Stale rows previously stamped with the root id are deleted
by `replace_for_paper` on the next rebuild for that paper and re-attributed.

## 3. Deterministic tests

`tests/test_sections_cache_lifecycle.py` (new, 11 tests) — no DB, no LLM:

1. persisted ids are unique and source-attributed; `paper_id` is the source paper;
2. every persisted key is a real `PaperSectionORM` column and every NOT-NULL column is populated;
3. related-paper sections are stored under their own paper;
4. duplicate section indices are dropped, not collided;
5. persist → load round-trip preserves text/title/role/source;
6. empty cache returns `None`;
7. `force_rebuild=True` bypasses the cache;
8. `replace=True` removes the previous generation (old truncated title gone);

9. replace is idempotent;
10. force-rebuild flag resolves from context, stage cfg and env var;
11. `SectionBuildTask` produces valid `PaperSection`s (regression for 1.4).

```
$ venv\Scripts\python.exe -m pytest tests\test_sections_cache_lifecycle.py \
    tests\test_paper_blog_generator.py tests\test_worker_model_bridge.py \
    tests\test_writer_runtime_override.py -q
34 passed in 14.92s
```

## 4. Acceptance steps

| # | Step | Result |
| --- | --- | --- |
| 1 | Build a paper with several distinct sections | PASS (unit) |
| 2 | Persist them and verify unique IDs | PASS |
| 3 | Reload them and compare content and provenance | PASS |
| 4 | Force a rebuild with revised headings | PASS (unit) — flag now exists and is honoured |
| 5 | New active set replaces the previous generation | PASS (unit) |
| 6 | No duplicate headings from mixed generations | PASS (unit); **live** check pending |
| 7 | Related-paper sections retain their own source IDs | PASS |
| 8 | A failed rebuild cannot leave a partially active result | PARTIAL — `replace_for_paper` is now one transaction; not yet proven against Postgres under induced failure |
| 9 | Repeated runs are idempotent | PASS |
| 10 | Existing CaseBook references remain valid | **BLOCKED** — needs the live DB; ids are unchanged in format, so existing `casebook` meta keys are unaffected, but unverified |

## 5. Known limitations

* `persist` still tolerates a failure (logs and returns the count written); it no
  longer fails silently, but a caller that ignores the return value would not notice.
* Section ordering within a paper is by `section_index`; across papers it is
  graph-node order with the root first (previously a single interleaved list).
* No run-level accounting / cost visibility (SWF01 phase 3 territory).

## 6. BLOCKED items (need a decision or authorization)

* **BLOCKED-1 — live DB inspection.** I did not query production rows in `co`.
  I need authorization to read `paper_sections` for `2506.21734` (read-only) to
  confirm §1.2's "persist never wrote" claim and to size the stale-data cleanup.
  `replace_for_paper` performs **destructive** deletes, so per the brief it must
  be exercised against an isolated database first.
* **BLOCKED-2 — live pipeline run.** A `force_rebuild_sections=true` run will now
  actually rebuild (spending money at ~$0.20/M under the current default policy)
  and will delete/rewrite section rows. Awaiting authorization.
* **BLOCKED-3 — Phase 7 integration matrix** (Next / Improve / Workbench /
  Sidecar). Not attempted: it requires inspecting those repositories' public
  contracts, which was out of scope for this unit of work.
* **Not started:** Phase 2 (near-duplicate section selection), Phase 3
  (reviewable output record + admission gate), Phase 4 (real Writer handoff),
  Phase 5 (SIS surface), Phase 6 (live verification).

## 7. Review corrections I accept

* §C of the review is right: `select_route` exempts `preferred_model` from the
  ceiling (`services/worker_model_bridge.py`), so the $0.40/M figure is a
  *selection* cap, not a spend cap. The wording should be corrected and
  per-run usage surfaced in SIS.
* §A.2 is right that this is an identity/persistence defect, not a heading bug.
* I withdraw my earlier claim that commit `c57a70e4` fixed duplicate headings.
