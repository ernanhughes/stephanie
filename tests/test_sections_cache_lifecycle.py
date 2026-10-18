"""Deterministic tests for the paper-section cache lifecycle (SWF01 phase 1).

These pin the fixes that the SWF01 review identified:

  * every persisted section keeps a stable, unique, source-attributed id
    (previously ``persist`` overwrote every id with the ROOT arxiv_id);
  * every persisted row carries the ORM's required columns (``path`` was
    missing, so the write raised IntegrityError and was silently swallowed);
  * ``maybe_load`` can be told to bypass the cache (the force-rebuild flag
    was read by nobody);
  * a rebuild REPLACES the stored generation instead of merging with it.

No database or LLM is needed.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest

from stephanie.components.information.data import PaperSection
from stephanie.components.information.paper.pipeline.pipeline_runner import (
    PaperPipelineRunner,
)
from stephanie.components.information.paper.pipeline.sections_cache import (
    PaperSectionsCache,
)
from stephanie.components.information.tasks.section_build_task import (
    SectionBuildConfig,
    SectionBuildTask,
)
from stephanie.orm.paper import PaperSectionORM

ROOT = "2506.21734"
RELATED = "2505.16135"


class FakePaperStore:
    """Minimal in-memory stand-in for PaperStore's section surface."""

    def __init__(self) -> None:
        self.rows: dict[str, dict] = {}
        self.upserts: list[dict] = []
        self.replaced: list[str] = []

    # -- reads --
    def get_sections_for_paper(self, paper_id: str):
        return [
            SimpleNamespace(**row)
            for row in sorted(
                (r for r in self.rows.values() if r["paper_id"] == paper_id),
                key=lambda r: r["section_index"],
            )
        ]

    # -- writes --
    def upsert_section(self, data: dict):
        self.upserts.append(dict(data))
        self.rows[data["id"]] = dict(data)

    def replace_sections_for_paper(self, paper_id: str, sections: list[dict]):
        """Mirrors PaperSectionStore.replace_for_paper: delete everything this
        paper owns, plus rows whose id *claims* this paper (historical
        mis-attribution) -- but never another paper's own rows."""
        self.replaced.append(paper_id)
        prefix = f"{paper_id}::"
        for rid in [
            rid
            for rid, row in self.rows.items()
            if row["paper_id"] == paper_id or rid.startswith(prefix)
        ]:
            del self.rows[rid]
        for section in sections:
            self.rows[section["id"]] = dict(section)
        return list(sections)


class BoomStore(FakePaperStore):
    """A store whose writes always fail, for failure-visibility tests."""

    def upsert_section(self, data: dict):
        raise RuntimeError("simulated write failure")

    def replace_sections_for_paper(self, paper_id: str, sections: list[dict]):
        raise RuntimeError("simulated write failure")


def _cache(store: FakePaperStore) -> PaperSectionsCache:
    return PaperSectionsCache(cfg={}, paper_store=store, logger=None)


def _section(paper_id: str, index: int, text: str, **kwargs) -> PaperSection:
    return PaperSection(
        section_id=f"{paper_id}::sec-{index}",
        paper_arxiv_id=paper_id,
        role=kwargs.pop("role", "root" if paper_id == ROOT else "reference"),
        section_index=index,
        text=text,
        title=kwargs.pop("title", f"Title {index}"),
        summary=kwargs.pop("summary", "summary"),
        **kwargs,
    )


# ---------------------------------------------------------------- identity


def test_persist_uses_source_paper_identity_and_required_columns():
    store = FakePaperStore()
    cache = _cache(store)

    cache.persist(
        arxiv_id=ROOT,
        sections=[
            _section(ROOT, 0, "root zero"),
            _section(ROOT, 1, "root one"),
            _section(RELATED, 0, "related zero"),
        ],
    )

    assert set(store.rows) == {f"{ROOT}::sec-0", f"{ROOT}::sec-1", f"{RELATED}::sec-0"}

    # Ownership: each section belongs to its OWN paper, not the root.
    assert store.rows[f"{RELATED}::sec-0"]["paper_id"] == RELATED

    orm_columns = {c.name for c in PaperSectionORM.__table__.columns}
    required_not_null = {"id", "paper_id", "level", "path", "role", "section_index"}
    for row in store.rows.values():
        assert set(row) <= orm_columns, f"unknown ORM columns: {set(row) - orm_columns}"
        for col in required_not_null:
            assert row[col] is not None, f"{col} missing on {row['id']}"
        assert row["content_hash"]
        assert isinstance(row["token_count"], int)


def test_persist_groups_related_sections_under_their_own_paper():
    store = FakePaperStore()
    _cache(store).persist(
        arxiv_id=ROOT,
        sections=[_section(ROOT, 0, "a"), _section(RELATED, 0, "b")],
    )
    assert {r["paper_id"] for r in store.rows.values()} == {ROOT, RELATED}


def test_duplicate_section_index_is_dropped_not_collided():
    store = FakePaperStore()
    _cache(store).persist(
        arxiv_id=ROOT,
        sections=[_section(ROOT, 0, "first"), _section(ROOT, 0, "second")],
    )
    assert list(store.rows) == [f"{ROOT}::sec-0"]
    assert store.rows[f"{ROOT}::sec-0"]["text"] == "first"


# ------------------------------------------------------------- load/roundtrip


def test_round_trip_preserves_content_and_provenance():
    store = FakePaperStore()
    cache = _cache(store)
    cache.persist(
        arxiv_id=ROOT,
        sections=[
            _section(ROOT, 0, "root zero", title="Root Zero"),
            _section(RELATED, 0, "related zero", title="Related Zero"),
        ],
    )

    loaded = cache.maybe_load(ROOT)
    assert loaded is not None
    assert [s.section_id for s in loaded] == [f"{ROOT}::sec-0"]
    assert loaded[0].paper_arxiv_id == ROOT
    assert loaded[0].text == "root zero"
    assert loaded[0].title == "Root Zero"

    related = cache.maybe_load(RELATED)
    assert [s.section_id for s in related] == [f"{RELATED}::sec-0"]
    assert related[0].paper_arxiv_id == RELATED


def test_maybe_load_returns_none_when_empty():
    assert _cache(FakePaperStore()).maybe_load(ROOT) is None


# --------------------------------------------------------------- force rebuild


def test_force_rebuild_bypasses_the_cache():
    store = FakePaperStore()
    cache = _cache(store)
    cache.persist(arxiv_id=ROOT, sections=[_section(ROOT, 0, "old")])

    assert cache.maybe_load(ROOT) is not None
    assert cache.maybe_load(ROOT, force_rebuild=True) is None


def test_replace_removes_the_previous_generation():
    store = FakePaperStore()
    cache = _cache(store)

    cache.persist(arxiv_id=ROOT, sections=[_section(ROOT, 0, "old", title="Old cut-off")])
    cache.persist(
        arxiv_id=ROOT,
        sections=[_section(ROOT, 0, "new", title="New Complete Title")],
        replace=True,
    )

    assert store.replaced == [ROOT]
    assert len(store.rows) == 1
    titles = [r["title"] for r in store.rows.values()]
    assert titles == ["New Complete Title"]  # no stale generation survives


def test_replace_is_idempotent():
    store = FakePaperStore()
    cache = _cache(store)
    sections = [_section(ROOT, 0, "a"), _section(ROOT, 1, "b")]

    cache.persist(arxiv_id=ROOT, sections=sections, replace=True)
    first = dict(store.rows)
    cache.persist(arxiv_id=ROOT, sections=sections, replace=True)
    assert store.rows == first
    assert len(store.rows) == 2


def test_replace_does_not_touch_another_papers_rows():
    store = FakePaperStore()
    cache = _cache(store)

    # RELATED legitimately owns its own row ...
    cache.persist(arxiv_id=RELATED, sections=[_section(RELATED, 0, "related")])

    # ... and a ROOT rebuild must leave it alone.
    cache.persist(arxiv_id=ROOT, sections=[_section(ROOT, 0, "root")], replace=True)

    assert f"{RELATED}::sec-0" in store.rows
    assert store.rows[f"{RELATED}::sec-0"]["paper_id"] == RELATED
    assert store.rows[f"{RELATED}::sec-0"]["text"] == "related"


def test_replace_repairs_rows_misattributed_to_another_paper():
    """Historical shape: the id says RELATED but paper_id says ROOT. A RELATED
    rebuild must re-home (and therefore delete) that row."""
    store = FakePaperStore()
    store.rows[f"{RELATED}::sec-0"] = {
        "id": f"{RELATED}::sec-0",
        "paper_id": ROOT,
        "section_index": 0,
        "text": "stale",
    }

    _cache(store).persist(
        arxiv_id=RELATED, sections=[_section(RELATED, 0, "fresh")], replace=True
    )

    assert store.rows[f"{RELATED}::sec-0"]["paper_id"] == RELATED
    assert store.rows[f"{RELATED}::sec-0"]["text"] == "fresh"


def test_source_paper_comes_from_the_id_prefix_not_paper_id():
    """Live DB shape: `paper_id` is stamped with the ROOT paper for every row
    while the id prefix holds the true source. Trusting `paper_id` re-attributes
    and re-numbers every section (observed live: it rewrote `2505.05522::sec-3`
    as `2506.21734::sec-3`, which made the blog about the wrong paper)."""
    store = FakePaperStore()
    cache = _cache(store)
    sec = _section(RELATED, 0, "related text")
    sec.paper_arxiv_id = ROOT  # exactly what the paper_id column says

    cache.persist(arxiv_id=ROOT, sections=[sec], replace=True)

    assert list(store.rows) == [f"{RELATED}::sec-0"]
    assert store.rows[f"{RELATED}::sec-0"]["paper_id"] == RELATED


def test_loading_prefers_the_id_prefix_for_attribution():
    store = FakePaperStore()
    store.rows[f"{RELATED}::sec-0"] = {
        "id": f"{RELATED}::sec-0",
        "paper_id": ROOT,          # mis-attributed, as in the live DB
        "section_index": 0,
        "role": "reference",
        "text": "t",
        "title": "T",
    }
    loaded = _cache(store).maybe_load(ROOT)

    assert loaded is not None and len(loaded) == 1
    assert loaded[0].section_id == f"{RELATED}::sec-0"
    assert loaded[0].paper_arxiv_id == RELATED  # not ROOT


def test_shrinking_rebuild_removes_stale_orphans():
    """A rebuild that yields fewer sections must not leave higher-index rows
    behind for maybe_load() to serve."""
    store = FakePaperStore()
    cache = _cache(store)

    cache.persist(
        arxiv_id=ROOT,
        sections=[_section(ROOT, i, f"t{i}") for i in range(3)],
        replace=True,
    )
    assert len(store.rows) == 3

    cache.persist(arxiv_id=ROOT, sections=[_section(ROOT, 0, "only")], replace=True)
    assert list(store.rows) == [f"{ROOT}::sec-0"]
    assert [s.section_id for s in cache.maybe_load(ROOT)] == [f"{ROOT}::sec-0"]


# ------------------------------------------------- failure visibility (fix 2)


def test_persist_returns_written_count():
    store = FakePaperStore()
    written = _cache(store).persist(
        arxiv_id=ROOT, sections=[_section(ROOT, 0, "a"), _section(ROOT, 1, "b")]
    )
    assert written == 2


def test_persist_failure_is_visible_to_the_caller(caplog):
    cache = _cache(BoomStore())
    with caplog.at_level(logging.ERROR):
        written = cache.persist(
            arxiv_id=ROOT, sections=[_section(ROOT, 0, "a")], replace=True
        )

    assert written == 0
    messages = [r.getMessage() for r in caplog.records]
    assert any("persist FAILED" in m for m in messages)
    assert any("nothing persisted" in m for m in messages)


# ------------------------------------------------------- force-rebuild flag


def _runner_with_cfg(cfg, *, env=None, monkeypatch=None):
    runner = PaperPipelineRunner.__new__(PaperPipelineRunner)
    runner.cfg = cfg
    return runner


def test_force_rebuild_flag_from_context_wins(monkeypatch):
    runner = _runner_with_cfg({"force_rebuild_sections": False})
    assert PaperPipelineRunner._force_rebuild_requested(runner, {"force_rebuild_sections": True})
    assert not PaperPipelineRunner._force_rebuild_requested(runner, {"force_rebuild_sections": False})


def test_force_rebuild_flag_from_cfg_and_env(monkeypatch):
    runner = _runner_with_cfg({"force_rebuild_sections": True})
    assert PaperPipelineRunner._force_rebuild_requested(runner, {})

    runner = _runner_with_cfg({})
    monkeypatch.setenv("STEPHANIE_FORCE_REBUILD_SECTIONS", "true")
    assert PaperPipelineRunner._force_rebuild_requested(runner, {})

    monkeypatch.setenv("STEPHANIE_FORCE_REBUILD_SECTIONS", "false")
    assert not PaperPipelineRunner._force_rebuild_requested(runner, {})

    monkeypatch.delenv("STEPHANIE_FORCE_REBUILD_SECTIONS", raising=False)
    assert not PaperPipelineRunner._force_rebuild_requested(runner, {})


# -------------------------------------------------- section build regression


def test_section_build_task_can_build_only_missing_papers():
    """One graph node without cached sections must not force a re-parse of the
    whole graph (19 papers, ~700 sections, hours). `only_node_ids` restricts the
    build to the missing papers."""
    graph = SimpleNamespace(
        nodes={
            "AAAA": SimpleNamespace(role="root"),
            "BBBB": SimpleNamespace(role="reference"),
        }
    )
    texts = {"AAAA": "a" * 200, "BBBB": "b" * 200}

    async def summarizer(text: str):
        return ("T", "S")

    task = SectionBuildTask(
        cfg=SectionBuildConfig(chars_per_section=50, min_chars=1, overlap=5),
        summarizer=summarizer,
        embedder=lambda t: [0.0],
    )

    only_b = asyncio.run(task.run(graph=graph, texts=texts, only_node_ids={"BBBB"}))
    assert only_b and all(s.paper_arxiv_id == "BBBB" for s in only_b)

    both = asyncio.run(task.run(graph=graph, texts=texts))
    assert {s.paper_arxiv_id for s in both} == {"AAAA", "BBBB"}


def test_section_build_task_produces_valid_paper_sections():
    """Regression: the builder used ``PaperSection(id=..., paper_role=...)``,
    which are not this dataclass's fields, so a real rebuild would raise
    TypeError. This pins the corrected constructor kwargs."""

    async def summarizer(text: str):
        return ("A Title", "A summary")

    def embedder(text: str):
        return [0.0, 0.1, 0.2]

    task = SectionBuildTask(
        cfg=SectionBuildConfig(chars_per_section=50, min_chars=1, overlap=5),
        summarizer=summarizer,
        embedder=embedder,
    )
    sections = asyncio.run(task._build_from_chunks(ROOT, "root", "x" * 200))

    assert sections
    for sec in sections:
        assert sec.section_id.startswith(f"{ROOT}::sec-")
        assert sec.paper_arxiv_id == ROOT
        assert sec.role == "root"
