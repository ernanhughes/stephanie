"""Regression test for PipelineRunORM.to_dict() serialization.

The home page lists pipeline runs, so a serialization error there takes down
``GET /``. This locks the created_at handling (which previously called the
zero-argument ``iso_now`` with an argument).
"""

from __future__ import annotations

from datetime import datetime, timezone

# Import the normal store registry so all ORM mappers resolve before use.
from stephanie.memory.memory_tool import MemoryTool as _RegisterMemoryStores  # noqa: F401
from stephanie.orm.pipeline_run import PipelineRunORM


def test_pipeline_run_to_dict_handles_missing_and_present_created_at():
    row = PipelineRunORM(run_id="run-fixture", goal_id=1, name="fixture")
    data = row.to_dict()
    assert data["run_id"] == "run-fixture"
    assert data["name"] == "fixture"
    assert data["created_at"] is None  # not persisted yet

    stamp = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    row.created_at = stamp
    assert row.to_dict()["created_at"] == stamp.isoformat()
