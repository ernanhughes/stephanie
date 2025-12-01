# stephanie/services/manifest_service.py
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from typing import Any, Dict

from stephanie.core.manifest import Manifest, ManifestManager
from stephanie.utils.json_sanitize import dumps_safe


def make_run_id(prefix: str | None = None) -> str:
    """
    Generates a globally unique run ID.
    Example: 20250203_142210_9f5ac3 or critic_20250203_142210_a82f91
    """
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    rnd = uuid.uuid4().hex[:6]
    if prefix:
        return f"{prefix}_{ts}_{rnd}"
    return f"{ts}_{rnd}"


class ManifestService:
    """
    Multi-run manifest controller.
    - start() returns a unique run_id
    - append_row(run_id, row)
    - finish(run_id)
    """

    def __init__(self, base_root: str = "data/ssp_runs"):
        self.manager = ManifestManager(base_root=base_root)

        # Active runs: run_id → Manifest + features_path
        self._runs: Dict[str, Dict[str, Any]] = {}

        # Lock per run to avoid concurrent writes
        self._locks: Dict[str, asyncio.Lock] = {}

    # --------------------------------------------------------------
    def start(
        self,
        *,
        dataset: str | None = None,
        models: Dict[str, str] | None = None,
        prefix: str | None = None,
    ) -> str:
        """
        Create a NEW manifest folder and return a unique run_id.
        """
        run_id = make_run_id(prefix)

        manifest = self.manager.start_run(
            run_id=run_id,
            dataset=dataset,
            models=models or {},
        )

        features_path = self.manager.features_jsonl_path(run_id)
        features_path.parent.mkdir(parents=True, exist_ok=True)
        if not features_path.exists():
            features_path.write_text("", encoding="utf-8")

        # Store session info
        self._runs[run_id] = {
            "manifest": manifest,
            "features_path": features_path,
        }
        self._locks[run_id] = asyncio.Lock()

        return run_id

    # --------------------------------------------------------------
    async def append_row(self, run_id: str, row: Dict[str, Any]):
        run = self._runs.get(run_id)
        if not run:
            raise RuntimeError(f"ManifestService: unknown run_id={run_id}")

        path = run["features_path"]
        lock = self._locks[run_id]

        async with lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(dumps_safe(row) + "\n")

    # --------------------------------------------------------------
    def add_stage(self, run_id: str, name: str, **payload):
        run = self._runs.get(run_id)
        if not run:
            raise RuntimeError(f"ManifestService: unknown run_id={run_id}")

        mf: Manifest = run["manifest"]
        mf.stage_start(name, **payload)
        self.manager.save_manifest(mf)

    # --------------------------------------------------------------
    def end_stage(self, run_id: str, name: str, **payload):
        run = self._runs.get(run_id)
        if not run:
            raise RuntimeError(f"ManifestService: unknown run_id={run_id}")

        mf: Manifest = run["manifest"]
        mf.stage_end(name, **payload)
        self.manager.save_manifest(mf)

    # --------------------------------------------------------------
    def finish(self, run_id: str, result: Dict[str, Any]):
        run = self._runs.get(run_id)
        if not run:
            raise RuntimeError(f"ManifestService: unknown run_id={run_id}")

        self.manager.finish_run(run_id, result)

        # Optional: clean memory
        del self._runs[run_id]
        del self._locks[run_id]
