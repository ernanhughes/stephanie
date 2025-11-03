# stephanie/core/manifest.py
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from stephanie.tools.time_utils import now_iso
from stephanie.utils.json_sanitize import dumps_safe


@dataclass
class Manifest:
    run_id: str
    dataset: str | None = None
    models: Dict[str, str] = field(default_factory=dict)
    stages: List[Dict[str, Any]] = field(default_factory=list)  # list of stage dicts
    started_at: str = field(default_factory=now_iso)
    finished_at: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    # ---------- serialization ----------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    # ---------- stage helpers ----------
    def add_stage(self, name: str, **payload: Any) -> Dict[str, Any]:
        """Alias for stage_start for backward compatibility."""
        return self.stage_start(name, **payload)

    def stage_start(self, name: str, **payload: Any) -> Dict[str, Any]:
        """Append a new stage entry (status defaults to 'running')."""
        stage: Dict[str, Any] = {
            "name": name,
            "started_at": now_iso(),
            "status": payload.pop("status", "running"),
            **payload,
        }
        self.stages.append(stage)
        return stage

    def stage_tick(self, name: str, **updates: Any) -> Dict[str, Any]:
        """
        Update fields on the most recent stage with this name.
        Useful for progress (e.g., count, total), arbitrary metadata, etc.
        """
        stage = self._find_stage(name)
        if stage is None:
            stage = self.stage_start(name)
        stage.update(updates)
        return stage

    def stage_end(self, name: str, **payload: Any) -> Dict[str, Any]:
        """
        Mark a stage finished. If the stage doesn't exist yet,
        it will be created first.
        """
        stage = self._find_stage(name)
        if stage is None:
            stage = self.stage_start(name)
        # allow caller to override status; default to 'ok'
        status = payload.pop("status", "ok")
        stage.update(payload)
        stage["status"] = status
        stage["finished_at"] = now_iso()
        return stage

    def get_stage(self, name: str) -> Optional[Dict[str, Any]]:
        """Return the most recent stage dict with this name (or None)."""
        return self._find_stage(name)

    # ---------- internals ----------
    def _find_stage(self, name: str) -> Optional[Dict[str, Any]]:
        # search from the end to get the most recent occurrence
        for st in reversed(self.stages):
            if st.get("name") == name:
                return st
        return None


class ManifestManager:
    """Tiny, storage-agnostic manifest manager."""
    def __init__(self, storage):
        self.storage = storage

    def start_run(self, *, run_id: str, dataset: str | None, models: Dict[str, str], base_root: str = "data/gap_runs") -> Manifest:
        """
        Create a reproducible run folder structure and a manifest.json.

        Layout:
        gap_runs/<run_id>/
            raw/ aligned/ visuals/ metrics/ reports/
            manifest.json
        """
        base = os.path.join(base_root, run_id)
        for sub in ("raw", "aligned", "visuals", "metrics", "reports"):
            os.makedirs(os.path.join(base, sub), exist_ok=True)
        m = Manifest(run_id=run_id, dataset=dataset, models=models)
        with open(os.path.join(base, "manifest.json"), "w", encoding="utf-8") as f:
            f.write(dumps_safe(asdict(m), indent=2))
        return m


    def attach_dimensions(self, run_id: str, dims: list[str]) -> None:
        m = self._load(run_id)
        m.extras.setdefault("dimensions", dims)
        self._save(m)

    def add_stage(self, run_id: str, name: str, payload: Dict[str, Any]) -> None:
        m = self._load(run_id)
        m.stage_start(name, **payload)
        self._save(m)

    def finish_run(self, run_id: str, result: Dict[str, Any]) -> None:
        m = self._load(run_id)
        m.finished_at = now_iso()
        m.extras["result_keys"] = list(result.keys())
        self._save(m)

    # --- internals
    def _save(self, m: Manifest) -> None:
        self.storage.save_json(
            m.run_id,
            subdir="results",
            name="manifest.json",
            obj=m.to_dict(),
        )

    def _load(self, run_id: str) -> Manifest:
        # simple read-back; tolerate missing to keep it lightweight
        try:
            import json
            p = self.storage.subdir(run_id, ".") / "manifest.json"
            if p.exists():
                return Manifest(**json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            pass
        return Manifest(run_id=run_id)
