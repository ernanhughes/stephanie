# stephanie/core/manifest.py
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from stephanie.utils.json_sanitize import dumps_safe
from stephanie.utils.time_utils import now_iso


@dataclass
class Manifest:
    run_id: str
    dataset: str | None = None
    models: Dict[str, str] = field(default_factory=dict)
    stages: List[Dict[str, Any]] = field(default_factory=list)
    started_at: str = field(default_factory=now_iso)
    finished_at: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def add_stage(self, name: str, **payload: Any) -> Dict[str, Any]:
        return self.stage_start(name, **payload)

    def stage_start(self, name: str, **payload: Any) -> Dict[str, Any]:
        stage: Dict[str, Any] = {
            "name": name,
            "started_at": now_iso(),
            "status": payload.pop("status", "running"),
            **payload,
        }
        self.stages.append(stage)
        return stage

    def stage_tick(self, name: str, **updates: Any) -> Dict[str, Any]:
        stage = self._find_stage(name)
        if stage is None:
            stage = self.stage_start(name)
        stage.update(updates)
        return stage

    def stage_end(self, name: str, **payload: Any) -> Dict[str, Any]:
        stage = self._find_stage(name)
        if stage is None:
            stage = self.stage_start(name)
        status = payload.pop("status", "ok")
        stage.update(payload)
        stage["status"] = status
        stage["finished_at"] = now_iso()
        return stage

    def get_stage(self, name: str) -> Optional[Dict[str, Any]]:
        return self._find_stage(name)

    def _find_stage(self, name: str) -> Optional[Dict[str, Any]]:
        for st in reversed(self.stages):
            if st.get("name") == name:
                return st
        return None

    def log_vpm_artifacts(self, vpms: List[np.ndarray], metas: List[dict]):
        for vpm, meta in zip(vpms, metas):
            layout_id = f"{meta['layout']}_{meta['seed']}"
            self.extras.setdefault("artifacts", {})[f"vpm_{layout_id}"] = {
                "shape": getattr(vpm, "shape", None),
                "dtype": str(getattr(vpm, "dtype", "")),
                "meta": meta,
            }


class ManifestManager:
    """
    Filesystem-backed, storage-agnostic manifest manager.
    Keeps all files under base_root/<run_id>/...
    """
    def __init__(self, base_root: str = "data/gap_runs"):
        self.base_root = Path(base_root)

    # --- path helpers
    def run_dir(self, run_id: str) -> Path:
        return self.base_root / run_id

    def subdir(self, run_id: str, name: str) -> Path:
        return self.run_dir(run_id) / name

    def manifest_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "manifest.json"

    def features_jsonl_path(self, run_id: str, name: str = "features.jsonl") -> Path:
        # default under raw/ to mirror your layout
        return self.subdir(run_id, "raw") / name

    # --- public API
    def start_run(self, *, run_id: str, dataset: str | None, models: Dict[str, str]) -> Manifest:
        """
        Creates:
          <base>/<run_id>/
            raw/ aligned/ visuals/ metrics/ reports/
            manifest.json
        """
        base = self.run_dir(run_id)
        for sub in ("raw", "aligned", "visuals", "metrics", "reports", "results"):
            (base / sub).mkdir(parents=True, exist_ok=True)

        m = Manifest(run_id=run_id, dataset=dataset, models=models)
        self.save_manifest(m)
        return m

    def attach_dimensions(self, run_id: str, dims: list[str]) -> None:
        m = self.load_manifest(run_id)
        m.extras.setdefault("dimensions", dims)
        self.save_manifest(m)

    def add_stage(self, run_id: str, name: str, payload: Dict[str, Any]) -> None:
        m = self.load_manifest(run_id)
        m.stage_start(name, **payload)
        self.save_manifest(m)

    def finish_run(self, run_id: str, result: Dict[str, Any]) -> None:
        m = self.load_manifest(run_id)
        m.finished_at = now_iso()
        m.extras["result_keys"] = list(result.keys())
        self.save_manifest(m)

    # --- internals
    def save_manifest(self, m: Manifest) -> None:
        p = self.manifest_path(m.run_id)
        p.write_text(dumps_safe(m.to_dict(), indent=2), encoding="utf-8")

    def load_manifest(self, run_id: str) -> Manifest:
        p = self.manifest_path(run_id)
        if p.exists():
            return Manifest(**json.loads(p.read_text(encoding="utf-8")))
        return Manifest(run_id=run_id)
