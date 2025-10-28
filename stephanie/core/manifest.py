from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
from .time import now_iso

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

class ManifestManager:
    """Tiny, storage-agnostic manifest manager."""
    def __init__(self, storage):
        self.storage = storage

    def start_run(self, *, run_id: str, dataset: str | None, models: Dict[str, str]) -> Manifest:
        m = Manifest(run_id=run_id, dataset=dataset, models=models)
        self._save(m)
        return m

    def attach_dimensions(self, run_id: str, dims: list[str]) -> None:
        m = self._load(run_id)
        m.extras.setdefault("dimensions", dims)
        self._save(m)

    def add_stage(self, run_id: str, name: str, payload: Dict[str, Any]) -> None:
        m = self._load(run_id)
        m.stages.append({"name": name, "time": now_iso(), **payload})
        self._save(m)

    def finish_run(self, run_id: str, result: Dict[str, Any]) -> None:
        m = self._load(run_id)
        m.finished_at = now_iso()
        m.extras["result_keys"] = list(result.keys())
        self._save(m)

    # --- internals
    def _save(self, m: Manifest) -> None:
        self.storage.save_json(m.run_id, "results", "manifest.json", m.to_dict())

    def _load(self, run_id: str) -> Manifest:
        # simple read-back; tolerate missing to keep it lightweight
        try:
            import json
            p = self.storage.subdir(run_id, "results") / "manifest.json"
            if p.exists():
                return Manifest(**json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            pass
        return Manifest(run_id=run_id)
