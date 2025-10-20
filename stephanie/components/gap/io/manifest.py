from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

from stephanie.components.gap.io.storage import GapStorageService

Timestamp = str

def _now_iso() -> Timestamp:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class GapRunManifest:
    run_id: str
    dataset: str
    models: Dict[str, Any] = field(default_factory=dict)
    created_at: Timestamp = field(default_factory=_now_iso)
    meta: Dict[str, Any] = field(default_factory=dict)

    # All stage records live here:
    #   stages[stage_name] = {
    #       "stage": str, "status": "running|ok|error|cancelled",
    #       "started_at": ts, "finished_at": ts?, "done": int?, "total": int?,
    #       ... any user fields ...
    #   }
    stages: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # ---------- Stage helpers ----------
    def stage_start(self, name: str, *, total: Optional[int] = None, **fields) -> None:
        rec = {
            "stage": name,
            "status": "running",
            "started_at": _now_iso(),
        }
        if total is not None:
            rec["total"] = int(total)
            rec["done"] = 0
        if fields:
            rec.update(fields)
        self.stages[name] = rec

    def stage_tick(self, name: str, *, done: int, total: Optional[int] = None, **fields) -> None:
        rec = self.stages.get(name) or {"stage": name, "status": "running", "started_at": _now_iso()}
        rec["done"] = int(done)
        if total is not None:
            rec["total"] = int(total)
        if fields:
            rec.update(fields)
        self.stages[name] = rec

    def stage_update(self, name: str, **fields) -> None:
        rec = self.stages.get(name) or {"stage": name, "status": "running", "started_at": _now_iso()}
        if fields:
            rec.update(fields)
        self.stages[name] = rec

    def stage_end(self, name: str, *, status: str = "ok", **fields) -> None:
        rec = self.stages.get(name) or {"stage": name, "started_at": _now_iso()}
        rec["status"] = status
        rec["finished_at"] = _now_iso()
        if fields:
            rec.update(fields)
        # If total exists but done not set, clamp to total
        if "total" in rec and "done" not in rec:
            rec["done"] = rec["total"]
        self.stages[name] = rec

    def get_stage(self, name: str) -> Dict[str, Any]:
        return self.stages.get(name, {})

    # ---------- Serialization ----------
    def to_dict(self) -> Dict[str, Any]:
        # deep copy to keep callers from mutating internals
        return {
            "run_id": self.run_id,
            "dataset": self.dataset,
            "models": copy.deepcopy(self.models),
            "created_at": self.created_at,
            "meta": copy.deepcopy(self.meta),
            "stages": copy.deepcopy(self.stages),
        }

    def setdefault(self, key: str, default: Any) -> Any:
        if key == "stages":
            if not self.stages:
                self.stages = {}
            return self.stages
        # only support "stages" to avoid surprising behavior
        raise AttributeError("GapRunManifest only supports setdefault('stages', ...)")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GapRunManifest":
        return cls(
            run_id=d.get("run_id", ""),
            dataset=d.get("dataset", ""),
            models=copy.deepcopy(d.get("models", {})),
            created_at=d.get("created_at", _now_iso()),
            meta=copy.deepcopy(d.get("meta", {})),
            stages=copy.deepcopy(d.get("stages", {})),
        )
    

class ManifestManager:
    def __init__(self, storage: GapStorageService):
        self.storage = storage

    def start_run(self, run_id: str, dataset: str, models: dict) -> GapRunManifest:
        m = GapRunManifest(run_id=run_id, dataset=dataset, models=models)
        self.storage.write_manifest(run_id, asdict(m))
        return m

    def attach_dimensions(self, run_id: str, dims):
        self.storage.patch_manifest(run_id, {"dimensions": list(dims)})

    def finish_run(self, run_id: str, final_payload: Dict[str, Any]):
        self.storage.patch_manifest(run_id, {"final": final_payload})
