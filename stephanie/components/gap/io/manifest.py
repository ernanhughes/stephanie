from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional
import time

from stephanie.components.gap.io.storage import GapStorageService


@dataclass
class GapRunManifest:
    run_id: str
    dataset: str
    models: dict
    preproc_version: str = "v1"
    created_at: float = field(default_factory=lambda: time.time())


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
