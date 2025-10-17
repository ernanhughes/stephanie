from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from stephanie.services.service_protocol import Service
from stephanie.utils.json_sanitize import dumps_safe

import logging
_logger = logging.getLogger(__name__)

class GapStorageService(Service):
    """
    Filesystem-backed storage for GAP runs.
    Conforms to the Service protocol so it can be registered in the ServiceContainer.
    """

    def __init__(self):
        # NOTE: do not access config here; container will call initialize(**kwargs)
        self._initialized = False
        self._base = None
        self.logger = logging.getLogger("gap.storage")
        self._writes = 0
        self._last_write = None

    # --- Service protocol -----------------------------------------------------

    def initialize(self, **kwargs) -> None:
        """
        Expected kwargs:
          - base_dir: str | Path (required)
          - logger: optional custom logger
        """
        base_dir = kwargs.get("base_dir")
        if not base_dir:
            raise ValueError("GapStorageService.initialize: 'base_dir' is required")
        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)
        self._initialized = True
        _logger.info(f"GapStorageInit path: {self._base}")

    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy" if self._initialized else "uninitialized",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "metrics": {
                "base_dir": str(self._base) if self._base else None,
                "writes": self._writes,
                "last_write": self._last_write,
            },
        }

    def shutdown(self) -> None:
        # Nothing persistent to close; keep for symmetry/logging
        _logger.debug("[GapStorage] Shutdown complete")

    @property
    def name(self) -> str:
        return "gap-storage-v1"

    # --- Public API used by GAP pipeline -------------------------------------

    @property
    def base_dir(self) -> Path:
        if not self._initialized:
            raise RuntimeError("GapStorageService not initialized")
        return self._base

    # Directory helpers
    def run_root(self, run_id: str) -> Path:
        root = self.base_dir / run_id
        root.mkdir(parents=True, exist_ok=True)
        return root

    def subdir(self, run_id: str, sub: str) -> Path:
        p = self.run_root(run_id) / sub
        p.mkdir(parents=True, exist_ok=True)
        return p

    # Manifest utilities
    def write_manifest(self, run_id: str, payload: Dict[str, Any]) -> Path:
        path = self.run_root(run_id) / "manifest.json"
        self._write_json(path, payload)
        return path

    def patch_manifest(self, run_id: str, patch: Dict[str, Any]) -> Path:
        path = self.run_root(run_id) / "manifest.json"
        try:
            cur = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            cur = {}
        # shallow/deep-ish merge
        for k, v in patch.items():
            if isinstance(v, dict) and isinstance(cur.get(k), dict):
                cur[k].update(v)
            else:
                cur[k] = v
        self._write_json(path, cur)
        return path

    # Matrix + names (aligned)
    def save_matrix(self, mat: np.ndarray, names: Sequence[str], run_id: str, *, tag: str) -> Dict[str, Any]:
        aligned = self.subdir(run_id, "aligned")
        mpath = aligned / f"{tag}_matrix.npy"
        npath = aligned / f"{tag}_metric_names.json"
        np.save(mpath, mat.astype(np.float32))
        self._write_json(npath, list(map(str, names)))
        return {"matrix": str(mpath), "names": str(npath), "shape": list(map(int, mat.shape))}

    # Rows for PHOS (both parquet & csv for portability)
    def save_rows_df(self, df: pd.DataFrame, run_id: str, *, name: str = "rows_for_df") -> Dict[str, str]:
        raw = self.subdir(run_id, "raw")
        pq = raw / f"{name}.parquet"
        csv = raw / f"{name}.csv"
        try:
            df.to_parquet(pq, index=False)
        except Exception:
            # parquet optional
            pass
        df.to_csv(csv, index=False)
        self._mark_write()
        return {"parquet": str(pq), "csv": str(csv)}

    # JSON helpers
    def save_json(self, run_id: str, sub: str, filename: str, payload: Dict[str, Any]) -> str:
        p = self.subdir(run_id, sub) / filename
        self._write_json(p, payload)
        return str(p)

    def save_text(self, run_id: str, sub: str, filename: str, text: str) -> str:
        p = self.subdir(run_id, sub) / filename
        p.write_text(text, encoding="utf-8")
        self._mark_write()
        return str(p)

    def copy_into(self, src_path: str, run_id: str, sub: str, dst_name: str | None = None) -> str:
        import shutil
        dst = self.subdir(run_id, sub) / (dst_name or Path(src_path).name)
        try:
            shutil.copyfile(src_path, dst)
        except Exception:
            # still return the intended destination
            pass
        self._mark_write()
        return str(dst)

    # --- Internals ------------------------------------------------------------

    def _write_json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dumps_safe(payload, indent=2), encoding="utf-8")
        self._mark_write()

    def _mark_write(self) -> None:
        self._writes += 1
        self._last_write = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
