from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

import numpy as np
import pandas as pd

from stephanie.services.service_protocol import Service
from stephanie.utils.json_sanitize import dumps_safe

_logger = logging.getLogger(__name__)

class StorageService(Service):
    """
    Filesystem-backed storage for GAP runs.
    Conforms to the Service protocol so it can be registered in the ServiceContainer.
    """

    def __init__(self):
        # NOTE: do not access config here; container will call initialize(**kwargs)
        self._initialized = False
        self._base = None
        self.logger = None
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
            raise ValueError("StorageService.initialize: 'base_dir' is required")
        custom_logger = kwargs.get("logger")
        if custom_logger is not None:
            self.logger = custom_logger

        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)
        self._initialized = True
        _logger.info("StorageInit path: %s", self._base)

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
        _logger.debug("[StorageService] Shutdown complete")

    @property
    def name(self) -> str:
        return "gap-storage-v1"

    # --- Public API used by GAP pipeline -------------------------------------

    @property
    def base_dir(self) -> Path:
        if not self._initialized:
            raise RuntimeError("StorageService not initialized")
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
    def save_matrix(
        self,
        mat: np.ndarray,
        names: Sequence[str],
        run_id: str,
        *,
        tag: str,
        subdir: str | None = "aligned",
    ) -> Dict[str, Any]:
        mpath = self._resolve_path(run_id, f"{tag}_matrix.npy", subdir)
        npath = self._resolve_path(run_id, f"{tag}_metric_names.json", subdir)
        np.save(mpath, mat.astype(np.float32))
        self._write_json(npath, list(map(str, names)))
        return {"matrix": str(mpath), "names": str(npath), "shape": list(map(int, mat.shape))}

    def save_rows_df(
        self,
        rows_or_df: Union[pd.DataFrame, List[Dict]],
        run_id: str,
        *,
        name: str = "rows_for_df",
        subdir: str | None = "raw",
    ) -> Dict[str, str | None]:
        if isinstance(rows_or_df, pd.DataFrame):
            df = rows_or_df.copy()
        elif isinstance(rows_or_df, list):
            df = pd.DataFrame(rows_or_df)
        else:
            raise TypeError(f"save_rows_df expected DataFrame or List[Dict], got {type(rows_or_df)}")

        pq_path = self._resolve_path(run_id, f"{name}.parquet", subdir)
        csv_path = self._resolve_path(run_id, f"{name}.csv", subdir)

        parquet_ok: str | None = None
        try:
            df.to_parquet(pq_path, index=False)
            parquet_ok = str(pq_path)
        except Exception as e:
            _logger.warning("Parquet save failed (%s); continuing with CSV.", e)

        df.to_csv(csv_path, index=False)
        _logger.info("RowsPersisted | run_id=%s rows=%d csv=%s parquet=%s",
                        run_id, len(df), str(csv_path), bool(parquet_ok))
        self._mark_write()
        return {"parquet": parquet_ok, "csv": str(csv_path)}

    def save_json(
        self,
        run_id: str,
        *,
        subdir: str | None = None,
        name: str,
        obj: Dict[str, Any],
    ) -> str:
        path = self._resolve_path(run_id, name, subdir)
        self._write_json(path, obj)
        return str(path)

    def save_text(
        self,
        run_id: str,
        *,
        subdir: str | None = None,
        name: str,
        text: str,
    ) -> str:
        path = self._resolve_path(run_id, name, subdir)
        path.write_text(text, encoding="utf-8")
        self._mark_write()
        return str(path)

    def copy_into(
        self,
        src_path: str,
        run_id: str,
        *,
        subdir: str,
        name: str | None = None,
    ) -> str:
        import shutil
        dst = self._resolve_path(run_id, name or Path(src_path).name, subdir)
        try:
            shutil.copyfile(src_path, dst)
        except Exception as e:
            _logger.warning("copy_into failed (%s), returning intended dst anyway.", e)
        self._mark_write()
        return str(dst)

    # --- Internals ------------------------------------------------------------

    def _write_json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(dumps_safe(payload, indent=2), encoding="utf-8")
        tmp.replace(path)
        self._mark_write()

    def _mark_write(self) -> None:
        self._writes += 1
        self._last_write = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def _resolve_path(self, run_id: str, filename: str, subdir: str | None) -> Path:
        base = self.run_root(run_id)
        target = base
        if subdir:
            target = base.joinpath(*Path(subdir).parts)
        path = target / filename
        path.parent.mkdir(parents=True, exist_ok=True)

        # Guard: stay within run root
        if not str(path.resolve()).startswith(str(base.resolve())):
            raise ValueError(f"Refusing to write outside run root: {path}")
        return path