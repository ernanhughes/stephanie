from __future__ import annotations
from pathlib import Path
from typing import Any, Sequence, Union, Dict, List
import numpy as np
import pandas as pd
from .paths import ensure_dir
from .typing import NDArray

class RunStorageService:
    """Generic, FS-backed run storage (superset of your GapStorageService)."""

    def __init__(self, base_dir: Union[str, Path]):
        self._base = Path(base_dir); self._base.mkdir(parents=True, exist_ok=True)

    @property
    def base_dir(self) -> Path:
        return self._base

    def run_root(self, run_id: str) -> Path:
        p = self._base / run_id; p.mkdir(parents=True, exist_ok=True); return p

    def subdir(self, run_id: str, sub: str) -> Path:
        p = self.run_root(run_id) / sub; p.mkdir(parents=True, exist_ok=True); return p

    # JSON/Text
    def save_json(self, run_id: str, sub: str, filename: str, payload: Any) -> str:
        import json
        p = self.subdir(run_id, sub) / filename
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(p)

    def save_text(self, run_id: str, sub: str, filename: str, text: str) -> str:
        p = self.subdir(run_id, sub) / filename
        p.write_text(text, encoding="utf-8")
        return str(p)

    # Arrays + Names
    def save_matrix(self, mat: NDArray, names: Sequence[str], run_id: str, *, tag: str) -> Dict[str, Any]:
        d = self.subdir(run_id, "aligned")
        mpath = d / f"{tag}_matrix.npy"
        npath = d / f"{tag}_metric_names.json"
        np.save(mpath, np.asarray(mat, dtype=np.float32))
        self.save_json(run_id, "aligned", f"{tag}_metric_names.json", list(map(str, names)))
        return {"matrix": str(mpath), "names": str(npath), "shape": list(map(int, mat.shape))}

    # DataFrames
    def save_rows_df(self, rows_or_df: Union[pd.DataFrame, List[Dict]], run_id: str, name: str="rows") -> Dict[str, Any]:
        d = self.subdir(run_id, "raw")
        if isinstance(rows_or_df, pd.DataFrame):
            df = rows_or_df.copy()
        else:
            df = pd.DataFrame(rows_or_df)
        pq = d / f"{name}.parquet"; csv = d / f"{name}.csv"
        parquet_ok = None
        try:
            df.to_parquet(pq, index=False); parquet_ok = str(pq)
        except Exception:
            pass
        df.to_csv(csv, index=False)
        return {"parquet": parquet_ok, "csv": str(csv)}
