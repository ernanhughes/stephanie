# scripts/build_risk_dataset.py
from __future__ import annotations
import json, os
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd

def _read(path: Path):
    if not path.exists():
        return None
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if path.suffix == ".npz":
        return dict(np.load(path, allow_pickle=True))
    return None

def build_from_run(run_dir: Path, out_parquet: Path, label_energy_thresh: float = 0.55):
    raw_dir = run_dir / "raw"
    metrics_dir = run_dir / "metrics"
    rows_path = raw_dir / "rows_for_df.parquet"
    prov_path = raw_dir / "row_provenance.json"

    if not rows_path.exists():
        raise FileNotFoundError(f"Missing {rows_path}")
    rows_df = pd.read_parquet(rows_path)

    prov = _read(prov_path) or []
    prov_df = pd.DataFrame(prov)

    # Build (row_index -> node_id, dimension, texts) mapping if available
    map_cols = [c for c in ["row_index","node_id","dimension","goal_text","output_text"] if c in prov_df.columns]
    prov_map_df = prov_df[map_cols].copy() if map_cols else pd.DataFrame(columns=["row_index"])

    # Start with rows_df and LEFT-JOIN provenance by index->row_index
    df = rows_df.merge(
        prov_map_df[["row_index","node_id","dimension"]] if {"row_index","node_id","dimension"}.issubset(prov_map_df.columns) else
        prov_map_df[["row_index","node_id"]] if {"row_index","node_id"}.issubset(prov_map_df.columns) else
        pd.DataFrame({"row_index": rows_df.index}),
        left_index=True, right_on="row_index", how="left"
    )

    # Ensure node_id exists; if not, synthesize stable ids
    if "node_id" not in df.columns:
        df["node_id"] = [f"row_{i}" for i in range(len(df))]

    # Optional: attach domain from dimension if present
    if "dimension" in df.columns:
        df["domain"] = df["dimension"].fillna("general")
    else:
        df["domain"] = "general"

    # Attach EG per-turn metadata if you saved any *_meta.json in /visuals (best-effort)
    eg_dir = run_dir / "visuals"
    eg_meta = []
    if eg_dir.exists():
        for f in eg_dir.glob("*_meta.json"):
            try:
                eg_meta.append(json.loads(f.read_text(encoding="utf-8")))
            except Exception:
                pass
    eg_df = pd.DataFrame(eg_meta) if eg_meta else pd.DataFrame(columns=["node_id"])

    if not eg_df.empty and "node_id" in eg_df.columns:
        df = df.merge(eg_df, on="node_id", how="left")

    # Build deltas between the two model aliases if they exist
    model_cols = [c for c in df.columns if "." in c and c not in ("node_id","model_alias")]
    roots = sorted(set(c.split(".", 1)[0] for c in model_cols))
    if len(roots) >= 2:
        a, b = roots[:2]
        dims = sorted(set(c.split(".",1)[1] for c in model_cols if c.startswith(a+".")))
        for d in dims:
            ca, cb = f"{a}.{d}", f"{b}.{d}"
            if ca in df.columns and cb in df.columns:
                df[f"delta.{d}"] = df[ca].astype(float) - df[cb].astype(float)

    # Label: prefer EG max_energy if present
    if "labels.max_energy" in df.columns:
        df["y"] = (pd.to_numeric(df["labels.max_energy"], errors="coerce").fillna(0.0) >= label_energy_thresh).astype(int)
    elif "max_energy" in df.columns:
        df["y"] = (pd.to_numeric(df["max_energy"], errors="coerce").fillna(0.0) >= label_energy_thresh).astype(int)
    else:
        df["y"] = 0

    # Attach texts if missing, using node_id merge (guarded)
    need_goal = "goal_text" not in df.columns or df["goal_text"].isna().all()
    need_out  = "output_text" not in df.columns or df["output_text"].isna().all()
    if (need_goal or need_out) and {"node_id","goal_text","output_text"}.issubset(prov_df.columns):
        df = df.merge(prov_df[["node_id","goal_text","output_text"]], on="node_id", how="left", suffixes=("","_prov"))

        # prefer non-null
        if "goal_text_prov" in df.columns:
            df["goal_text"] = df["goal_text"].fillna(df["goal_text_prov"])
        if "output_text_prov" in df.columns:
            df["output_text"] = df["output_text"].fillna(df["output_text_prov"])
        # cleanup
        for c in ("goal_text_prov","output_text_prov"):
            if c in df.columns: df.drop(columns=[c], inplace=True)

    # Basic text length features
    df["len_goal"] = df["goal_text"].fillna("").map(len) if "goal_text" in df.columns else 0
    df["len_out"]  = df["output_text"].fillna("").map(len) if "output_text" in df.columns else 0

    # Keep only meaningful columns
    keep = ["node_id","domain","y","len_goal","len_out"] + \
           [c for c in df.columns if c.startswith("delta.") or c.startswith("scm.")]
    out_df = df.loc[:, [c for c in keep if c in df.columns]].copy()

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_parquet)
    print(f"Wrote {out_parquet} with {len(out_df)} rows, {out_df.shape[1]} columns.")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="GAP run directory (contains raw/, metrics/, reports/)")
    ap.add_argument("--out", default="./reports/risk_dataset.parquet")
    ap.add_argument("--energy_thresh", type=float, default=0.55)
    args = ap.parse_args()
    build_from_run(Path(args.run_dir), Path(args.out), label_energy_thresh=args.energy_thresh)
