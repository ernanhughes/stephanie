# npz_to_csv_backfill.py
# Backfill baseline/targeted CSVs for historical VisiCalc runs using
# headers (metric columns) from a reference run's CSV.
#
# It expects each historical run to contain `visicalc_ab_dataset.npz`
# with arrays: X (N x D), y (N,) where 0=baseline, 1=targeted.
# If the NPZ also includes item IDs (keys: item_ids, item_ids_base, item_ids_tgt),
# they'll be used. Otherwise, we try to pull IDs from JSON sidecars or synthesize them.
#
# Usage examples:
#   python npz_to_csv_backfill.py --runs_dir runs/visicalc --ref_run 909
#   python npz_to_csv_backfill.py --runs_dir runs/visicalc --ref_csv runs/visicalc/909/visicalc_targeted_matrix.csv
#
import argparse
import csv
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np


def load_header_from_csv(csv_path: Path) -> List[str]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        header = next(r)
    if not header or header[0] != "scorable_id":
        raise ValueError(f"Reference CSV {csv_path} missing 'scorable_id' header")
    return header[1:]  # metric columns only


def find_reference_csvs(ref_run_dir: Path) -> Path:
    """
    Prefer targeted CSV for header source; baseline has same header.
    """
    tgt = ref_run_dir / "visicalc_targeted_matrix.csv"
    base = ref_run_dir / "visicalc_baseline_matrix.csv"
    if tgt.exists():
        return tgt
    if base.exists():
        return base
    raise FileNotFoundError(f"No reference CSV found in {ref_run_dir}")


def load_ids_from_json_sidecars(run_dir: Path, which: str) -> Optional[List[str]]:
    """
    Try to extract item_ids for 'baseline' or 'targeted' from JSON files in run_dir.
    """
    if which == "baseline":
        candidates = [run_dir / "visicalc_baseline.json", run_dir / "visicalc_report.json"]
    else:
        candidates = [run_dir / "visicalc_targeted.json", run_dir / "visicalc_report.json"]

    def _extract_ids(j: Dict[str, Any]) -> Optional[List[str]]:
        # Common places to look
        for key in ("item_ids", "rows", "scorable_ids"):
            v = j.get(key)
            if isinstance(v, list) and v and isinstance(v[0], (str, int)):
                return [str(x) for x in v]
        # Rows with dicts
        if isinstance(j.get("rows"), list):
            maybe = []
            for r in j["rows"]:
                if isinstance(r, dict) and "scorable_id" in r:
                    maybe.append(str(r["scorable_id"]))
            if maybe:
                return maybe
        # Nested objects
        for nest in ("report", "data", "visicalc", "payload"):
            if isinstance(j.get(nest), dict):
                got = _extract_ids(j[nest])
                if got:
                    return got
        return None

    for p in candidates:
        if p.exists():
            try:
                j = json.loads(p.read_text(encoding="utf-8"))
                ids = _extract_ids(j)
                if ids:
                    return ids
            except Exception:
                continue
    return None


def save_matrix_csv(matrix: np.ndarray, metric_names: List[str], item_ids: List[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scorable_id", *metric_names])
        for ridx in range(matrix.shape[0]):
            w.writerow([item_ids[ridx], *[float(x) for x in matrix[ridx]]])


def build_csvs_from_npz(run_dir: Path, ref_metric_names: List[str]) -> bool:
    npz_path = run_dir / "visicalc_ab_dataset.npz"
    if not npz_path.exists():
        return False

    out_tgt = run_dir / "visicalc_targeted_matrix.csv"
    out_base = run_dir / "visicalc_baseline_matrix.csv"
    # If both exist, nothing to do.
    if out_tgt.exists() and out_base.exists():
        return True

    data = np.load(npz_path, allow_pickle=True)
    X = data.get("X")
    y = data.get("y")
    if X is None or y is None:
        raise ValueError(f"{npz_path} missing X/y arrays")

    # Determine metric order/size
    D_ref = len(ref_metric_names)
    D_x = X.shape[1]
    if D_x != D_ref:
        # Try to use metric_names from NPZ (if present) to align
        names_npz = data.get("metric_names")
        if names_npz is not None:
            names_npz = list(names_npz.tolist())
            # Build a reindex map from ref order -> npz order
            pos = {name: idx for idx, name in enumerate(names_npz)}
            missing = [n for n in ref_metric_names if n not in pos]
            if missing:
                raise ValueError(f"{run_dir.name}: NPZ missing columns: {missing[:5]} (and possibly more)")
            cols = [pos[n] for n in ref_metric_names]
            X = X[:, cols]
            D_x = X.shape[1]
        else:
            raise ValueError(f"{run_dir.name}: X dims {D_x} do not match ref {D_ref} and NPZ lacks metric_names")

    # Split rows by label
    base_rows = X[y == 0]
    tgt_rows = X[y == 1]

    # Attempt to recover item ids
    ids_all = data.get("item_ids")
    ids_base = data.get("item_ids_base")
    ids_tgt = data.get("item_ids_tgt")

    if ids_base is None or ids_tgt is None:
        # Try JSON sidecars
        if ids_base is None:
            ids_base = load_ids_from_json_sidecars(run_dir, "baseline")
        if ids_tgt is None:
            ids_tgt = load_ids_from_json_sidecars(run_dir, "targeted")

    # Fallback: synthesize IDs if still missing
    if ids_base is None:
        ids_base = [f"baseline_{i:04d}" for i in range(base_rows.shape[0])]
    if ids_tgt is None:
        ids_tgt = [f"targeted_{i:04d}" for i in range(tgt_rows.shape[0])]

    # Final sanity
    if len(ids_base) != base_rows.shape[0]:
        raise ValueError(f"{run_dir.name}: baseline id count mismatch ({len(ids_base)} vs {base_rows.shape[0]})")
    if len(ids_tgt) != tgt_rows.shape[0]:
        raise ValueError(f"{run_dir.name}: targeted id count mismatch ({len(ids_tgt)} vs {tgt_rows.shape[0]})")

    # Write CSVs if missing
    if not out_base.exists():
        save_matrix_csv(base_rows, ref_metric_names, list(map(str, ids_base)), out_base)
    if not out_tgt.exists():
        save_matrix_csv(tgt_rows, ref_metric_names, list(map(str, ids_tgt)), out_tgt)

    return True


def main():
    ap = argparse.ArgumentParser(description="Backfill baseline/targeted CSVs from NPZ using reference headers")
    ap.add_argument("--runs_dir", type=Path, required=True, help="Parent folder containing run subfolders")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--ref_run", type=str, help="Run ID whose CSV defines the canonical metric order (e.g., 909)")
    grp.add_argument("--ref_csv", type=Path, help="Direct path to a reference CSV file")
    args = ap.parse_args()

    if args.ref_csv:
        ref_csv = args.ref_csv
    else:
        ref_csv = find_reference_csvs(args.runs_dir / args.ref_run)

    ref_metric_names = load_header_from_csv(ref_csv)

    for run_dir in sorted([p for p in args.runs_dir.iterdir() if p.is_dir()]):
        try:
            ok = build_csvs_from_npz(run_dir, ref_metric_names)
            status = "ok" if ok else "skip (no NPZ)"
            print(f"[{status}] {run_dir.name}")
        except Exception as e:
            print(f"[error] {run_dir.name}: {e}")


if __name__ == "__main__":
    main()
