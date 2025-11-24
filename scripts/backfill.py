
import json
import csv
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import numpy as np

# --------- CONFIG ---------
RUNS_DIR = Path("runs/visicalc")  # parent folder containing per-run subfolders
# --------------------------

def _save_matrix_csv(matrix: np.ndarray, metric_names: List[str], item_ids: List[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scorable_id", *metric_names])
        for ridx in range(matrix.shape[0]):
            row = [item_ids[ridx]] + [float(x) for x in matrix[ridx]]
            w.writerow(row)

def _save_ab_npz(vpm_base: np.ndarray, vpm_tgt: np.ndarray, metric_names: List[str], out_path: Path) -> None:
    X = np.concatenate([vpm_base, vpm_tgt], axis=0).astype(np.float32, copy=False)
    y = np.concatenate([np.zeros(vpm_base.shape[0], dtype=np.int64),
                        np.ones(vpm_tgt.shape[0], dtype=np.int64)], axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path.as_posix(), X=X, y=y, metric_names=np.array(metric_names, dtype=object))

def _extract_matrix_from_json(j: Dict[str, Any]) -> Optional[Tuple[np.ndarray, List[str], List[str]]]:
    candidates = [j]
    for key in ("report", "data", "visicalc", "payload"):
        if isinstance(j.get(key), dict):
            candidates.append(j[key])

    for obj in candidates:
        if not isinstance(obj, dict):
            continue
        scores = obj.get("scores") or obj.get("matrix") or obj.get("vpm")
        metric_names = obj.get("metric_names") or obj.get("metrics") or obj.get("columns")
        item_ids = obj.get("item_ids") or obj.get("rows") or obj.get("scorable_ids")
        if scores is None and isinstance(obj.get("rows"), list):
            try:
                rows_list = obj["rows"]
                maybe_ids = []
                maybe_scores = []
                for r in rows_list:
                    if isinstance(r, dict) and "metrics_values" in r:
                        maybe_ids.append(str(r.get("scorable_id", "unknown")))
                        maybe_scores.append([float(x) for x in r["metrics_values"]])
                if maybe_scores:
                    scores = maybe_scores
                    item_ids = maybe_ids
            except Exception:
                pass
        if scores is not None and metric_names is not None and item_ids is not None:
            try:
                arr = np.array(scores, dtype=float)
                names = [str(x) for x in metric_names]
                ids = [str(x) for x in item_ids]
                if arr.ndim == 2 and arr.shape[1] == len(names) and arr.shape[0] == len(ids):
                    return arr, names, ids
            except Exception:
                continue
    return None

def _try_rebuild_ab_from_json(run_dir: Path) -> bool:
    tjson = run_dir / "visicalc_targeted.json"
    bjson = run_dir / "visicalc_baseline.json"
    if not (tjson.exists() and bjson.exists()):
        return False
    try:
        tgt = json.loads(tjson.read_text(encoding="utf-8"))
        base = json.loads(bjson.read_text(encoding="utf-8"))
        t_tuple = _extract_matrix_from_json(tgt)
        b_tuple = _extract_matrix_from_json(base)
        if not (t_tuple and b_tuple):
            return False
        vpm_tgt, names_tgt, ids_tgt = t_tuple
        vpm_base, names_base, ids_base = b_tuple
        if names_tgt != names_base:
            return False
        _save_matrix_csv(vpm_tgt, names_tgt, ids_tgt, run_dir / "visicalc_targeted_matrix.csv")
        _save_matrix_csv(vpm_base, names_base, ids_base, run_dir / "visicalc_baseline_matrix.csv")
        _save_ab_npz(vpm_base, vpm_tgt, names_tgt, run_dir / "visicalc_ab_dataset.npz")
        return True
    except Exception:
        return False

def _try_rebuild_single_from_json(run_dir: Path) -> bool:
    sjson = run_dir / "visicalc_report.json"
    if not sjson.exists():
        return False
    try:
        j = json.loads(sjson.read_text(encoding="utf-8"))
        tup = _extract_matrix_from_json(j)
        if not tup:
            return False
        vpm, names, ids = tup
        _save_matrix_csv(vpm, names, ids, run_dir / "visicalc_cohort_matrix.csv")
        return True
    except Exception:
        return False

def load_scorables_for_run(run_id: str) -> Optional[List[Dict[str, Any]]]:
    return None  # implement if you want automatic recompute using your DB

def _try_recompute_with_agent(run_dir: Path) -> bool:
    try:
        run_id = run_dir.name
        scorables = load_scorables_for_run(run_id)
        if not scorables:
            return False
        from visicalc import VisiCalcAgent
        agent = VisiCalcAgent(cfg={"visicalc": {"out_dir": str(RUNS_DIR)}}, memory=None, container=None, logger=None)
        context = {agent.input_key: scorables, "run_id": run_id}
        import asyncio
        asyncio.get_event_loop().run_until_complete(agent.run(context))
        ok = _try_rebuild_ab_from_json(run_dir) or _try_rebuild_single_from_json(run_dir)
        return ok
    except Exception:
        return False

def backfill_runs(root: Path) -> Dict[str, str]:
    results: Dict[str, str] = {}
    if not root.exists():
        return results
    for run_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        run_id = run_dir.name
        need_npz = not (run_dir / "visicalc_ab_dataset.npz").exists()
        need_t_csv = not (run_dir / "visicalc_targeted_matrix.csv").exists()
        need_b_csv = not (run_dir / "visicalc_baseline_matrix.csv").exists()
        need_c_csv = not (run_dir / "visicalc_cohort_matrix.csv").exists()
        if not (need_npz or need_t_csv or need_b_csv or need_c_csv):
            results[run_id] = "already complete"
            continue
        if _try_rebuild_ab_from_json(run_dir) or _try_rebuild_single_from_json(run_dir):
            results[run_id] = "rebuilt from JSON"
            continue
        if _try_recompute_with_agent(run_dir):
            results[run_id] = "recomputed with agent"
            continue
        results[run_id] = "skipped (no JSON/matrices; implement loader)"
    return results

if __name__ == "__main__":
    summary = backfill_runs(RUNS_DIR)
    report_path = RUNS_DIR / "_backfill_summary.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
