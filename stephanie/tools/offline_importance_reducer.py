# stephanie/tools/offline_importance_reducer.py
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List


def _parse_importance_file(path: Path) -> List[Dict[str, Any]]:
    """
    Normalize any of our importance JSON formats into a list:
        [{"name": str, "auc": float|None, "cohen_d": float|None}, ...]
    Supports:
      - GAP-style list:
          [
            {"name": "...", "cohen_d": ..., "auc": ...},
            ...
          ]
      - dict with:
          - "metrics": [...]
          - "metric_importance": [...]
          - "metrics_by_cohens_d": [
                {"metric": "...", "stats": {...}},
                ...
            ]
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    metrics: List[Dict[str, Any]] = []

    def add_metric(name: str | None, stats: Dict[str, Any]):
        if not name:
            return
        # AUC can appear as "auc" or "auc_roc"
        auc = stats.get("auc")
        if auc is None:
            auc = stats.get("auc_roc")

        # Effect size can appear as "cohen_d", "cohens_d", or "abs_cohen_d"
        cd = stats.get("cohen_d")
        if cd is None:
            cd = stats.get("cohens_d")
        if cd is None:
            cd = stats.get("abs_cohen_d")

        metrics.append(
            {
                "name": str(name),
                "auc": float(auc) if auc is not None else None,
                "cohen_d": float(cd) if cd is not None else None,
            }
        )

    def handle_list(lst: List[Any]):
        for entry in lst:
            if isinstance(entry, str):
                metrics.append({"name": entry, "auc": None, "cohen_d": None})
            elif isinstance(entry, dict):
                name = (
                    entry.get("metric")
                    or entry.get("name")
                    or entry.get("metric_name")
                    or entry.get("key")
                )
                add_metric(name, entry)

    # --- Shape 1: plain list (GAP-style metric_importance.json)
    if isinstance(data, list):
        handle_list(data)

    # --- Shape 2: dict with various lists
    elif isinstance(data, dict):
        # Possible keys: "metrics", "metric_importance", "metrics_by_cohens_d"
        if isinstance(data.get("metrics"), list):
            handle_list(data["metrics"])

        if isinstance(data.get("metric_importance"), list):
            handle_list(data["metric_importance"])

        if isinstance(data.get("metrics_by_cohens_d"), list):
            for entry in data["metrics_by_cohens_d"]:
                if not isinstance(entry, dict):
                    continue
                name = entry.get("metric") or entry.get("name")
                stats = entry.get("stats") or {}
                add_metric(name, stats)

        # We *could* also parse data.get("metrics_all"), but
        # "metrics_by_cohens_d" is already sorted and richer.

    return metrics


def _aggregate_metrics(all_runs: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Aggregate per-run metric entries into global stats.

    Input:
      all_runs = [
        [ {"name": "...", "auc": ..., "cohen_d": ...}, ... ],  # run 1
        [ {"name": "...", "auc": ..., "cohen_d": ...}, ... ],  # run 2
        ...
      ]

    Output (sorted best-first):
      [
        {
          "name": str,
          "count": int,
          "auc_mean": float|None,
          "auc_std": float,
          "cohen_d_mean": float|None,
          "cohen_d_abs_mean": float|None,
        },
        ...
      ]
    """
    agg: Dict[str, Dict[str, Any]] = {}

    for run_metrics in all_runs:
        for m in run_metrics:
            name = m.get("name")
            if not name:
                continue
            rec = agg.setdefault(name, {"auc": [], "cohen_d": [], "count": 0})
            if m.get("auc") is not None:
                rec["auc"].append(float(m["auc"]))
            if m.get("cohen_d") is not None:
                rec["cohen_d"].append(float(m["cohen_d"]))
            rec["count"] += 1

    summary: List[Dict[str, Any]] = []

    for name, vals in agg.items():
        auc_vals = vals["auc"]
        cd_vals = vals["cohen_d"]

        if auc_vals:
            auc_mean = float(statistics.mean(auc_vals))
            auc_std = float(statistics.pstdev(auc_vals)) if len(auc_vals) > 1 else 0.0
        else:
            auc_mean = None
            auc_std = 0.0

        if cd_vals:
            cd_mean = float(statistics.mean(cd_vals))
            cd_abs_mean = float(statistics.mean([abs(x) for x in cd_vals]))
        else:
            cd_mean = None
            cd_abs_mean = None

        summary.append(
            {
                "name": name,
                "count": int(vals["count"]),
                "auc_mean": auc_mean,
                "auc_std": auc_std,
                "cohen_d_mean": cd_mean,
                "cohen_d_abs_mean": cd_abs_mean,
            }
        )

    def sort_key(e: Dict[str, Any]):
        # Treat None as neutral values.
        auc = e["auc_mean"] if e["auc_mean"] is not None else 0.5
        cd = e["cohen_d_abs_mean"] if e["cohen_d_abs_mean"] is not None else 0.0
        return (auc, cd)

    summary.sort(key=sort_key, reverse=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate metric importance across VisiCalc runs "
        "and write a core metric subset usable by CriticCohortAgent."
    )
    parser.add_argument(
        "--runs-root",
        type=str,
        default="runs/visicalc",
        help="Root directory containing per-run subdirs "
        "(default: runs/visicalc)",
    )
    parser.add_argument(
        "--importance-filenames",
        nargs="*",
        default=None,
        help=(
            "Candidate importance filenames to look for in each run dir. "
            "If not provided, defaults to: metric_importance.json "
            "and visicalc_metric_importance.json"
        ),
    )
    parser.add_argument(
        "--min-runs",
        type=int,
        default=1,
        help="Minimum number of runs a metric must appear in to be kept.",
    )
    parser.add_argument(
        "--min-auc",
        type=float,
        default=0.0,
        help="Minimum mean AUC to keep a metric (0–1).",
    )
    parser.add_argument(
        "--min-abs-cohen-d",
        type=float,
        default=0.0,
        help="Minimum mean |Cohen's d| to keep a metric.",
    )
    parser.add_argument(
        "--max-core-metrics",
        type=int,
        default=150,
        help="Maximum number of core metrics to keep (after filtering).",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default="config/core_metrics.json",
        help="Where to write the aggregated core metrics JSON.",
    )

    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    if not runs_root.exists():
        print(f"[OfflineImportanceReducer] runs_root does not exist: {runs_root}")
        return

    importance_filenames = args.importance_filenames or [
        "metric_importance.json",
        "visicalc_metric_importance.json",
    ]

    all_runs: List[List[Dict[str, Any]]] = []
    num_files = 0

    print(f"[OfflineImportanceReducer] Scanning runs under: {runs_root}")

    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue

        loaded_for_run = False

        for fname in importance_filenames:
            fpath = run_dir / fname
            if not fpath.exists():
                continue

            try:
                metrics = _parse_importance_file(fpath)
            except Exception as e:
                print(
                    f"[OfflineImportanceReducer] ERROR reading {fpath}: {e}. Skipping."
                )
                continue

            if not metrics:
                print(
                    f"[OfflineImportanceReducer] {run_dir.name}: "
                    f"{fname} contained 0 metrics. Skipping."
                )
                continue

            print(
                f"[OfflineImportanceReducer] {run_dir.name}: "
                f"loaded {len(metrics)} metrics from {fname}"
            )
            all_runs.append(metrics)
            num_files += 1
            loaded_for_run = True
            break  # don't double-count this run with multiple files

        if not loaded_for_run:
            # Not fatal; just informational if you want:
            # print(f"[OfflineImportanceReducer] {run_dir.name}: no importance file found.")
            pass

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not all_runs:
        print(
            "[OfflineImportanceReducer] Aggregated 0 metrics across runs "
            "(no usable importance files found)."
        )
        empty = {"num_runs": 0, "num_files": 0, "metrics": []}
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(empty, f, indent=2)
        print(f"[OfflineImportanceReducer] Wrote empty core metrics → {out_path}")
        return

    summary = _aggregate_metrics(all_runs)

    # Apply filters
    filtered: List[Dict[str, Any]] = []
    for rec in summary:
        if rec["count"] < args.min_runs:
            continue
        if rec["auc_mean"] is not None and rec["auc_mean"] < args.min_auc:
            continue
        if (
            rec["cohen_d_abs_mean"] is not None
            and rec["cohen_d_abs_mean"] < args.min_abs_cohen_d
        ):
            continue
        filtered.append(rec)

    if args.max_core_metrics and args.max_core_metrics > 0:
        filtered = filtered[: args.max_core_metrics]

    aggregated_total = sum(len(run) for run in all_runs)

    out_obj = {
        "num_runs": len(all_runs),
        "num_files": num_files,
        "num_metrics": len(summary),
        "num_core_metrics": len(filtered),
        # IMPORTANT: "metrics" is a list of dicts with 'name', which
        # CriticCohortAgent._load_important_metric_names knows how to read.
        "metrics": filtered,
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2)

    print(
        f"[OfflineImportanceReducer] Aggregated {aggregated_total} metrics "
        f"across {num_files} files in {len(all_runs)} runs"
    )
    print(
        f"[OfflineImportanceReducer] Selected {len(filtered)} core metrics "
        f"(min_runs={args.min_runs}, min_auc={args.min_auc}, "
        f"min_abs_cohen_d={args.min_abs_cohen_d})"
    )
    print(f"[OfflineImportanceReducer] Wrote → {out_path}")


if __name__ == "__main__":
    main()
