# stephanie/components/critic/tiny_critic_dataset.py
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Tuple
import csv
import numpy as np
import logging
from stephanie.scoring.metrics.dynamic_features import (
    load_core_metric_names,
    build_dynamic_feature_vector,
)

log = logging.getLogger(__name__)

CORE_METRIC_PATH = Path("config/core_metrics.json")
CORE_FEATURE_NAMES = [
    "stability", "middle_dip", "std_dev", "sparsity",
    "entropy", "trend", "mid_bad_ratio", "frontier_util"
]
CORE_FEATURE_COUNT = len(CORE_FEATURE_NAMES)

ALIAS_MAP = {
    "Tiny.coverage.attr.raw01": "Tiny.coverage.attr.tiny.score01",
    "Tiny.coverage.attr.values[0]": "Tiny.coverage.attr.tiny.score01",
    "Tiny.coverage.attr.vector.tiny.score01": "Tiny.coverage.attr.tiny.score01",
    "Tiny.coverage.score": "Tiny.coverage.attr.tiny.score01",
    "Tiny.coverage.attr.values[1]": "Tiny.coverage.attr.tiny.score100",
    "Tiny.coverage.attr.vector.tiny.score100": "Tiny.coverage.attr.tiny.score100",
}

def canonicalize_metric_names(names: list[str]) -> list[str]:
    return [ALIAS_MAP.get(n, n) for n in names]

def _ensure_dir(p: Path) -> None:
    """Ensure directory exists with proper error handling."""
    try:
        p.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        log.error(f"Failed to create directory {p}: {e}")
        return False

def _load_metric_means_from_csv(run_dir: Path, label: str) -> dict:
    """
    Load metric means from CSV matrix file.
    Handles:
      - visicalc_targeted_matrix.csv (label="targeted")
      - visicalc_baseline_matrix.csv (label="baseline")
    """
    csv_name = f"visicalc_{label}_matrix.csv"
    csv_path = run_dir / csv_name
    if not csv_path.exists():
        return {}

    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            r = csv.reader(f)
            header = next(r, None)
            if not header or header[0] != "scorable_id":
                raise ValueError(f"{csv_name} missing 'scorable_id' header")
            metric_names = header[1:]
            sums = [0.0] * len(metric_names)
            cnt = 0
            for row in r:
                if not row:
                    continue
                values = row[1:]
                # coerce to float; blank → 0.0
                vals = [float(v) if v not in ("", None) else 0.0 for v in values]
                if len(vals) != len(metric_names):
                    raise ValueError(f"{csv_name}: row width mismatch at line {cnt+2}")
                for i, v in enumerate(vals):
                    sums[i] += v
                cnt += 1

            if cnt == 0:
                return {}

            means = [s / cnt for s in sums]
            return dict(zip(metric_names, means))
    except Exception as e:
        log.error(f"Failed to load metric means from {csv_path}: {e}")
        return {}

def load_metrics_for_run(run_dir: Path, label: str) -> dict:
    """
    Load metrics from CSV matrix first, then fall back to JSON.
    Returns:
      {metric_name: mean_value} dictionary
    """
    # 1) CSV means (new path)
    means = _load_metric_means_from_csv(run_dir, label)
    if means:
        log.info(f"✅ Loaded {len(means)} metrics from CSV for {label} in {run_dir.name}")
        return means

    # 2) Fallback JSON (legacy path)
    metrics_path = run_dir / f"metrics_{label}.json"
    if metrics_path.exists():
        try:
            with metrics_path.open("r", encoding="utf-8") as f:
                data = json.load(f) or {}
            log.info(f"✅ Loaded JSON metrics from {metrics_path}")
            return data
        except Exception as e:
            log.error(f"❌ Failed to load metrics from {metrics_path}: {e}")
    else:
        log.debug(f"ℹ️ No metrics file for {label} at {metrics_path}, using defaults")

    return {}

def load_visicalc_report(path: Path) -> dict:
    """Load VisiCalc report with detailed error handling."""
    log.info(f"📂 Loading VisiCalc report: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        # Validate expected structure
        required_keys = ['frontier', 'global', 'regions']
        if not all(k in data for k in required_keys):
            log.warning(f"⚠️  VisiCalc report missing expected keys: {required_keys}")
        log.info(
            f"✅ Successfully loaded VisiCalc report: {path.name} "
            f"(keys: {list(data.keys()) if data else 'empty'})"
        )
        return data
    except Exception as e:
        log.error(f"❌ Failed to load VisiCalc report {path}: {e}")
        raise

def collect_visicalc_samples(visicalc_root: str | Path) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Collect VisiCalc samples with proper core/dynamic feature handling.
    Returns:
      (X, y, metric_names, groups) where:
        - X: [n_samples, n_features] feature matrix
        - y: [n_samples] binary labels (1=targeted, 0=baseline)
        - metric_names: list of feature names in order
        - groups: list of run identifiers for each sample
    """
    visicalc_root = Path(visicalc_root)
    X: List[np.ndarray] = []
    y: List[int] = []

    log.info(f"🔍 Starting VisiCalc sample collection from: {visicalc_root.absolute()}")

    if not visicalc_root.exists():
        log.error(f"❌ VisiCalc root directory not found: {visicalc_root}")
        raise FileNotFoundError(f"VisiCalc root not found: {visicalc_root}")

    # Count directories for progress tracking
    all_dirs = [d for d in sorted(visicalc_root.iterdir()) if d.is_dir()]
    log.info(f"📁 Found {len(all_dirs)} potential run directories to process")

    processed_runs = 0
    successful_runs = 0
    skipped_runs = 0

    # Get dynamic metric names (core features are handled internally)
    dynamic_metric_names = load_core_metric_names(CORE_METRIC_PATH)
    log.info(f"📌 Loaded {len(dynamic_metric_names)} dynamic metric names from MARS summary")

    # Complete feature names (core + dynamic)
    metric_names = CORE_FEATURE_NAMES + dynamic_metric_names
    log.info(f"📌 Total feature names: {len(metric_names)} (8 core + {len(dynamic_metric_names)} dynamic)")

    # Each subdirectory under runs/visicalc is a run_id (e.g. 8967)
    groups: list[str] = []
    for run_dir in all_dirs:
        processed_runs += 1
        log.info(f"🔄 Processing run {processed_runs}/{len(all_dirs)}: {run_dir.name}")

        targeted_path = run_dir / "visicalc_targeted.json"
        baseline_path = run_dir / "visicalc_baseline.json"

        # Check if required files exist
        if not targeted_path.exists():
            log.warning(
                f"⚠️  Skipping {run_dir.name}: targeted report missing at {targeted_path}"
            )
            skipped_runs += 1
            continue
        if not baseline_path.exists():
            log.warning(
                f"⚠️  Skipping {run_dir.name}: baseline report missing at {baseline_path}"
            )
            skipped_runs += 1
            continue

        # Load metrics first (may be empty dicts)
        targeted_metrics = load_metrics_for_run(run_dir, label="targeted")
        baseline_metrics = load_metrics_for_run(run_dir, label="baseline")

        try:
            log.info(f"📊 Loading reports for {run_dir.name}...")
            targeted_report = load_visicalc_report(targeted_path)
            baseline_report = load_visicalc_report(baseline_path)

            # Extract combined features for targeted (label=1)
            # IMPORTANT: Only pass dynamic metrics to build_dynamic_feature_vector
            # (core features are added internally)
            log.info(
                f"🔧 Building combined features for targeted report {targeted_path}"
            )
            xt = build_dynamic_feature_vector(
                targeted_report,
                targeted_metrics,
                dynamic_metric_names
            )
            # Validate feature vector length
            if len(xt) != len(metric_names):
                log.error(
                    f"❌ Feature vector length mismatch: "
                    f"expected {len(metric_names)}, got {len(xt)}"
                )
                # Attempt to fix by padding or truncating
                if len(xt) < len(metric_names):
                    xt = np.pad(xt, (0, len(metric_names) - len(xt)))
                else:
                    xt = xt[:len(metric_names)]

            X.append(xt)
            y.append(1)
            groups.append(run_dir.name) # Group for targeted sample

            log.info(
                f"✅ Targeted combined features: shape={xt.shape}, dtype={xt.dtype}"
            )

            # Extract combined features for baseline (label=0)
            log.info(
                f"🔧 Building combined features for baseline report {baseline_path}"
            )
            xb = build_dynamic_feature_vector(
                baseline_report,
                baseline_metrics,
                dynamic_metric_names
            )
            # Validate feature vector length
            if len(xb) != len(metric_names):
                log.error(
                    f"❌ Feature vector length mismatch: "
                    f"expected {len(metric_names)}, got {len(xb)}"
                )
                # Attempt to fix by padding or truncating
                if len(xb) < len(metric_names):
                    xb = np.pad(xb, (0, len(metric_names) - len(xb)))
                else:
                    xb = xb[:len(metric_names)]

            X.append(xb)
            y.append(0)
            groups.append(run_dir.name) # Group for baseline sample

            log.info(
                f"✅ Baseline combined features: shape={xb.shape}, dtype={xb.dtype}"
            )

            successful_runs += 1
            log.info(
                f"🎉 Successfully processed run {run_dir.name} "
                f"(total samples: {len(X)}, successful runs: {successful_runs})"
            )

        except Exception as e:
            log.error(f"❌ Failed to process run {run_dir.name}: {e}")
            skipped_runs += 1
            continue

    # Final summary
    log.info(
        "📊 Collection complete! Processed: %d, Successful: %d, Skipped: %d",
        processed_runs,
        successful_runs,
        skipped_runs,
    )

    if not X:
        log.error(f"❌ No VisiCalc samples found under {visicalc_root}")
        raise RuntimeError(f"No VisiCalc samples found under {visicalc_root}")

    # Convert to numpy arrays
    log.info(f"🔄 Converting {len(X)} samples to numpy arrays...")
    X_arr = np.stack(X, axis=0)
    y_arr = np.asarray(y, dtype=np.int64)

    log.info(
        "✅ Dataset created: X.shape=%s, y.shape=%s, dtypes: X=%s, y=%s",
        X_arr.shape,
        y_arr.shape,
        X_arr.dtype,
        y_arr.dtype,
    )

    # Log some statistics
    log.info(
        "📈 Dataset statistics: Targeted samples (y=1): %d, Baseline samples (y=0): %d, "
        "Feature range: [%.3f, %.3f]",
        int(np.sum(y_arr == 1)),
        int(np.sum(y_arr == 0)),
        float(X_arr.min()),
        float(X_arr.max()),
    )

    return X_arr, y_arr, metric_names, groups

def save_dataset(
    X: np.ndarray,
    y: np.ndarray,
    metric_names: List[str],
    groups: List[str],
    out_path: str | Path
) -> None:
    """Save dataset with metric names and groups included and comprehensive logging."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Save with complete feature names and groups
        np.savez_compressed(
            out_path,
            X=X,
            y=y,
            metric_names=np.array(metric_names, dtype=object),
            groups=np.array(groups, dtype=object) # Include groups for later CV
        )
        # Log file info
        file_size = out_path.stat().st_size / (1024 * 1024) if out_path.exists() else 0
        log.info(
            "✅ Dataset saved successfully! File: %s Size: %.2f MB X.shape: %s y.shape: %s",
            out_path,
            file_size,
            X.shape,
            y.shape,
        )
    except Exception as e:
        log.error(f"❌ Failed to save dataset: {e}")
        raise

def main(args):
    """Main function for dataset generation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    root = Path("runs/visicalc")
    out = Path("data/critic.npz")

    log.info("🚀 Starting Tiny Critic Dataset creation...")
    log.info(f"Source: {root.absolute()}")
    log.info(f"Destination: {out.absolute()}")

    try:
        # 1. Collect dataset
        X, y, metric_names, groups = collect_visicalc_samples(root)
        # 1b. Canonicalize names once for consistency in saved file
        metric_names = canonicalize_metric_names(list(metric_names))

        # 2. Core sanity checks
        core_dim = CORE_FEATURE_COUNT
        core_block = X[:, :core_dim]
        if np.isnan(core_block).any():
            raise ValueError("NaN in VisiCalc core features — check VisiCalc computation.")
        zero_var_core = int((core_block.std(axis=0) <= 1e-12).sum())
        if zero_var_core:
            log.warning("VisiCalc: %d core features are ~constant across samples", zero_var_core)

        # 3. Save full dataset (including groups)
        save_dataset(X, y, metric_names, groups, out)

        log.info("🎉 Tiny Critic Dataset generation completed successfully!")

    except Exception as e:
        log.error(f"💥 Failed to create Tiny Critic Dataset: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    # Keep core_dim for consistency, but it's primarily used in the reporting script now
    p = argparse.ArgumentParser(description="Tiny Critic Dataset Builder (Data Generation Only)")
    p.add_argument(
        "--core_dim",
        type=int,
        default=CORE_FEATURE_COUNT,
        help="Number of core VisiCalc features (for consistency, used in reporting script)"
    )
    main(p.parse_args())