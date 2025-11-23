# stephanie/dataloaders/tiny_critic_dataset.py
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

from stephanie.scoring.metrics.visicalc_features import extract_tiny_features
from stephanie.scoring.metrics.dynamic_features import (
    load_core_metric_names,
    build_dynamic_feature_vector,
)

import logging

log = logging.getLogger(__name__)

CORE_METRIC_PATH = Path("config/metrics/core_metrics.json")


def load_metrics_for_run(run_dir: Path, label: str) -> dict:
    """
    Load dynamic metrics for this run + label ("targeted"/"baseline").

    Expected layout (adjust as needed):

      runs/visicalc/<run_id>/metrics/metrics_targeted.json
      runs/visicalc/<run_id>/metrics/metrics_baseline.json

    Returns:
        dict of metric_name -> value, or {} if unavailable.
    """
    metrics_path = run_dir / "metrics" / f"metrics_{label}.json"
    if metrics_path.exists():
        try:
            with metrics_path.open("r", encoding="utf-8") as f:
                data = json.load(f) or {}
            log.info(f"✅ Successfully loaded metrics from {metrics_path}")
            return data
        except Exception as e:
            log.error(f"❌ Failed to load metrics from {metrics_path}: {e}")
    else:
        log.debug(f"ℹ️ No metrics file for {label} at {metrics_path}, using defaults")
    return {}


def load_visicalc_report(path: Path) -> dict:
    """Load a VisiCalc report with detailed logging."""
    log.info(f"📂 Loading VisiCalc report: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        log.info(
            "✅ Successfully loaded VisiCalc report: %s (keys: %s)",
            path.name,
            list(data.keys()) if data else "empty",
        )
        return data
    except Exception as e:
        log.error(f"❌ Failed to load VisiCalc report {path}: {e}")
        raise


def collect_visicalc_samples(visicalc_root: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Walk runs/visicalc/* and collect (X, y) pairs:

      - X: [n_samples, n_features] from
            build_dynamic_feature_vector(visicalc_report, metrics, metric_names)
      - y: 1 for 'targeted', 0 for 'baseline'
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

    # Dynamic metric feature names
    metric_names = load_core_metric_names(CORE_METRIC_PATH)
    log.info(f"📌 Using {len(metric_names)} dynamic metric features")

    # Each subdirectory under runs/visicalc is a run_id (e.g. 8967)
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
            log.info(
                f"🔧 Building combined features for targeted report {targeted_path}"
            )
            xt = build_dynamic_feature_vector(
                targeted_report, targeted_metrics, metric_names
            )
            X.append(xt)
            y.append(1)
            log.info(
                "✅ Targeted combined features: shape=%s, dtype=%s",
                xt.shape,
                xt.dtype,
            )

            # Extract combined features for baseline (label=0)
            log.info(
                f"🔧 Building combined features for baseline report {baseline_path}"
            )
            xb = build_dynamic_feature_vector(
                baseline_report, baseline_metrics, metric_names
            )
            X.append(xb)
            y.append(0)
            log.info(
                "✅ Baseline combined features: shape=%s, dtype=%s",
                xb.shape,
                xb.dtype,
            )

            successful_runs += 1
            log.info(
                "🎉 Successfully processed run %s (total samples: %d, successful runs: %d)",
                run_dir.name,
                len(X),
                successful_runs,
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

    return X_arr, y_arr


def save_dataset(X: np.ndarray, y: np.ndarray, out_path: str | Path) -> None:
    """Save dataset with detailed logging."""
    out_path = Path(out_path)
    log.info(f"💾 Saving dataset to: {out_path.absolute()}")

    # Create parent directory if needed
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"📁 Ensuring directory exists: {out_path.parent}")

    # Save the dataset
    np.savez_compressed(out_path, X=X, y=y)

    # Log file info
    file_size = out_path.stat().st_size if out_path.exists() else 0
    file_size_mb = file_size / (1024 * 1024)

    log.info(
        "✅ Dataset saved successfully! File: %s Size: %.2f MB X.shape: %s y.shape: %s",
        out_path,
        file_size_mb,
        X.shape,
        y.shape,
    )


def visualize_features(X: np.ndarray, y: np.ndarray):
    """
    Create visualizations of feature distributions for the core 8 VisiCalc features.

    NOTE: If you have additional dynamic metrics, this currently visualizes only the
    first 8 dims (the original VisiCalc structural features).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    # Core VisiCalc feature names
    feature_names = [
        "stability",
        "middle_dip",
        "std_dev",
        "sparsity",
        "entropy",
        "trend",
        "mid_bad_ratio",
        "frontier_util",
    ]

    n_core = min(len(feature_names), X.shape[1])

    plt.figure(figsize=(15, 10))
    for i in range(n_core):
        name = feature_names[i]
        plt.subplot(2, 4, i + 1)
        sns.histplot(x=X[:, i], hue=y, kde=True, bins=20)
        plt.title(f"Distribution: {name}")
        plt.xlabel("Value")
        plt.ylabel("Count")

    plt.tight_layout()
    plt.savefig("data/feature_distributions.png", dpi=300)
    plt.close()

    # Correlation heatmap on the same core features
    plt.figure(figsize=(10, 8))
    df = pd.DataFrame(X[:, :n_core], columns=feature_names[:n_core])
    df["is_good"] = y
    corr = df.corr()

    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlations (core VisiCalc features)")
    plt.tight_layout()
    plt.savefig("data/feature_correlations.png", dpi=300)
    plt.close()

    log.info(
        "📊 Feature visualizations saved: "
        "data/feature_distributions.png, data/feature_correlations.png"
    )


def main():
    """Main function with comprehensive logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    root = Path("runs/visicalc")
    out = Path("data/tiny_visicalc_critic.npz")

    log.info("🚀 Starting Tiny Critic Dataset creation...")
    log.info(f"Source: {root.absolute()}")
    log.info(f"Destination: {out.absolute()}")

    try:
        X, y = collect_visicalc_samples(root)
        visualize_features(X, y)
        save_dataset(X, y, out)
        log.info("🎉 Tiny Critic Dataset creation completed successfully!")
    except Exception as e:
        log.error(f"💥 Failed to create Tiny Critic Dataset: {e}")
        raise


# python stephanie/dataloaders/tiny_critic_dataset.py
if __name__ == "__main__":
    main()
