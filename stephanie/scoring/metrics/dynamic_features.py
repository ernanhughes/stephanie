# stephanie/scoring/metrics/dynamic_features.py
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

CORE_METRICS_PATH = Path("config/core_metrics.json")

log = logging.getLogger(__name__)


def load_core_metric_names(path: Path) -> List[str]:
    """
    Load the list of core metric names from JSON.

    Supported formats:

    1) Simple list:
       ["metric.a", "metric.b", ...]

    2) Dict with explicit list:
       {"core_metrics": ["metric.a", ...]}
       {"metrics": ["metric.a", ...]}

    3) MARS summary format (your file):
       {
         "num_core_metrics": 150,
         "metrics": [
           {"name": "sicql.faithfulness.attr.advantage", ...},
           {"name": "Tiny.coverage.attr.consistency_hat", ...},
           ...
         ]
       }
    """
    log.info(f"📂 Loading core metrics from: {path.absolute()}")
    
    if not path.exists():
        log.warning(f"⚠️  No core metric config found at {path}")
        log.info("ℹ️  Dynamic metrics will be unnamed - using VisiCalc features only")
        return []

    try:
        log.debug(f"🔍 Reading core metrics file: {path}")
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        log.debug(f"✅ Successfully loaded core metrics file (size: {len(str(data))} chars)")
    except Exception as e:
        log.error(f"❌ Failed to load core metric config from {path}: {e}")
        return []

    # --- Case 1: file is just a list of names ---
    if isinstance(data, list):
        names = [str(x) for x in data]
        log.info(f"📌 Loaded {len(names)} core metric names from simple list")
        log.debug(f"🔢 First 5 metrics: {names[:5]}")
        return names

    # --- Case 2: dict with explicit list fields ---
    if isinstance(data, dict):
        log.debug(f"📊 Processing dictionary format with keys: {list(data.keys())}")
        
        # If it's already a list of strings under core_metrics / metrics
        for key in ("core_metrics", "metrics"):
            if key in data and isinstance(data[key], list):
                if all(isinstance(x, str) for x in data[key]):
                    names = [str(x) for x in data[key]]
                    log.info(f"📌 Loaded {len(names)} core metric names from '{key}' field")
                    log.debug(f"🔢 First 5 metrics: {names[:5]}")
                    return names
                else:
                    log.warning(f"⚠️  Field '{key}' exists but contains non-string elements")

        # --- Case 3: MARS summary style ---
        #   { "num_core_metrics": 150,
        #     "metrics": [ {"name": "...", "auc_mean": ..., ...}, ... ] }
        metrics = data.get("metrics")
        if isinstance(metrics, list) and metrics:
            log.debug(f"📋 Found {len(metrics)} metric entries in MARS format")
            
            # Check if first element is a dict with name field
            if isinstance(metrics[0], dict) and "name" in metrics[0]:
                all_names = [m.get("name") for m in metrics if m.get("name")]
                valid_names = [name for name in all_names if name is not None]
                
                # Optional: respect num_core_metrics if present
                n_core = data.get("num_core_metrics")
                if isinstance(n_core, int) and n_core > 0:
                    log.debug(f"🎯 Limiting to top {n_core} metrics as per num_core_metrics")
                    valid_names = valid_names[:n_core]
                else:
                    log.debug("ℹ️  No num_core_metrics specified, using all valid metric names")

                log.info(f"📌 Loaded {len(valid_names)} core metric names from MARS summary")
                if valid_names:
                    log.debug(f"🔢 First 5 metrics: {valid_names[:5]}")
                    # Log some statistics about the metrics if available
                    if len(metrics) > 0 and "auc_mean" in metrics[0]:
                        auc_values = [m.get("auc_mean", 0) for m in metrics[:5] if m.get("auc_mean")]
                        if auc_values:
                            log.debug(f"📈 Sample AUC values: {auc_values}")
                return valid_names
            else:
                log.warning(f"⚠️  Metrics list doesn't contain dictionaries with 'name' field in file: {path}")
        else:
            log.warning(f"⚠️  'metrics' field not found or not a list in MARS format in file: {path}")

    log.warning(
        f"⚠️  Core metric config at {path} has unexpected format; "
        "no dynamic metric names will be used"
    )
    log.debug(f"🔍 Data type: {type(data)}, sample: {str(data)[:200]}...")
    return []


def build_dynamic_feature_vector(
    visicalc_report: dict,
    metrics: Dict[str, float],
    metric_names: List[str],
    problem_id: Optional[str] = None
) -> np.ndarray:
    """
    Build [VisiCalc 8] + [dynamic metrics] feature vector.

    - visicalc_report: full VisiCalc report dict
    - metrics: dict of metric_name -> value (from HRM/SICQL/Tiny/etc.)
    - metric_names: ordered list of metric keys to pull from metrics
    - problem_id: optional identifier for logging

    Missing metrics are filled with 0.0.
    """
    start_time = datetime.now()
    problem_context = f" for {problem_id}" if problem_id else ""
    
    log.debug(f"🔧 Building dynamic feature vector{problem_context}")
    log.debug(f"📊 Input metrics dict has {len(metrics)} total metrics")
    log.debug(f"🎯 Target metric names: {len(metric_names)} core metrics")

    # 1) VisiCalc structural features (8 dims)
    log.debug("🔄 Extracting VisiCalc structural features...")
    try:
        v_feats = extract_tiny_features(visicalc_report).astype(np.float32)
        log.debug(f"✅ VisiCalc features extracted: shape={v_feats.shape}, dtype={v_feats.dtype}")
        log.debug(f"   VisiCalc feature stats: min={v_feats.min():.3f}, max={v_feats.max():.3f}, mean={v_feats.mean():.3f}")
    except Exception as e:
        log.error(f"❌ Failed to extract VisiCalc features: {e}")
        # Create fallback VisiCalc features
        v_feats = np.zeros(8, dtype=np.float32)
        log.warning("⚠️  Using zero-filled VisiCalc features as fallback")

    # 2) Dynamic metric features (N dims)
    log.debug("🔄 Processing dynamic metric features...")
    m_vals = []
    missing_metrics = []
    available_metrics = []
    
    for i, name in enumerate(metric_names):
        if name in metrics:
            try:
                value = float(metrics[name])
                m_vals.append(value)
                available_metrics.append((i, name, value))
            except (ValueError, TypeError) as e:
                log.warning(f"⚠️  Metric '{name}' has non-float value: {metrics[name]} - using 0.0")
                m_vals.append(0.0)
                missing_metrics.append(name)
        else:
            m_vals.append(0.0)
            missing_metrics.append(name)
    
    m_feats = np.asarray(m_vals, dtype=np.float32)
    
    # Log detailed metrics information
    log.debug(f"📈 Dynamic metrics: {len(available_metrics)} available, {len(missing_metrics)} missing")
    
    if available_metrics:
        # Log first few available metrics for debugging
        sample_metrics = available_metrics[:3]
        sample_str = ", ".join([f"{name}={value:.3f}" for _, name, value in sample_metrics])
        log.debug(f"🔢 Sample available metrics: {sample_str}")
        
        # Log statistics of available metrics
        if available_metrics:
            available_values = [val for _, _, val in available_metrics]
            log.debug(f"📊 Available metrics stats: min={min(available_values):.3f}, "
                     f"max={max(available_values):.3f}, mean={np.mean(available_values):.3f}")
    
    if missing_metrics:
        log.debug(f"❌ Missing metrics ({len(missing_metrics)}): {missing_metrics[:5]}{'...' if len(missing_metrics) > 5 else ''}")
        if len(missing_metrics) > 10:
            log.debug(f"   ... and {len(missing_metrics) - 5} more missing metrics")

    # 3) Concatenate → [8 + N]
    log.debug("🔗 Concatenating VisiCalc and dynamic metrics...")
    try:
        combined_features = np.concatenate([v_feats, m_feats], axis=-1)
        
        duration = (datetime.now() - start_time).total_seconds()
        log.debug(f"✅ Feature vector built successfully in {duration:.3f}s")
        log.debug(f"📐 Final feature vector: shape={combined_features.shape}, "
                 f"dtype={combined_features.dtype}")
        log.debug(f"📊 Combined stats: min={combined_features.min():.3f}, "
                 f"max={combined_features.max():.3f}, mean={combined_features.mean():.3f}")
        
        # Log feature composition
        log.debug(f"🧩 Feature composition: {v_feats.shape[0]} VisiCalc + {m_feats.shape[0]} dynamic = {combined_features.shape[0]} total")
        
        return combined_features
        
    except Exception as e:
        log.error(f"❌ Failed to concatenate features: {e}")
        log.error(f"   VisiCalc features shape: {v_feats.shape}, dtype: {v_feats.dtype}")
        log.error(f"   Dynamic features shape: {m_feats.shape}, dtype: {m_feats.dtype}")
        raise


def build_dynamic_feature_vectors_batch(
    visicalc_reports: List[dict],
    metrics_list: List[Dict[str, float]],
    metric_names: List[str],
    problem_ids: Optional[List[str]] = None
) -> np.ndarray:
    """
    Build dynamic feature vectors for a batch of examples.
    
    Args:
        visicalc_reports: List of VisiCalc report dicts
        metrics_list: List of metric dictionaries
        metric_names: Ordered list of metric keys
        problem_ids: Optional list of problem identifiers for logging
        
    Returns:
        np.ndarray: Batch of feature vectors [batch_size, 8 + len(metric_names)]
    """
    log.info(f"🏭 Building batch of {len(visicalc_reports)} dynamic feature vectors")
    log.info(f"🎯 Using {len(metric_names)} core metrics")
    
    if problem_ids is None:
        problem_ids = [f"example_{i}" for i in range(len(visicalc_reports))]
    
    if len(visicalc_reports) != len(metrics_list):
        log.error(f"❌ Mismatched batch sizes: {len(visicalc_reports)} reports vs {len(metrics_list)} metrics")
        raise ValueError("visicalc_reports and metrics_list must have same length")
    
    if len(visicalc_reports) != len(problem_ids):
        log.error(f"❌ Mismatched batch sizes: {len(visicalc_reports)} reports vs {len(problem_ids)} problem_ids")
        raise ValueError("visicalc_reports and problem_ids must have same length")
    
    batch_features = []
    successful = 0
    failed = 0
    
    for i, (report, metrics, problem_id) in enumerate(zip(visicalc_reports, metrics_list, problem_ids)):
        try:
            features = build_dynamic_feature_vector(
                visicalc_report=report,
                metrics=metrics,
                metric_names=metric_names,
                problem_id=problem_id
            )
            batch_features.append(features)
            successful += 1
            
            # Log progress for large batches
            if (i + 1) % 50 == 0:
                log.info(f"📦 Processed {i + 1}/{len(visicalc_reports)} examples")
                
        except Exception as e:
            log.error(f"❌ Failed to build features for {problem_id}: {e}")
            failed += 1
            # Create zero vector as fallback
            fallback_features = np.zeros(8 + len(metric_names), dtype=np.float32)
            batch_features.append(fallback_features)
    
    if batch_features:
        batch_array = np.stack(batch_features, axis=0)
        log.info(f"✅ Batch feature construction complete: {successful} successful, {failed} failed")
        log.info(f"📦 Final batch shape: {batch_array.shape}")
        log.info(f"📊 Batch stats - min: {batch_array.min():.3f}, max: {batch_array.max():.3f}, "
                f"mean: {batch_array.mean():.3f}, std: {batch_array.std():.3f}")
        return batch_array
    else:
        log.error("❌ No features were successfully built")
        raise ValueError("Failed to build any feature vectors")


def validate_core_metrics_coverage(
    metrics_list: List[Dict[str, float]],
    metric_names: List[str],
    sample_size: int = 10
) -> Dict[str, float]:
    """
    Validate how many core metrics are actually available in the data.
    
    Returns coverage statistics for debugging data quality issues.
    """
    log.info("🔍 Validating core metrics coverage in dataset...")
    
    if not metrics_list:
        log.warning("⚠️  No metrics data provided for validation")
        return {}
    
    # Sample a subset for analysis
    sample_metrics = metrics_list[:sample_size] if len(metrics_list) > sample_size else metrics_list
    
    coverage_stats = {}
    total_possible = len(metric_names) * len(sample_metrics)
    total_found = 0
    
    for metric_name in metric_names:
        metric_found = 0
        for metrics_dict in sample_metrics:
            if metric_name in metrics_dict:
                try:
                    float(metrics_dict[metric_name])
                    metric_found += 1
                    total_found += 1
                except (ValueError, TypeError):
                    pass
        
        coverage = metric_found / len(sample_metrics)
        coverage_stats[metric_name] = coverage
        
        if coverage < 0.5:
            log.debug(f"⚠️  Low coverage for {metric_name}: {coverage:.1%}")
    
    overall_coverage = total_found / total_possible if total_possible > 0 else 0
    log.info(f"📊 Core metrics coverage: {overall_coverage:.1%} "
            f"({total_found}/{total_possible} expected metrics found)")
    
    # Log best and worst covered metrics
    if coverage_stats:
        sorted_coverage = sorted(coverage_stats.items(), key=lambda x: x[1], reverse=True)
        best_metrics = sorted_coverage[:3]
        worst_metrics = sorted_coverage[-3:] if len(sorted_coverage) >= 3 else sorted_coverage
        
        log.info(f"🏆 Best covered metrics: {', '.join([f'{name}({cov:.1%})' for name, cov in best_metrics])}")
        log.info(f"🔻 Worst covered metrics: {', '.join([f'{name}({cov:.1%})' for name, cov in worst_metrics])}")
    
    return coverage_stats


def extract_tiny_features(report):
    # 1. Overall stability (less variation = better)
    region_values = [region["mean_frontier_value"] for region in report["regions"]]
    stability = np.std(region_values)
    
    # 2. Middle region dip (more negative = worse)
    middle_dip = region_values[2] - min(region_values[0], region_values[3])
    
    # 3. Standard deviation (lower = better consistency)
    std_dev = report["global"]["std"]
    
    # 4. Sparsity (lower = more meaningful signal)
    sparsity = report["global"]["sparsity_level_e3"]
    
    # 5. Entropy (higher = more diverse reasoning)
    entropy = report["global"]["entropy"]
    
    # 6. Trend pattern (positive = improving)
    trend = region_values[-1] - region_values[0]
    
    # 7. Mid-bad ratio (how much worse is middle vs average)
    mid_bad_ratio = region_values[2] / np.mean(region_values)
    
    # 8. Frontier band utilization (even if 0, the pattern matters)
    frontier_util = report["global"]["frontier_frac"]
    
    return np.array([stability, middle_dip, std_dev, sparsity, 
                    entropy, trend, mid_bad_ratio, frontier_util])