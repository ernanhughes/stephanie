# stephanie/components/critic/utils/statistics.py
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_auc_score


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return float("nan")

def paired_bootstrap_auc_diff(
    y_true: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    *,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: Optional[int] = 42,
    stratified: bool = True,
) -> Dict[str, float]:
    """
    Paired bootstrap test for AUROC difference (B - A).
    Returns ΔAUC, CI, two-tailed p-value, and win_rate.
    Seeds are supported for reproducibility.

    stratified=True keeps class balance in each bootstrap sample.
    """
    y_true = np.asarray(y_true).astype(int)
    pa = np.asarray(probs_a, dtype=float)
    pb = np.asarray(probs_b, dtype=float)

    n = len(y_true)
    assert pa.shape == pb.shape == (n,), "Probability vectors must be 1-D and same length as y_true"

    rng = np.random.default_rng(seed)

    auc_a = _safe_auc(y_true, pa)
    auc_b = _safe_auc(y_true, pb)
    delta = auc_b - auc_a

    # Indices for stratified resampling
    if stratified:
        pos_idx = np.where(y_true == 1)[0]
        neg_idx = np.where(y_true == 0)[0]
        n_pos = len(pos_idx)
        n_neg = len(neg_idx)

    deltas = np.empty(n_boot, dtype=float)
    wins = 0

    for i in range(n_boot):
        if stratified and n_pos > 0 and n_neg > 0:
            bs_pos = rng.choice(pos_idx, size=n_pos, replace=True)
            bs_neg = rng.choice(neg_idx, size=n_neg, replace=True)
            bs_idx = np.concatenate([bs_pos, bs_neg])
        else:
            bs_idx = rng.choice(n, size=n, replace=True)

        ya = y_true[bs_idx]
        aa = pa[bs_idx]
        bb = pb[bs_idx]

        a_auc = _safe_auc(ya, aa)
        b_auc = _safe_auc(ya, bb)
        d = b_auc - a_auc
        deltas[i] = d
        wins += (d > 0)

    # Percentile CI & p-value from bootstrap distribution
    lo = float(np.percentile(deltas, 100 * alpha / 2))
    hi = float(np.percentile(deltas, 100 * (1 - alpha / 2)))
    # Two-tailed p: probability of observing delta as/extreme as observed
    # relative to bootstrap centered at 0. Use symmetric tail mass.
    p_two = 2 * min(
        float(np.mean(deltas >= delta)),
        float(np.mean(deltas <= delta))
    )
    p_two = min(1.0, max(0.0, p_two))

    return {
        "auc_a": float(auc_a),
        "auc_b": float(auc_b),
        "delta": float(delta),
        "ci_low": lo,
        "ci_high": hi,
        "p_value": p_two,
        "win_rate": float(wins / n_boot),
        "n_boot": int(n_boot),
        "alpha": float(alpha),
        "seed": int(seed) if seed is not None else None,
    }


def paired_bootstrap(
    y_true: np.ndarray,
    y_pred1: np.ndarray,
    y_pred2: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 5000,
    alpha: float = 0.05
) -> Dict[str, float]:
    """
    Compute confidence interval for metric difference using paired bootstrap.
    
    Args:
        y_true: Ground truth labels
        y_pred1: Predictions from model 1
        y_pred2: Predictions from model 2
        metric_func: Function that computes the metric (e.g., roc_auc_score)
        n_bootstrap: Number of bootstrap samples
        alpha: Significance level for confidence interval
        
    Returns:
        Dictionary containing mean difference, confidence interval, p-value
    """
    n = len(y_true)
    metric_diffs = []
    
    for _ in range(n_bootstrap):
        # Sample with replacement, same indices for both models
        idx = np.random.choice(n, n, replace=True)
        metric1 = metric_func(y_true[idx], y_pred1[idx])
        metric2 = metric_func(y_true[idx], y_pred2[idx])
        metric_diffs.append(metric2 - metric1)
    
    # Calculate confidence interval
    ci_lower = np.percentile(metric_diffs, 100 * (alpha/2))
    ci_upper = np.percentile(metric_diffs, 100 * (1 - alpha/2))
    
    # Calculate p-value (two-tailed)
    p_value = 2 * min(
        np.mean(np.array(metric_diffs) >= 0),
        np.mean(np.array(metric_diffs) <= 0)
    )
    
    return {
        "mean_diff": float(np.mean(metric_diffs)),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "p_value": float(p_value),
        "n_bootstrap": n_bootstrap,
        "significant": p_value < alpha and ci_lower * ci_upper > 0
    }

def compute_ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error.
    
    Args:
        probs: Predicted probabilities
        y_true: Ground truth labels
        n_bins: Number of bins for calibration
        
    Returns:
        ECE value
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculate bin's confidence and accuracy
        in_bin = (probs >= bin_lower) & (probs < bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = probs[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece

def generate_calibration_plot(
    y_true: np.ndarray,
    probs: np.ndarray,
    output_path: str,
    n_bins: int = 10,
    title: str = "Reliability Diagram"
) -> None:
    """
    Generate and save a calibration plot.
    
    Args:
        y_true: Ground truth labels
        probs: Predicted probabilities
        output_path: Path to save the plot
        n_bins: Number of bins for calibration
        title: Plot title
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    # Calculate bin metrics
    accuracies = []
    confidences = []
    bin_counts = []
    
    for bl, bu in zip(bin_lowers, bin_uppers):
        in_bin = (probs >= bl) & (probs < bu)
        if in_bin.sum() > 0:
            accuracy = y_true[in_bin].mean()
            confidence = probs[in_bin].mean()
            count = in_bin.sum()
        else:
            accuracy = np.nan
            confidence = (bl + bu) / 2
            count = 0
            
        accuracies.append(accuracy)
        confidences.append(confidence)
        bin_counts.append(count)
    
    # Create plot
    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")
    
    # Plot calibration curve
    plt.plot(confidences, accuracies, "s-", label="Model")
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
    
    # Add error bars based on bin counts
    bin_sizes = np.array(bin_counts) / sum(bin_counts)
    plt.errorbar(
        confidences, 
        accuracies, 
        yerr=[0.1 * size for size in bin_sizes],  # Simplified error representation
        fmt='none', 
        ecolor='gray', 
        alpha=0.5
    )
    
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()