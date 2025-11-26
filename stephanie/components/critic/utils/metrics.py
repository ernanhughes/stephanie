# stephanie/components/critic/utils/critic_metrics.py
import json
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import (brier_score_loss, precision_recall_curve,
                             precision_score, roc_auc_score, roc_curve)


def compute_ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error"""
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

def lift_at_k(y_true: np.ndarray, probs: np.ndarray, k: float = 0.1) -> float:
    """Compute precision at top-k% of predictions"""
    n = len(y_true)
    k_count = int(n * k)
    
    # Get indices of top-k predictions
    top_k_idx = np.argsort(probs)[-k_count:]
    
    # Calculate precision on these examples
    return y_true[top_k_idx].mean()

def gain_curve(y_true: np.ndarray, probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute cumulative gain curve"""
    # Sort by prediction score
    sorted_idx = np.argsort(probs)[::-1]
    y_sorted = y_true[sorted_idx]
    
    # Calculate cumulative positives
    cum_positives = np.cumsum(y_sorted)
    total_positives = y_sorted.sum()
    
    # Calculate fraction reviewed
    fractions = np.arange(1, len(y_sorted) + 1) / len(y_sorted)
    
    # Calculate cumulative gain
    cumulative_gain = cum_positives / total_positives
    
    return fractions, cumulative_gain

def paired_bootstrap(
    y_true: np.ndarray, 
    probs1: np.ndarray, 
    probs2: np.ndarray,
    metric_func,
    n_bootstrap: int = 5000,
    alpha: float = 0.05
) -> Dict[str, float]:
    """Compute confidence interval for metric difference using paired bootstrap"""
    n = len(y_true)
    metric_diffs = []
    
    for _ in range(n_bootstrap):
        # Sample with replacement, same indices for both models
        idx = np.random.choice(n, n, replace=True)
        metric1 = metric_func(y_true[idx], probs1[idx])
        metric2 = metric_func(y_true[idx], probs2[idx])
        metric_diffs.append(metric2 - metric1)
    
    # Calculate confidence interval
    ci_lower = np.percentile(metric_diffs, 100 * (alpha/2))
    ci_upper = np.percentile(metric_diffs, 100 * (1 - alpha/2))
    p_value = 2 * min(
        np.mean(np.array(metric_diffs) >= 0),
        np.mean(np.array(metric_diffs) <= 0)
    )
    
    return {
        "mean_diff": np.mean(metric_diffs),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
        "n_bootstrap": n_bootstrap
    }

def win_rate(y_true: np.ndarray, probs1: np.ndarray, probs2: np.ndarray) -> float:
    """Calculate fraction where model2 gives higher probability to true class"""
    # For positive examples (y=1), higher prob is better
    # For negative examples (y=0), lower prob is better (higher 1-prob)
    better = np.zeros_like(y_true, dtype=bool)
    better[y_true == 1] = probs2[y_true == 1] > probs1[y_true == 1]
    better[y_true == 0] = (1 - probs2[y_true == 0]) > (1 - probs1[y_true == 0])
    
    return better.mean()

def generate_evaluation_report(
    y_true: np.ndarray,
    probs_current: np.ndarray,
    probs_candidate: np.ndarray,
    feature_names: List[str],
    output_dir: str = "reports/evaluation"
) -> Dict[str, any]:
    """Generate comprehensive evaluation report with metrics and visualizations"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Compute basic metrics
    metrics = {
        "current": {
            "auroc": roc_auc_score(y_true, probs_current),
            "brier": brier_score_loss(y_true, probs_current),
            "ece": compute_ece(probs_current, y_true),
            "lift_5": lift_at_k(y_true, probs_current, 0.05),
            "lift_10": lift_at_k(y_true, probs_current, 0.10),
            "lift_20": lift_at_k(y_true, probs_current, 0.20)
        },
        "candidate": {
            "auroc": roc_auc_score(y_true, probs_candidate),
            "brier": brier_score_loss(y_true, probs_candidate),
            "ece": compute_ece(probs_candidate, y_true),
            "lift_5": lift_at_k(y_true, probs_candidate, 0.05),
            "lift_10": lift_at_k(y_true, probs_candidate, 0.10),
            "lift_20": lift_at_k(y_true, probs_candidate, 0.20)
        }
    }
    
    # 2. Compute statistical significance
    def auroc_func(y, p):
        return roc_auc_score(y, p)
    
    def brier_func(y, p):
        return -brier_score_loss(y, p)  # Negative because higher is better
    
    auroc_test = paired_bootstrap(y_true, probs_current, probs_candidate, auroc_func)
    brier_test = paired_bootstrap(y_true, probs_current, probs_candidate, brier_func)
    win_rate_val = win_rate(y_true, probs_current, probs_candidate)
    
    # 3. Generate visualizations
    # ROC Curve
    plt.figure(figsize=(8, 6))
    fpr_cur, tpr_cur, _ = roc_curve(y_true, probs_current)
    fpr_cand, tpr_cand, _ = roc_curve(y_true, probs_candidate)
    
    plt.plot(fpr_cur, tpr_cur, label=f'Current (AUROC = {metrics["current"]["auroc"]:.3f})')
    plt.plot(fpr_cand, tpr_cand, label=f'Candidate (AUROC = {metrics["candidate"]["auroc"]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend()
    plt.savefig(f"{output_dir}/roc_curve.png")
    plt.close()
    
    # Reliability Curve (Calibration)
    plt.figure(figsize=(8, 6))
    bin_boundaries = np.linspace(0, 1, 11)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    # Current model
    current_acc = []
    current_conf = []
    for bl, bu in zip(bin_lowers, bin_uppers):
        in_bin = (probs_current >= bl) & (probs_current < bu)
        if in_bin.sum() > 0:
            current_acc.append(y_true[in_bin].mean())
            current_conf.append(probs_current[in_bin].mean())
    
    # Candidate model
    candidate_acc = []
    candidate_conf = []
    for bl, bu in zip(bin_lowers, bin_uppers):
        in_bin = (probs_candidate >= bl) & (probs_candidate < bu)
        if in_bin.sum() > 0:
            candidate_acc.append(y_true[in_bin].mean())
            candidate_conf.append(probs_candidate[in_bin].mean())
    
    plt.plot(current_conf, current_acc, "s-", label=f"Current (ECE={metrics['current']['ece']:.3f})")
    plt.plot(candidate_conf, candidate_acc, "o-", label=f"Candidate (ECE={metrics['candidate']['ece']:.3f})")
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Reliability Diagram")
    plt.legend()
    plt.savefig(f"{output_dir}/calibration.png")
    plt.close()
    
    # Gain Curve
    fractions, current_gain = gain_curve(y_true, probs_current)
    _, candidate_gain = gain_curve(y_true, probs_candidate)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fractions, current_gain, label="Current")
    plt.plot(fractions, candidate_gain, label="Candidate")
    plt.plot([0, 1], [0, 1], "k--", label="Random Selection")
    plt.xlabel("Fraction of Examples Reviewed")
    plt.ylabel("Cumulative Fraction of Positives Captured")
    plt.title("Gain Curve")
    plt.legend()
    plt.savefig(f"{output_dir}/gain_curve.png")
    plt.close()
    
    # Lift at k
    plt.figure(figsize=(8, 6))
    k_values = [5, 10, 20]
    current_lifts = [metrics["current"][f"lift_{k}"] for k in k_values]
    candidate_lifts = [metrics["candidate"][f"lift_{k}"] for k in k_values]
    
    x = np.arange(len(k_values))
    width = 0.35
    plt.bar(x - width/2, current_lifts, width, label="Current")
    plt.bar(x + width/2, candidate_lifts, width, label="Candidate")
    
    plt.xlabel("Top-k% Examples")
    plt.ylabel("Precision")
    plt.title("Lift at k")
    plt.xticks(x, [f"{k}%" for k in k_values])
    plt.legend()
    
    # Add value labels
    for i, v in enumerate(current_lifts):
        plt.text(i - width/2, v + 0.02, f"{v:.2f}", ha='center')
    for i, v in enumerate(candidate_lifts):
        plt.text(i + width/2, v + 0.02, f"{v:.2f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/lift_at_k.png")
    plt.close()
    
    # 4. Prepare report data
    report_data = {
        "metrics": metrics,
        "statistical_tests": {
            "auroc": auroc_test,
            "brier": brier_test,
            "win_rate": win_rate_val
        },
        "visualization_paths": {
            "roc_curve": "roc_curve.png",
            "calibration": "calibration.png",
            "gain_curve": "gain_curve.png",
            "lift_at_k": "lift_at_k.png"
        },
        "feature_names": feature_names,
        "n_samples": len(y_true),
        "positive_rate": y_true.mean()
    }
    
    # 5. Save JSON report
    with open(f"{output_dir}/evaluation_report.json", "w") as f:
        json.dump(report_data, f, indent=2)
    
    # 6. Generate markdown report
    md_content = f"""# Critic Evaluation Report

## Summary
- **Total samples**: {len(y_true)}
- **Positive rate**: {y_true.mean():.2%}
- **Feature count**: {len(feature_names)}

## Headline Metrics

| Metric | Current | Candidate | Δ | 95% CI | p-value | Significant |
|--------|---------|-----------|----|--------|---------|-------------|
| AUROC | {metrics['current']['auroc']:.3f} | **{metrics['candidate']['auroc']:.3f}** | **{auroc_test['mean_diff']:.3f}** | [{auroc_test['ci_lower']:.3f}, {auroc_test['ci_upper']:.3f}] | {auroc_test['p_value']:.4f} | {'✅' if auroc_test['p_value'] < 0.05 and auroc_test['ci_lower'] > 0 else '❌'} |
| Brier Score ↓ | {metrics['current']['brier']:.3f} | **{metrics['candidate']['brier']:.3f}** | **{brier_test['mean_diff']:.3f}** | [{brier_test['ci_lower']:.3f}, {brier_test['ci_upper']:.3f}] | {brier_test['p_value']:.4f} | {'✅' if brier_test['p_value'] < 0.05 and brier_test['ci_upper'] < 0 else '❌'} |
| ECE ↓ | {metrics['current']['ece']:.3f} | **{metrics['candidate']['ece']:.3f}** | **{metrics['current']['ece'] - metrics['candidate']['ece']:.3f}** | - | - | - |
| Win Rate | - | **{win_rate_val:.3f}** | **{win_rate_val - 0.5:.3f}** | - | - | {'✅' if win_rate_val > 0.52 else '❌'} |

## Key Visualizations

### ROC Curve
![ROC Curve](roc_curve.png)

### Calibration
![Calibration](calibration.png)

### Gain Curve
![Gain Curve](gain_curve.png)

### Lift at k
![Lift at k](lift_at_k.png)

## Promotion Decision

"""
    
    # Make promotion recommendation
    promote = (
        auroc_test["ci_lower"] > 0 and 
        brier_test["ci_upper"] < 0 and
        win_rate_val > 0.52
    )
    
    if promote:
        md_content += "**PROMOTE** ✅ Candidate model shows statistically significant improvements across multiple metrics.\n"
        md_content += f"- AUROC improvement: +{auroc_test['mean_diff']:.3f} (CI: [{auroc_test['ci_lower']:.3f}, {auroc_test['ci_upper']:.3f}], p={auroc_test['p_value']:.4f})\n"
        md_content += f"- Brier score improvement: {brier_test['mean_diff']:.3f} (CI: [{brier_test['ci_lower']:.3f}, {brier_test['ci_upper']:.3f}], p={brier_test['p_value']:.4f})\n"
        md_content += f"- Win rate: {win_rate_val:.2%} (>{52:.0f}% threshold)\n"
    else:
        md_content += "**DO NOT PROMOTE** ❌ Candidate model does not show consistent, statistically significant improvements.\n"
        reasons = []
        if auroc_test["ci_lower"] <= 0:
            reasons.append(f"AUROC improvement not significant (CI: [{auroc_test['ci_lower']:.3f}, {auroc_test['ci_upper']:.3f}])")
        if brier_test["ci_upper"] >= 0:
            reasons.append(f"Brier score not significantly better (CI: [{brier_test['ci_lower']:.3f}, {brier_test['ci_upper']:.3f}])")
        if win_rate_val <= 0.52:
            reasons.append(f"Win rate too low ({win_rate_val:.1%} ≤ 52%)")
        md_content += "- " + "\n- ".join(reasons)
    
    with open(f"{output_dir}/evaluation_report.md", "w") as f:
        f.write(md_content)
    
    return report_data