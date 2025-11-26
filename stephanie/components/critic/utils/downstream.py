# stephanie/components/critic/utils/downstream.py
from __future__ import annotations

import os
from typing import Callable, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def compute_downstream_impact(
    y_true: np.ndarray,
    critic_scores: np.ndarray,
    accuracy_func: Callable[[np.ndarray], float],
    budget_levels: List[float] = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5],
    n_simulations: int = 100
) -> Dict[str, List[float]]:
    """
    Compute downstream impact of using critic scores for selection.
    
    Args:
        y_true: Ground truth labels (1=correct, 0=incorrect)
        critic_scores: Critic scores (higher = better quality)
        accuracy_func: Function that computes accuracy given selected indices
        budget_levels: Fraction of total examples to select
        n_simulations: Number of simulations for random selection
        
    Returns:
        Dictionary with impact metrics
    """
    n = len(y_true)
    results = {
        "budget_levels": budget_levels,
        "critic_accuracies": [],
        "random_accuracies_mean": [],
        "random_accuracies_std": [],
        "top_p_accuracies": [],
        "improvement_over_random": [],
        "improvement_over_top_p": []
    }
    
    # Sort by critic score (highest first)
    sorted_idx = np.argsort(critic_scores)[::-1]
    
    for budget in budget_levels:
        k = max(1, int(n * budget))  # Ensure at least 1 sample
        
        # Critic-based selection (top-k by critic score)
        critic_idx = sorted_idx[:k]
        critic_acc = accuracy_func(critic_idx)
        results["critic_accuracies"].append(float(critic_acc))
        
        # Random selection (average over n_simulations)
        random_accs = []
        for _ in range(n_simulations):
            if k > 0:
                random_idx = np.random.choice(n, k, replace=False)
                random_accs.append(accuracy_func(random_idx))
            else:
                random_accs.append(0.0)
        results["random_accuracies_mean"].append(float(np.mean(random_accs)))
        results["random_accuracies_std"].append(float(np.std(random_accs)))
        
        # Top-p selection (for comparison, if applicable)
        # This would be based on the model's own confidence, not the critic
        # For now, we'll simulate it as random within the top-p
        top_p_accs = []
        for _ in range(n_simulations):
            # Simulate top-p selection (e.g., take top 50% of model's confidence)
            top_p_idx = np.random.choice(sorted_idx[:int(n*0.5)], k, replace=False)
            top_p_accs.append(accuracy_func(top_p_idx))
        results["top_p_accuracies"].append(float(np.mean(top_p_accs)))
    
    # Calculate improvements
    for i, budget in enumerate(budget_levels):
        results["improvement_over_random"].append(
            results["critic_accuracies"][i] - results["random_accuracies_mean"][i]
        )
        results["improvement_over_top_p"].append(
            results["critic_accuracies"][i] - results["top_p_accuracies"][i]
        )
    
    return results

def generate_downstream_plot(
    results: Dict[str, List[float]],
    output_path: str,
    title: str = "Downstream Impact of Critic Selection"
) -> None:
    """
    Generate and save a downstream impact plot.
    
    Args:
        results: Results from compute_downstream_impact
        output_path: Path to save the plot
        title: Plot title
    """
    plt.figure(figsize=(10, 7))
    sns.set_style("whitegrid")
    
    budget_levels = [b * 100 for b in results["budget_levels"]]
    
    # Plot critic accuracy
    plt.plot(budget_levels, results["critic_accuracies"], "o-", 
             linewidth=2.5, markersize=8, label="Critic Selection")
    
    # Plot random selection with error bars
    plt.errorbar(
        budget_levels, 
        results["random_accuracies_mean"],
        yerr=results["random_accuracies_std"],
        fmt="s-", 
        linewidth=2, 
        markersize=7,
        label="Random Selection"
    )
    
    # Plot top-p selection
    plt.plot(budget_levels, results["top_p_accuracies"], "^-", 
             linewidth=2, markersize=7, label="Top-p Selection")
    
    plt.xlabel("Budget (% of Total Examples)")
    plt.ylabel("Task Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    
    # Add value labels for critic selection
    for i, acc in enumerate(results["critic_accuracies"]):
        plt.annotate(f"{acc:.2f}", 
                    (budget_levels[i], acc),
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_lift_curve(
    y_true: np.ndarray,
    critic_scores: np.ndarray,
    output_path: str,
    title: str = "Lift Curve"
) -> None:
    """
    Generate and save a lift curve.
    
    Args:
        y_true: Ground truth labels
        critic_scores: Critic scores
        output_path: Path to save the plot
        title: Plot title
    """
    n = len(y_true)
    
    # Sort by critic score (highest first)
    sorted_idx = np.argsort(critic_scores)[::-1]
    y_sorted = y_true[sorted_idx]
    
    # Calculate cumulative positives
    cum_positives = np.cumsum(y_sorted)
    total_positives = y_sorted.sum()
    
    # Calculate fraction reviewed
    fractions = np.arange(1, n + 1) / n
    
    # Calculate cumulative gain
    cumulative_gain = cum_positives / total_positives
    
    # Calculate random selection baseline
    random_gain = fractions.copy()
    
    # Calculate lift
    lift = cumulative_gain / fractions
    
    plt.figure(figsize=(10, 7))
    sns.set_style("whitegrid")
    
    # Plot gain curve
    plt.plot(fractions, cumulative_gain, "o-", 
             linewidth=2.5, markersize=4, label="Critic Selection")
    plt.plot(fractions, random_gain, "--", 
             linewidth=2, label="Random Selection")
    
    plt.xlabel("Fraction of Examples Reviewed")
    plt.ylabel("Cumulative Fraction of Positives Captured")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()




def run_downstream_experiment(
    y_true: np.ndarray,
    probs: np.ndarray,
    accuracy_func: Callable[[np.ndarray], float],
    budget_levels: List[float] = [0.05, 0.1, 0.2, 0.5],
    output_dir: str = "reports/downstream"
) -> Dict[str, any]:
    """
    Evaluate how critic scores improve downstream task performance
    
    Args:
        y_true: Ground truth correctness (1=correct, 0=incorrect)
        probs: Critic scores (probability of being correct)
        accuracy_func: Function that computes accuracy given selected indices
        budget_levels: Fraction of total examples to select
        output_dir: Output directory for reports
    
    Returns:
        Dictionary with evaluation results
    """
    os.makedirs(output_dir, exist_ok=True)
    n = len(y_true)
    
    # Sort by critic score (highest first)
    sorted_idx = np.argsort(probs)[::-1]
    
    # Calculate baseline (random selection)
    baseline_accuracies = []
    for budget in budget_levels:
        k = int(n * budget)
        # Simulate random selection by shuffling
        random_idx = np.random.permutation(n)[:k]
        baseline_accuracies.append(accuracy_func(random_idx))
    
    # Calculate critic-based selection
    critic_accuracies = []
    for i, budget in enumerate(budget_levels):
        k = int(n * budget)
        selected_idx = sorted_idx[:k]
        critic_accuracies.append(accuracy_func(selected_idx))
    
    # Calculate improvement
    improvements = [
        critic_accuracies[i] - baseline_accuracies[i] 
        for i in range(len(budget_levels))
    ]
    
    # Generate visualization
    plt.figure(figsize=(10, 6))
    x = np.arange(len(budget_levels))
    width = 0.35
    
    plt.bar(x - width/2, baseline_accuracies, width, label="Random Selection")
    plt.bar(x + width/2, critic_accuracies, width, label="Critic Selection")
    
    plt.xlabel("Budget (Fraction of Total Examples)")
    plt.ylabel("Task Accuracy")
    plt.title("Downstream Task Improvement")
    plt.xticks(x, [f"{int(b*100)}%" for b in budget_levels])
    plt.legend()
    
    # Add value labels
    for i, v in enumerate(baseline_accuracies):
        plt.text(i - width/2, v + 0.02, f"{v:.2f}", ha='center')
    for i, v in enumerate(critic_accuracies):
        plt.text(i + width/2, v + 0.02, f"{v:.2f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/downstream_improvement.png")
    plt.close()
    
    # Prepare report data
    report_data = {
        "budget_levels": [float(b) for b in budget_levels],
        "baseline_accuracies": [float(a) for a in baseline_accuracies],
        "critic_accuracies": [float(a) for a in critic_accuracies],
        "improvements": [float(i) for i in improvements],
        "visualization_path": "downstream_improvement.png"
    }
    
    # Generate markdown report
    md_content = f"""# Downstream Task Improvement Report

## Experimental Setup
- Total examples: {n}
- Budget levels tested: {', '.join([f'{int(b*100)}%' for b in budget_levels])}

## Downstream Impact Notes"

- Budget levels < 1% may show 0.0 accuracy due to insufficient samples
- Empty selections are reported as 0.0 accuracy for consistency

## Results

| Budget | Random Selection | Critic Selection | Improvement |
|--------|------------------|------------------|-------------|
"""
    
    for i, budget in enumerate(budget_levels):
        md_content += f"| {int(budget*100)}% | {baseline_accuracies[i]:.2f} | **{critic_accuracies[i]:.2f}** | **+{improvements[i]:.2f}** |\n"
    
    md_content += "\n## Key Finding\n"
    md_content += f"The critic improves task accuracy by **{max(improvements):.2f}** at the {int(budget_levels[np.argmax(improvements)]*100)}% budget level.\n\n"
    md_content += "![Downstream Improvement](downstream_improvement.png)\n"
    
    with open(f"{output_dir}/downstream_report.md", "w") as f:
        f.write(md_content)
    
    return report_data

def example_gsm8k_accuracy(selected_idx: np.ndarray, correct_answers: np.ndarray) -> float:
    """Example accuracy function for GSM8K problems"""
    return correct_answers[selected_idx].mean()