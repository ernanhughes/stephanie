# stephanie/components/critic/utils/downstream_evaluator.py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable
import matplotlib.pyplot as plt
import os

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

## Results

| Budget | Random Selection | Critic Selection | Improvement |
|--------|------------------|------------------|-------------|
"""
    
    for i, budget in enumerate(budget_levels):
        md_content += f"| {int(budget*100)}% | {baseline_accuracies[i]:.2f} | **{critic_accuracies[i]:.2f}** | **+{improvements[i]:.2f}** |\n"
    
    md_content += f"\n## Key Finding\n"
    md_content += f"The critic improves task accuracy by **{max(improvements):.2f}** at the {int(budget_levels[np.argmax(improvements)]*100)}% budget level.\n\n"
    md_content += "![Downstream Improvement](downstream_improvement.png)\n"
    
    with open(f"{output_dir}/downstream_report.md", "w") as f:
        f.write(md_content)
    
    return report_data

def example_gsm8k_accuracy(selected_idx: np.ndarray, correct_answers: np.ndarray) -> float:
    """Example accuracy function for GSM8K problems"""
    return correct_answers[selected_idx].mean()