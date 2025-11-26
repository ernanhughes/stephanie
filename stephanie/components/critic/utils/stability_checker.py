# stephanie/components/critic/utils/stability_checker.py
from typing import Any, Callable, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score


def check_feature_stability(run_ids: List[str], memory) -> Dict[str, Any]:
    """Check stability of feature selection across multiple runs"""
    feature_sets = []
    
    for run_id in run_ids:
        kept = memory.metrics.get_kept_columns(run_id)
        if kept:
            feature_sets.append(set(kept))
    
    if not feature_sets:
        return {"error": "No feature sets found"}
    
    # Calculate Jaccard similarity between all pairs
    similarities = []
    for i in range(len(feature_sets)):
        for j in range(i+1, len(feature_sets)):
            intersection = len(feature_sets[i] & feature_sets[j])
            union = len(feature_sets[i] | feature_sets[j])
            similarities.append(intersection / union if union > 0 else 1.0)
    
    # Find common features
    common_features = set.intersection(*feature_sets) if feature_sets else set()
    
    return {
        "mean_jaccard": float(np.mean(similarities)) if similarities else 1.0,
        "std_jaccard": float(np.std(similarities)) if similarities else 0.0,
        "common_features_count": len(common_features),
        "common_features": list(common_features),
        "total_runs": len(feature_sets)
    }

def run_ablation_study(
    base_features: List[str],
    X: np.ndarray,
    y: np.ndarray,
    model_factory: Callable[[], Any],
    ablation_groups: Dict[str, List[str]],
    output_dir: str = "reports/ablation"
) -> Dict[str, Any]:
    """Run ablation study by removing feature groups"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    # Train base model
    base_model = model_factory()
    base_model.fit(X, y)
    base_probs = base_model.predict_proba(X)[:, 1]
    base_auroc = roc_auc_score(y, base_probs)
    
    results["base"] = {
        "auroc": float(base_auroc),
        "feature_count": int(X.shape[1])
    }
    
    # Ablate each group
    for group_name, group_features in ablation_groups.items():
        # Create mask for features to keep
        keep_mask = np.array([f not in group_features for f in base_features])
        
        # Project to reduced feature set
        X_ablated = X[:, keep_mask]
        kept_features = [f for f, keep in zip(base_features, keep_mask) if keep]
        
        # Train ablated model
        ablated_model = model_factory()
        ablated_model.fit(X_ablated, y)
        ablated_probs = ablated_model.predict_proba(X_ablated)[:, 1]
        ablated_auroc = roc_auc_score(y, ablated_probs)
        
        # Store results
        results[group_name] = {
            "auroc": float(ablated_auroc),
            "feature_count": int(X_ablated.shape[1]),
            "features_removed": len(group_features),
            "delta_auroc": float(ablated_auroc - base_auroc)
        }
    
    # Generate visualization
    plt.figure(figsize=(10, 6))
    groups = list(results.keys())
    aurocs = [results[g]["auroc"] for g in groups]
    
    plt.bar(groups, aurocs)
    plt.axhline(y=results["base"]["auroc"], color='r', linestyle='-', alpha=0.3)
    plt.xlabel("Model Configuration")
    plt.ylabel("AUROC")
    plt.title("Ablation Study Results")
    
    # Add value labels
    for i, v in enumerate(aurocs):
        plt.text(i, v + 0.01, f"{v:.3f}", ha='center')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ablation_study.png")
    plt.close()
    
    # Generate markdown report
    md_content = f"""# Ablation Study Report

## Experimental Setup
- Base AUROC: {results['base']['auroc']:.3f}
- Total features: {results['base']['feature_count']}

## Results

| Configuration | AUROC | Δ from Base | Features |
|---------------|-------|-------------|----------|
| Base (all features) | {results['base']['auroc']:.3f} | - | {results['base']['feature_count']} |
"""
    
    for group_name, result in results.items():
        if group_name == "base":
            continue
        md_content += f"| Without {group_name} | {result['auroc']:.3f} | {result['delta_auroc']:.3f} | {result['feature_count']} |\n"
    
    md_content += "\n![Ablation Study](ablation_study.png)\n"
    
    with open(f"{output_dir}/ablation_report.md", "w") as f:
        f.write(md_content)
    
    return {
        "results": results,
        "visualization_path": "ablation_study.png"
    }

def label_shuffle_sanity_check(
    X: np.ndarray,
    y: np.ndarray,
    model_factory: Callable[[], Any],
    n_runs: int = 5,
    output_dir: str = "reports/sanity"
) -> Dict[str, Any]:
    """Run label shuffle sanity check to verify no signal leakage"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    original_model = model_factory()
    original_model.fit(X, y)
    original_probs = original_model.predict_proba(X)[:, 1]
    original_auroc = roc_auc_score(y, original_probs)
    
    shuffled_aurocs = []
    for _ in range(n_runs):
        # Shuffle labels
        y_shuffled = np.random.permutation(y)
        
        # Train on shuffled data
        shuffled_model = model_factory()
        shuffled_model.fit(X, y_shuffled)
        shuffled_probs = shuffled_model.predict_proba(X)[:, 1]
        shuffled_auroc = roc_auc_score(y_shuffled, shuffled_probs)
        shuffled_aurocs.append(shuffled_auroc)
    
    # Generate report
    md_content = f"""# Label Shuffle Sanity Check

This test verifies that the model isn't learning from data leakage by training on randomly shuffled labels.

## Results
- Original AUROC: {original_auroc:.3f}
- Mean AUROC with shuffled labels: {np.mean(shuffled_aurocs):.3f}
- Standard deviation: {np.std(shuffled_aurocs):.3f}

## Conclusion
"""
    
    if np.mean(shuffled_aurocs) < 0.55:
        md_content += "✅ The model shows no significant performance with shuffled labels, indicating no major data leakage."
    else:
        md_content += "❌ WARNING: The model shows significant performance with shuffled labels, suggesting potential data leakage."
    
    with open(f"{output_dir}/sanity_check.md", "w") as f:
        f.write(md_content)
    
    return {
        "original_auroc": float(original_auroc),
        "shuffled_aurocs": [float(a) for a in shuffled_aurocs],
        "mean_shuffled_auroc": float(np.mean(shuffled_aurocs)),
        "conclusion": "no leakage" if np.mean(shuffled_aurocs) < 0.55 else "potential leakage"
    }