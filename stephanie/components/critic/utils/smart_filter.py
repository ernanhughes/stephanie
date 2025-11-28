# stephanie/components/critic/utils/smart_filter.py
from __future__ import annotations

from typing import List
import numpy as np
import logging

log = logging.getLogger(__name__)

def get_frontier_relevant_metrics(
    self,
    metric_matrix: np.ndarray,
    metric_names: List[str],
    frontier_metric: str,
    frontier_low: float,
    frontier_high: float,
    target_size: int = 8
) -> List[str]:
    """
    Identify metrics most relevant to the frontier band (not just good vs bad).
    
    This is the "SVM for reasoning" special sauce.
    
    Args:
        metric_matrix: N×M matrix of metric values
        metric_names: Names of all metrics
        frontier_metric: The selected frontier metric
        frontier_low: Lower bound of frontier band
        frontier_high: Upper bound of frontier band
        target_size: Target number of metrics to keep
    
    Returns:
        List of frontier-relevant metric names
    """
    log.info(f"🔍 Identifying frontier-relevant metrics (target size={target_size})...")
    
    # 1. Get frontier metric index and values
    frontier_idx = metric_names.index(frontier_metric)
    frontier_values = metric_matrix[:, frontier_idx]
    
    # 2. Create frontier band labels:
    # 0 = clearly bad (below frontier_low)
    # 1 = in frontier band (between low and high)
    # 2 = clearly good (above frontier_high)
    frontier_labels = np.zeros(len(frontier_values))
    frontier_labels[(frontier_values >= frontier_low) & (frontier_values <= frontier_high)] = 1
    frontier_labels[frontier_values > frontier_high] = 2
    
    # 3. Focus on distinguishing frontier band from others
    y_frontier = (frontier_labels == 1).astype(int)
    
    # 4. Calculate metric relevance to frontier band
    metric_relevance = []
    for i, metric_name in enumerate(metric_names):
        if metric_name == frontier_metric:
            continue  # Skip the frontier metric itself
        
        metric_values = metric_matrix[:, i]
        
        # Calculate AUC for identifying frontier band
        auc = self._calculate_auc(metric_values, y_frontier)
        
        # Calculate Cohen's d for frontier band separation
        cohens_d = self._calculate_cohens_d(metric_values, y_frontier)
        
        # Composite relevance score
        relevance = 0.6 * auc + 0.4 * cohens_d
        metric_relevance.append((metric_name, relevance, auc, cohens_d))
    
    # 5. Select top metrics
    metric_relevance.sort(key=lambda x: x[1], reverse=True)
    kept_metrics = [m[0] for m in metric_relevance[:target_size]]
    
    log.info(f"   Frontier-relevant metrics: {', '.join(kept_metrics)}")
    return kept_metrics