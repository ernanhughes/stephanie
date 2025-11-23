import numpy as np

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