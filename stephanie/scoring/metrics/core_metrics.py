# stephanie/scoring/metrics/core_metrics.py
from __future__ import annotations

CORE_METRIC_MAPPING = {
    # VisiCalc core metrics
    "frontier_util": "frontier_util",
    "stability": "stability",
    "middle_dip": "middle_dip",
    "std_dev": "std_dev",
    "sparsity": "sparsity",
    "entropy": "entropy",
    "trend": "trend",
    
    # Scorer aggregate metrics
    "hrm.aggregate": "HRM.aggregate",
    "sicql.aggregate": "SICQL.aggregate",
    "tiny.aggregate": "Tiny.aggregate",
    
    # Additional critical metrics
    "verification_present": "Verification.present",
    "step_count": "Reasoning.steps"
}