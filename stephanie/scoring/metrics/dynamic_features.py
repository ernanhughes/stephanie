# stephanie/scoring/metrics/dynamic_features.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from stephanie.scoring.metrics.visicalc_features import extract_tiny_features

CORE_METRICS_PATH = Path("config/metrics/core_metrics.json")


def load_core_metric_names(path: Path = CORE_METRICS_PATH) -> List[str]:
    """Load the ordered list of dynamic metric feature names."""
    cfg = json.loads(path.read_text(encoding="utf-8"))
    return list(cfg.get("metric_features", []))


def build_dynamic_feature_vector(
    visicalc_report: dict,
    metrics: Dict[str, float],
    metric_names: List[str],
) -> np.ndarray:
    """
    Build [VisiCalc 8] + [dynamic metrics] feature vector.

    - visicalc_report: full VisiCalc report dict
    - metrics: dict of metric_name -> value (from HRM/SICQL/Tiny/etc.)
    - metric_names: ordered list of metric keys to pull from metrics

    Missing metrics are filled with 0.0.
    """
    # 1) VisiCalc structural features (8 dims)
    v_feats = extract_tiny_features(visicalc_report).astype(np.float32)

    # 2) Dynamic metric features (N dims)
    m_vals = []
    for name in metric_names:
        v = metrics.get(name, 0.0)
        try:
            m_vals.append(float(v))
        except Exception:
            m_vals.append(0.0)
    m_feats = np.asarray(m_vals, dtype=np.float32)

    # 3) Concatenate → [8 + N]
    return np.concatenate([v_feats, m_feats], axis=-1)
