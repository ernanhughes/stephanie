# stephanie/components/critic/utils/feature_proj.py
from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np

log = logging.getLogger(__name__)


def project_features_to_model(
    X: np.ndarray,
    incoming_names: List[str],
    model_feature_names: List[str],
    strict: bool = True,
    fill_value: float = 0.0,
) -> Tuple[np.ndarray, List[str]]:
    """
    Reorder/align an incoming feature matrix so that its columns match the
    critic model's expected feature order.

    Args:
        X:
            Incoming feature matrix, shape (N, M_incoming)
        incoming_names:
            Names associated with X's columns
        model_feature_names:
            Names expected by the model
        strict:
            If True -> error on missing model features.
            If False -> missing features are filled with `fill_value`
        fill_value:
            Value used for missing features in non-strict mode

    Returns:
        (X_projected, projected_feature_names)
    """
    # Safety: ensure arrays
    X = np.asarray(X)
    incoming_names = list(incoming_names)
    model_feature_names = list(model_feature_names)

    n_samples = X.shape[0]

    # Map incoming → index
    incoming_index = {name: idx for idx, name in enumerate(incoming_names)}

    # Prepare projected output
    projected = np.zeros((n_samples, len(model_feature_names)), dtype=np.float32)
    projected_names: List[str] = []

    missing: List[str] = []

    for j, feat in enumerate(model_feature_names):
        if feat in incoming_index:
            idx = incoming_index[feat]
            projected[:, j] = X[:, idx]
            projected_names.append(feat)
        else:
            missing.append(feat)
            if strict:
                raise KeyError(
                    f"Feature '{feat}' required by model but missing in incoming names.\n"
                    f"Incoming available: {incoming_names[:10]}..."
                )
            else:
                # non-strict: fill with constant
                projected[:, j] = fill_value
                projected_names.append(feat)

    if missing and not strict:
        log.warning(
            f"[project_features_to_model] Filled {len(missing)} missing features "
            f"with {fill_value}. Missing: {missing[:10]}..."
        )

    return projected, projected_names
