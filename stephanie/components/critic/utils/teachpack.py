# stephanie/components/critic/critic_teachpack.py
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def export_teachpack(
    out_path: Path,
    X_proj: np.ndarray,
    feature_names: List[str],
    y: np.ndarray,
    teacher_probs: np.ndarray,
    meta: Dict[str, Any],
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=X_proj.astype(np.float32),
        y=y.astype(np.int64),
        p_teacher=teacher_probs.astype(np.float32),
        feature_names=np.array(feature_names, dtype=object),
        meta=json.dumps(meta),
    )
    return out_path

def teachpack_meta(model_fp: str, feature_names: List[str], calib: Dict[str, float] | None):
    return {
        "model_fingerprint": model_fp,
        "feature_fingerprint": hashlib.sha256("|".join(feature_names).encode()).hexdigest()[:12],
        "calibration": calib or {},
        "schema": {
            "X": "float32 (n, d) aligned to feature_names",
            "y": "int64 {0,1}",
            "p_teacher": "float32 (n,) calibrated probs",
            "feature_names": "object (d,)"
        }
    }
