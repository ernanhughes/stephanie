# stephanie/components/critic/model/shadow.py
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger(__name__)    

def save_shadow_pack(path: str | Path,
                     X: np.ndarray,
                     y: Optional[np.ndarray] = None,
                     feature_names: Optional[Sequence[str]] = None,
                     groups: Optional[Sequence[str]] = None,
                     meta: Optional[Mapping[str, Any]] = None,
                     *,
                     kept: Optional[Sequence[str]] = None):
    """
    Canonical format for critic shadow packs.

    Accepts both `feature_names` and alias `kept`. If both are provided,
    `feature_names` wins; we still save BOTH 'feature_names' and 'kept'
    arrays into the archive for maximum compatibility.

    Keys saved:
      - X: float32 [n, d]
      - y: int64 [n]
      - feature_names: object[str] [d]
      - kept: object[str] [d]        # alias for legacy/new code
      - groups: object[str] [n]
      - meta: json string
    """
    X = np.asarray(X, dtype=np.float32)
    if y is None:
        y = np.zeros((X.shape[0],), dtype=np.int64)
    else:
        y = np.asarray(y, dtype=np.int64)

    # Resolve names from feature_names or kept
    names_seq = feature_names if feature_names is not None else kept
    if names_seq is None:
        raise ValueError("save_shadow_pack: must provide feature_names or kept")
    names_list = list(map(str, names_seq))
    names_arr = np.asarray(names_list, dtype=object)
    kept_arr  = names_arr  # write both keys

    if groups is None:
        groups = np.array(["unknown"] * X.shape[0], dtype=object)
    else:
        groups = np.asarray(list(groups), dtype=object)

    meta_str = json.dumps(dict(meta or {}))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path,
             X=X.astype(np.float32),
             y=y.astype(np.int8),
             feature_names=np.array(names_arr, dtype=object),
             kept=kept_arr,
             groups=np.array(groups, dtype=object) if groups is not None else np.array([], dtype=object),
             meta=meta_str)
    log.info("CriticTrainer: saved shadow pack → %s", path)



def load_shadow_pack(path: str | Path) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, Dict]:
    """
    Load the pack: returns (X, y, kept, groups, meta_dict)
    """
    path = Path(path)
    with np.load(path, allow_pickle=True) as z:
        X = z["X"]
        y = z["y"]
        groups = z["groups"]
        kept = [str(s) for s in z["kept"].tolist()]
        meta_json = z["meta"].tolist()[0]
        meta = json.loads(meta_json) if isinstance(meta_json, (str, bytes)) else {}
    return X, y, kept, groups, meta
