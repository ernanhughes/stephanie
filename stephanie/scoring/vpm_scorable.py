# stephanie/scoring/vpm_scorable.py

from __future__ import annotations
from typing import Any, Dict, List, Optional
import time
import uuid
import numpy as np
from stephanie.scoring.scorable import Scorable


class VPMScorable(Scorable):
    """
    Scorable wrapper around a VPM row (metric vector).
    Honors metric order and optional per-metric weights.
    """
    def __init__(
        self,
        id: str,
        run_id: str,
        step: int,
        metric_names: List[str],
        values: List[float],
        *,
        order_weights: Optional[List[float]] = None,        # optional position weights (e.g., order_decay ** idx)
        metric_weights: Optional[Dict[str, float]] = None,  # per-metric overrides by name
        meta: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(id=id, text="VPM row", meta=meta or {})
        self.run_id = run_id
        self.step = step
        self.metric_names = list(metric_names or [])
        self.values = np.asarray(values or [], dtype=np.float32)
        self.order_weights = np.asarray(order_weights or [], dtype=np.float32)
        self.metric_weights = dict(metric_weights or {})

    # ---------- NEW: convenience constructor from (torch) tensor ---------- #
    @classmethod
    def from_tensor(
        cls,
        tensor_like: Any,
        *,
        id: Optional[str] = None,
        run_id: str = "adhoc_vpm_run",
        step: int = 0,
        metric_names: Optional[List[str]] = None,
        order_weights: Optional[List[float]] = None,
        metric_weights: Optional[Dict[str, float]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> "VPMScorable":
        """
        Build a VPMScorable from a 1D embedding/tensor.

        Args:
            tensor_like: torch.Tensor | np.ndarray | list-like (will be flattened)
            id: optional explicit id; default uses time+uuid
            run_id: logical group id (default 'adhoc_vpm_run')
            step: step index within run (default 0)
            metric_names: optional names; default = ["f0", "f1", ...]
            order_weights: optional position-wise weights (len == n_dims)
            metric_weights: optional per-metric name→weight
            metadata: optional extra metadata to carry through scoring

        Returns:
            VPMScorable instance
        """
        # best-effort torch import (keeps this file torch-optional)
        try:
            import torch  # type: ignore
            is_torch = isinstance(tensor_like, torch.Tensor)
        except Exception:  # torch not installed
            torch = None     # type: ignore
            is_torch = False

        # → 1D float32 numpy array
        if is_torch:
            vec = tensor_like.detach().float().cpu().numpy().ravel().astype(np.float32)
        else:
            vec = np.asarray(tensor_like, dtype=np.float32).ravel()

        n = int(vec.size)
        names = list(metric_names) if metric_names else [f"f{i}" for i in range(n)]

        # auto id if not provided
        _id = id or f"vpmrow_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}"

        # align order_weights length if provided
        ow = None
        if order_weights is not None:
            ow = np.asarray(order_weights, dtype=np.float32).ravel()
            if ow.size != n:
                # pad/truncate to match dimensionality
                if ow.size < n:
                    ow = np.pad(ow, (0, n - ow.size), constant_values=1.0)
                else:
                    ow = ow[:n]
            ow = ow.tolist()

        return cls(
            id=_id,
            run_id=run_id,
            step=step,
            metric_names=names,
            values=vec.tolist(),
            order_weights=ow,
            metric_weights=metric_weights,
            meta=meta,
        )

    def get_metric_vector(self) -> np.ndarray:
        v = self.values
        # apply order weights if given (by index)
        if self.order_weights.size and self.order_weights.size == v.size:
            v = v * self.order_weights
        # apply per-metric weights if provided (by name)
        if self.metric_weights:
            w = np.ones_like(v, dtype=np.float32)
            for i, name in enumerate(self.metric_names):
                if name in self.metric_weights:
                    w[i] = float(self.metric_weights[name])
            v = v * w
        # normalize to [0,1] defensively
        vmax = np.max(np.abs(v)) + 1e-8
        return (v / vmax).astype(np.float32)

    def get_names(self) -> List[str]:
        return self.metric_names

    def to_dict(self):
        d = super().to_dict()
        d.update({
            "run_id": self.run_id,
            "step": self.step,
            "metric_names": self.metric_names,
        })
        return d
