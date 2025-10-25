# stephanie/scoring/vpm_scorable.py

from __future__ import annotations
from typing import Any, Dict, List, Optional
import time, uuid
import numpy as np
from stephanie.scoring.scorable import Scorable


class VPMScorable(Scorable):
    """
    Scorable wrapper around a VPM row (metric vector).
    Honors metric order and optional per-metric weights, and can render
    a VPM "image" (square-packed) for vision-based scorers.
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
        image_array: Optional[np.ndarray] = None,           # optional pre-rendered HxW or HxWxC
    ):
        super().__init__(id=id, text="VPM row", meta=meta or {})
        self.run_id = run_id
        self.step = step
        self.metric_names = list(metric_names or [])
        self.values = np.asarray(values or [], dtype=np.float32)
        self.order_weights = np.asarray(order_weights or [], dtype=np.float32)
        self.metric_weights = dict(metric_weights or {})
        self._img_cache: Optional[np.ndarray] = None

        # allow caller to inject a pre-built image (will be returned as-is)
        if image_array is not None:
            arr = np.asarray(image_array)
            # normalize dtype
            if arr.dtype != np.float32:
                arr = arr.astype(np.float32)
            # clamp to [0,1] if it's obviously outside
            if arr.max() > 1.0:
                arr = arr / 255.0
            self._img_cache = arr

    # ---------- convenience constructor from (torch/np) tensor ---------- #
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
        image_array: Optional[np.ndarray] = None,
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
        except Exception:
            torch = None  # type: ignore
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
            image_array=image_array,
        )

    # ---------- vector → normalized (weights + order) ---------- #
    def get_metric_vector(self) -> np.ndarray:
        v = self.values
        # apply order weights if given (by index)
        if self.order_weights.size and self.order_weights.size == v.size:
            v = v * self.order_weights
        # per-metric name weights
        if self.metric_weights:
            w = np.ones_like(v, dtype=np.float32)
            for i, name in enumerate(self.metric_names):
                if name in self.metric_weights:
                    w[i] = float(self.metric_weights[name])
            v = v * w
        # normalize to [-1,1] by max-abs, then robust 0..1 in get_image_array()
        vmax = np.max(np.abs(v)) + 1e-8
        return (v / vmax).astype(np.float32)

    def get_names(self) -> List[str]:
        return self.metric_names

    # ---------- NEW: image for vision-based scorers ---------- #
    def get_image_array(self) -> np.ndarray:
        """
        Returns an HxW or HxWx1 float32 array in [0,1].
        Default renderer: robust 0–1 + optional PHOS-style descending pack into a square.
        Caches the result for reuse.
        """
        if self._img_cache is not None:
            return self._img_cache

        v = self.get_metric_vector().ravel()
        if v.size == 0:
            self._img_cache = np.zeros((8, 8), dtype=np.float32)
            return self._img_cache

        # robust 0..1 (percentile-based)
        v01 = self._robust01(v)

        # optional ordering (PHOS-like): sort desc to concentrate mass in TL
        phos_sort = bool((self.meta or {}).get("phos_sort", True))
        if phos_sort:
            v01 = np.sort(v01)[::-1]

        # square-pack with top-left padding
        img = self._phos_pack(v01)

        # cache & return
        self._img_cache = img.astype(np.float32)
        return self._img_cache

    # ---------- helpers ---------- #
    @staticmethod
    def _robust01(x: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.size == 0:
            return x
        lo_v, hi_v = np.percentile(x, [lo, hi])
        if not np.isfinite(lo_v) or not np.isfinite(hi_v) or hi_v <= lo_v:
            # degenerate; center to zeros
            return np.zeros_like(x, dtype=np.float32)
        y = (x - lo_v) / (hi_v - lo_v)
        return np.clip(y, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _phos_pack(v: np.ndarray) -> np.ndarray:
        """
        Square-pack a 1D vector (already scaled 0..1) into an s×s image.
        Pads on the bottom-right; keeps top-left concentration if sorted desc.
        """
        v = np.asarray(v, dtype=np.float32).ravel()
        s = int(np.ceil(np.sqrt(v.size)))
        pad = s * s - v.size
        if pad > 0:
            v = np.pad(v, (0, pad), constant_values=0.0)
        return v.reshape(s, s)

    def to_dict(self):
        d = super().to_dict()
        d.update({
            "run_id": self.run_id,
            "step": self.step,
            "metric_names": self.metric_names,
        })
        return d
