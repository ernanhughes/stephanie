# stephanie/components/nexus/vpm/maps.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from zeromodel.vpm.logic import normalize_vpm

from stephanie.components.nexus.vpm.explain_adapter import \
    occlusion_importance_from_vpm
from stephanie.services.zeromodel_service import ZeroModelService


@dataclass
class MapSet:
    quality: np.ndarray
    novelty: np.ndarray
    uncertainty: np.ndarray
    extras: Dict[str, np.ndarray]  # optional, e.g., risk, coverage
    maps: Dict[str, np.ndarray]  # name -> [H,W] float in [0,1]


class MapProvider:
    """Abstract provider that returns single-channel float32 maps in [0,1]."""
    def __init__(self, zm: ZeroModelService): 
        self.zm = zm
    
    def get_maps(self, *, X: np.ndarray, H: int, W: int) -> MapSet:
        raise NotImplementedError
    
    def _get(self, name: str, H: int, W: int) -> Optional[np.ndarray]:
        try:
            arr = self.zm.vpm_for_dimension(name, H=H, W=W)
            arr = arr[0] if arr.ndim == 3 else arr
            return normalize_vpm(arr).astype(np.float32)
        except Exception:
            return None

    def build(self, X: np.ndarray) -> MapSet:
        H,W = X.shape[-2], X.shape[-1]
        node = normalize_vpm(X[0] if X.ndim==3 else X)
        gx = np.abs(np.diff(node, axis=1, prepend=node[:, :1]))
        gy = np.abs(np.diff(node, axis=0, prepend=node[:1, :]))
        novelty = (gx+gy); M = float(novelty.max() or 1.0); novelty = (novelty/M).astype(np.float32)
        v_rgb = np.transpose(X, (1,2,0)) if X.ndim == 3 else np.repeat(X[:, :, None], 3, axis=2)
        importance, imp_meta = occlusion_importance_from_vpm(v_rgb, stride=8, patch_h=12, patch_w=12, prior="top_left")
        

        m = {
            "quality": self._get("quality", H, W) or node,
            "novelty": self._get("novelty", H, W) or novelty,
            "uncertainty": self._get("uncertainty", H, W) or (1.0 - node),
            "importance": normalize_vpm(importance)
        }
        for extra in ("risk","coverage","bridge"):
            arr = self._get(extra, H, W)
            if arr is not None: m[extra] = arr
        return MapSet(m)

class ZeroModelMapProvider(MapProvider):
    def __init__(self, zm: ZeroModelService):
        self.zm = zm

    def _dim(self, name: str, H: int, W: int) -> Optional[np.ndarray]:
        try:
            vpm = self.zm.vpm_for_dimension(name, H=H, W=W)
        except Exception:
            vpm = None
        if vpm is None:
            return None
        vpm = np.asarray(vpm)
        if vpm.ndim == 3: vpm = vpm[0]          # [C,H,W] -> [H,W]
        return normalize_vpm(vpm).astype(np.float32)

    def get_maps(self, *, X: np.ndarray, H: int, W: int) -> MapSet:
        q = self._dim("quality", H, W)
        n = self._dim("novelty", H, W)
        u = self._dim("uncertainty", H, W)

        if q is None or n is None or u is None:
            # derivatives from X as safe fallback
            node = X[0] if X.ndim == 3 else X
            node = normalize_vpm(node)
            gx = np.abs(np.diff(node, axis=1, prepend=node[:, :1]))
            gy = np.abs(np.diff(node, axis=0, prepend=node[:1, :]))
            novelty = (gx + gy); m = float(novelty.max() or 1.0)
            novelty = (novelty / m).astype(np.float32)
            q = q or node
            n = n or novelty
            u = u or (1.0 - node)

        extras = {}
        # If ZeroModel exposes more dims (risk, bridge, coherence...), register here
        for k in ("risk", "coverage", "bridge"):
            d = self._dim(k, H, W)
            if d is not None: extras[k] = d
        return MapSet(q, n, u, extras)
