# stephanie/components/nexus/vpm/maps.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from zeromodel.vpm.logic import normalize_vpm

from stephanie.components.nexus.vpm.explain_adapter import \
    occlusion_importance_from_vpm
from stephanie.services.zeromodel_service import ZeroModelService


from typing import Any

def _as01(x: Any) -> np.ndarray:
    a = np.asarray(x, dtype=np.float32)
    if a.size == 0: 
        return np.zeros_like(a, dtype=np.float32)
    if a.max() > 1.0 or a.min() < 0.0:
        a = (a - a.min()) / (a.max() - a.min() + 1e-9)
    return np.clip(a, 0.0, 1.0)

@dataclass
class MapSet:
    # Named “well-known” maps (optional)
    quality: Optional[np.ndarray] = None
    novelty: Optional[np.ndarray] = None
    uncertainty: Optional[np.ndarray] = None
    # Arbitrary extras
    extras: Dict[str, np.ndarray] = field(default_factory=dict)

    # Allow a dict constructor: MapSet(maps=...) or MapSet({...})
    def __init__(self, maps: Optional[Dict[str, Any]] = None, **named: Any):
        object.__setattr__(self, "quality", None)
        object.__setattr__(self, "novelty", None)
        object.__setattr__(self, "uncertainty", None)
        object.__setattr__(self, "extras", {})

        m = dict(maps or {})
        m.update(named or {})
        # Pull out the canonical three if present; keep the rest in extras
        for k in ("quality", "novelty", "uncertainty"):
            if k in m and m[k] is not None:
                object.__setattr__(self, k, _as01(m.pop(k)))
        object.__setattr__(self, "extras", {k: _as01(v) for k, v in m.items()})


    @classmethod
    def from_dict(cls, maps: Dict[str, Any]) -> "MapSet":
        """Back-compat for older provider code that constructs via MapSet.from_dict."""
        return cls(maps=maps)

    def to_dict(self) -> Dict[str, np.ndarray]:
        """Convenience for callers that expect a plain dict."""
        return self.maps

    # Optional: make dict(MapSet) / iteration work if any legacy code expects it
    def __iter__(self):
        yield from self.maps.items()

    @property
    def maps(self) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = dict(self.extras)
        if self.quality is not None:    out["quality"] = self.quality
        if self.novelty is not None:    out["novelty"] = self.novelty
        if self.uncertainty is not None:out["uncertainty"] = self.uncertainty
        return out


class MapProvider:
    """
    Lightweight adapter:
    - If a ZeroModelService exposes a map function, use it.
    - Else return an empty MapSet() (your agent adds safe fallbacks).
    """
    def __init__(self, zeromodel=None):
        self.zm = zeromodel

    def build(self, X: np.ndarray) -> MapSet:
        for attr in ("maps_from_vpm", "build_maps", "get_maps"):
            fn = getattr(self.zm, attr, None)
            if callable(fn):
                try:
                    m = fn(X)
                    if isinstance(m, MapSet):
                        return m
                    if isinstance(m, dict):
                        return MapSet(maps=m)
                    # Any other truthy type → try best-effort dict()
                    if m:
                        return MapSet(maps=dict(m))
                except Exception:
                    pass
        return MapSet()  # empty; agent will add safe fallbacks
    
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
        return MapSet.from_dict(m)

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
