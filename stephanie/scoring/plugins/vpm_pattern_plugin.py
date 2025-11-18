# stephanie/scoring/plugins/vpm_energy_plugin.py
from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np

from .registry import register


def _to_vec(x) -> np.ndarray:
    if x is None:
        return np.zeros((1,), dtype=np.float32)
    if isinstance(x, (list, tuple)):
        x = np.asarray(x, dtype=np.float32)
    elif hasattr(x, "detach"):  # torch.Tensor
        x = x.detach().cpu().float().numpy()
    else:
        x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        x = x.reshape(-1)
    if x.size == 0:
        return np.zeros((1,), dtype=np.float32)
    x = x - x.mean()
    n = float(np.linalg.norm(x)) + 1e-8
    return (x / n).astype(np.float32)

def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))

@register("vpm_pattern")
class VPMPatternPlugin:
    """
    Mammalian-style pattern & valence from VPM embedding.
    - Reads:  attributes["vpm_embedding"]
    - Optional sources:
        * weight,bias (params) OR container.memory.vpms.get_pattern_head()
        * valence_axis (params) OR container.memory.vpms.get_valence_axis()
        * zeromodel-service-v2 (optional): can be used to derive weights/axes centrally
    - Outputs:
        mammalian.pattern_conf01 ∈ [0,1]
        mammalian.valence        ∈ [-1,1]
        mammalian.valence01      ∈ [0,1]
    """
    def __init__(
        self,
        container=None,
        logger=None,
        host=None,
        *,
        weight: Optional[Any] = None,
        bias: float = 0.0,
        valence_axis: Optional[Any] = None,
        use_zero_model: bool = False,
    ):
        self.container = container
        self.logger = logger
        self.host = host

        self.w = _to_vec(weight) if weight is not None else None
        self.b = float(bias)
        self.axis = _to_vec(valence_axis) if valence_axis is not None else None
        self.use_zero_model = bool(use_zero_model)

        # Optional stores/services
        self.mem = getattr(container, "memory", None) if container else None
        self.vpms = getattr(self.mem, "vpms", None) if self.mem else None

        try:
            self.zm = container.get("zeromodel-service-v2") if (container and use_zero_model and hasattr(container, "get")) else None
        except Exception:
            self.zm = None

    def _load_head_lazy(self):
        if self.w is not None:
            return
        try:
            if self.vpms and hasattr(self.vpms, "get_pattern_head"):
                data = self.vpms.get_pattern_head()  # {"weight":[...], "bias":float}
                if data and isinstance(data, dict):
                    if "weight" in data:
                        self.w = _to_vec(data["weight"])
                    if "bias" in data:
                        self.b = float(data["bias"])
        except Exception as e:
            if self.logger:
                self.logger.log("VPMPattern.head.load.error", {"error": str(e)})

    def _load_axis_lazy(self):
        if self.axis is not None:
            return
        try:
            if self.vpms and hasattr(self.vpms, "get_valence_axis"):
                a = self.vpms.get_valence_axis()
                if a is not None:
                    self.axis = _to_vec(a)
        except Exception as e:
            if self.logger:
                self.logger.log("VPMPattern.axis.load.error", {"error": str(e)})

    def _pattern_score(self, x: np.ndarray) -> float:
        self._load_head_lazy()
        if self.w is None:
            # very gentle default head; dimension-agnostic
            rng = max(1, x.size)
            self.w = np.ones((rng,), dtype=np.float32) / float(rng)
            self.b = 0.0
        margin = float(np.dot(x, self.w) + self.b)
        return _sigmoid(margin)

    def _valence(self, x: np.ndarray) -> float:
        self._load_axis_lazy()
        if self.axis is None or self.axis.size != x.size:
            return 0.0
        v = float(np.clip(np.dot(x, self.axis), -1.0, 1.0))
        return v

    def post_process(self, *, tap_output: Dict[str, Any]) -> Dict[str, float]:
        attrs = tap_output.get("attributes") or {}
        emb = attrs.get("vpm_embedding")
        if emb is None:
            return {}

        x = _to_vec(emb)
        conf = self._pattern_score(x)          # [0,1]
        val  = self._valence(x)                # [-1,1]
        val01 = 0.5 * (val + 1.0)              # to [0,1]

        return {
            "mammalian.pattern_conf01": float(conf),
            "mammalian.valence": float(val),
            "mammalian.valence01": float(val01),
        }
