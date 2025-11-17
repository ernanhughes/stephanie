# stephanie/scoring/plugins/vpm_energy_plugin.py
from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np

from .registry import register
from stephanie.utils.similarity_utils import cosine


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

@register("vpm_energy")
class VPMEnergyPlugin:
    """
    Reptilian-style boundary/threat scoring for VPM embeddings.
    - Reads:  attributes["vpm_embedding"]
    - Optional proto source:
         1) params.core_prototype (list/np)
         2) container.memory.vpms.get_core_prototype()
    - Outputs:
        vpm.cosine
        vpm.energy01           = (1 - cosine)/2
        reptilian.threat01     = σ(gain*energy + bias)
        reptilian.compat01     = 1 - energy
    """
    def __init__(
        self,
        container=None,
        logger=None,
        host=None,
        *,
        gain: float = 3.0,
        bias: float = -1.0,
        ema_update: bool = True,
        ema_decay: float = 0.95,
        core_prototype: Optional[Any] = None,
    ):
        self.container = container
        self.logger = logger
        self.host = host
        self.gain = float(gain)
        self.bias = float(bias)
        self.ema_update = bool(ema_update)
        self.ema_decay = float(ema_decay)
        self._proto = _to_vec(core_prototype) if core_prototype is not None else None

        # Memory/VPM store (optional)
        self.mem = getattr(container, "memory", None) if container else None
        self.vpms = getattr(self.mem, "vpms", None) if self.mem else None

    def _get_proto(self) -> Optional[np.ndarray]:
        if self._proto is not None:
            return self._proto
        try:
            if self.vpms and hasattr(self.vpms, "get_core_prototype"):
                p = self.vpms.get_core_prototype()
                if p is not None:
                    self._proto = _to_vec(p)
                    return self._proto
        except Exception as e:
            if self.logger:
                self.logger.log("VPMEnergy.get_proto.error", {"error": str(e)})
        return None

    def _ema_update_proto(self, x: np.ndarray):
        if not self.ema_update:
            return
        try:
            p = self._get_proto()
            if p is None:
                self._proto = _to_vec(x)
            else:
                self._proto = _to_vec(self.ema_decay * p + (1.0 - self.ema_decay) * _to_vec(x))
            # best-effort persistence
            if self.vpms and hasattr(self.vpms, "set_core_prototype"):
                self.vpms.set_core_prototype(self._proto.tolist())
        except Exception as e:
            if self.logger:
                self.logger.log("VPMEnergy.ema.error", {"error": str(e)})

    def post_process(self, *, tap_output: Dict[str, Any]) -> Dict[str, float]:
        attrs = tap_output.get("attributes") or {}
        emb = attrs.get("vpm_embedding")
        if emb is None:
            return {}

        x = _to_vec(emb)
        proto = self._get_proto() or x  # self-compat if no proto yet
        cos = cosine(x, proto)
        energy = (1.0 - cos) * 0.5               # [0,1]
        compat = 1.0 - energy
        threat = 1.0 / (1.0 + math.exp(-(self.gain * energy + self.bias)))  # σ

        # keep proto fresh
        self._ema_update_proto(x)

        return {
            "vpm.cosine": cos,
            "vpm.energy01": float(energy),
            "reptilian.compat01": float(compat),
            "reptilian.threat01": float(threat),
        }
