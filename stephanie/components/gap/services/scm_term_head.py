# stephanie/components/gap/services/scm_term_head.py
from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional
from stephanie.services.service_protocol import Service

import torch
import torch.nn as nn
import numpy as np
import time

_logger = logging.getLogger(__name__)

# Fixed order of Shared Core Metrics we want every model to emit
SCM_FEATURE_KEYS: List[str] = [
    "scm.reasoning.score01",
    "scm.knowledge.score01",
    "scm.clarity.score01",
    "scm.faithfulness.score01",
    "scm.coverage.score01",
    "scm.aggregate01",
    "scm.uncertainty01",
    "scm.ood_hat01",
    "scm.consistency01",
    "scm.length_norm01",
    "scm.temp01",
    "scm.agree_hat01",
]

def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        if isinstance(x, torch.Tensor):
            return float(x.detach().float().mean().item())
        return float(x)
    except Exception:
        return default

def _extract_vector(payload: Mapping[str, Any]) -> Dict[str, float]:
    """
    Get a flat dict of metrics from a scorer payload. Supports either:
      - payload["vector"] (preferred), or
      - payload["columns"] + payload["values"].
    """
    vec = payload.get("vector")
    if isinstance(vec, dict):
        return {str(k): _to_float(v) for k, v in vec.items()}
    cols = payload.get("columns")
    vals = payload.get("values")
    if isinstance(cols, list) and isinstance(vals, list) and len(cols) == len(vals):
        return {str(c): _to_float(v) for c, v in zip(cols, vals)}
    return {}

def _scm_feature_row(metrics_payload: Mapping[str, Any]) -> List[float]:
    """
    Build the SCM feature row in the fixed order.
    If a key is missing, default to 0.0 (or derive later if desired).
    """
    vec = _extract_vector(metrics_payload)
    return [ _to_float(vec.get(k, 0.0)) for k in SCM_FEATURE_KEYS ]


class _SimpleAdapter(nn.Module):
    """
    Small per-model adapter that projects the SCM feature vector (len K)
    into a compact latent (latent_dim). The first Linear layer's IN dim is
    set to len(SCM_FEATURE_KEYS), so we never get shape errors.
    """
    def __init__(self, in_dim: int, latent_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_dim]
        return self.mlp(x)  # [B, latent_dim]


class SCMTermHeadService(Service):
    """
    Universal SCM term head:
      - Extracts a fixed SCM feature vector from any scorer payload
      - Passes it through a small per-model adapter (optional latent)
      - Returns the SCM metrics dict (and can be extended to learned heads)

    Usage:
        scm = SCMTermHeadService(device="cpu")
        out = scm.infer("hrm", hrm_metrics_payload)
        # out is { "scm.reasoning.score01": ..., ... }
    """
    def __init__(self, device: Optional[str] = None, latent_dim: int = 32):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        in_dim = len(SCM_FEATURE_KEYS)
        # Build per-model adapters with the correct input size
        self.adapters: Dict[str, _SimpleAdapter] = {
            "hrm": _SimpleAdapter(in_dim=in_dim, latent_dim=latent_dim).to(self.device),
            "tiny": _SimpleAdapter(in_dim=in_dim, latent_dim=latent_dim).to(self.device),
        }

        # (Optional) learned output heads could go here if you want z -> refined SCM
        # For now, we emit the input SCM values so downstream alignment is guaranteed.


    def initialize(self, **kwargs) -> None:
        """
        Expected kwargs:
          - base_dir: str | Path (required)
          - logger: optional custom logger
        """
        self._initialized = True
        _logger.info("SCMTermHeadService initialized")

    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy" if self._initialized else "uninitialized",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "in_dim": str(self.in_dim) if self._base else None,
            "adapters": str(list(self.adapters.keys())),
            "device": str(self.device),
        }

    def shutdown(self) -> None:
        # Nothing persistent to close; keep for symmetry/logging
        _logger.debug("[SCMTermHeadService] Shutdown complete")

    @property
    def name(self) -> str:
        return "scm-term-head-v1"

    def infer(self, model: str, metrics_payload: Mapping[str, Any]) -> Dict[str, float]:
        """
        Produce SCM metrics from a scorer payload. If you later want to make these
        learned (z -> SCM), you already have z computed below.
        """
        if model not in self.adapters:
            raise KeyError(f"SCMTermHeadService: unknown model '{model}' (expected one of {list(self.adapters)})")

        # 1) Build the feature row in fixed order
        feats = _scm_feature_row(metrics_payload)  # length K (12)
        x = torch.tensor(feats, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, K]

        # 2) Project to a small latent (not strictly needed for pass-through)
        with torch.no_grad():
            z = self.adapters[model](x)  # [1, latent_dim]

        # 3) For now, return the SCM metrics as-is (pass-through).
        #    If you want to refine/normalize via z, you can add small heads here.
        scm_dict = {k: float(v) for k, v in zip(SCM_FEATURE_KEYS, feats)}

        # Optionally expose latent for debugging
        # scm_dict["_latent_norm"] = float(z.norm(p=2).item())

        return scm_dict
