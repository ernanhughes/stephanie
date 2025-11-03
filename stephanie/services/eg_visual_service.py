# stephanie/services/eg_visual_service.py
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from stephanie.components.risk.epi.epistemic_guard import EGVisual
from stephanie.services.service_protocol import Service


class EGVisualService(Service):
    """
    DEPRECATED SERVICE.

    This service is provided for backward compatibility only.
    In production, VPM channels are now streamed directly to ZeroModelService
    via VPMWorkerInline.add_channels() using real hallucination signals.

    Do not use this service for new integrations. It renders static PNGs from
    synthetic or legacy VPM tensors. The real visualization pipeline is now:
        HallucinationSignals → VPMWorkerInline.add_channels() → Jitter VPM Tiles.

    If you must use it (e.g., for legacy dashboards), ensure your VPM tensor
    is a numpy array of shape (n_tokens, 4) with channels: R, G, B, A.
    """

    def __init__(self):
        self._logger = logging.getLogger(self.name)
        self._visual: Optional[EGVisual] = None
        self._up = False

    @property
    def name(self) -> str:
        return "eg-visual-v2"

    def initialize(self, **kwargs) -> None:
        cfg = (kwargs.get("config") or {}) if kwargs else {}
        logger = kwargs.get("logger")
        if logger is not None:
            self._logger = logger

        out_dir = cfg.get("out_dir", "./runs/eg/img")
        seed = int(cfg.get("seed", 42))
        self._visual = EGVisual(out_dir=out_dir, seed=seed)
        self._up = True
        self._logger.info(
            "EGVisualService initialized (DEPRECATED)",
            extra={
                "out_dir": out_dir,
                "seed": seed,
                "warning": "Use VPMWorkerInline.add_channels() for real-time VPM in Jitter."
            },
        )

    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy" if self._up else "unhealthy",
            "metrics": {},
            "dependencies": {
                "visual": "ready" if self._visual else "missing",
                "deprecated": True
            },
        }

    def shutdown(self) -> None:
        self._up = False
        self._visual = None
        self._logger.info("EGVisualService shutdown (DEPRECATED)")

    # --- Pass-through (for legacy use only) ---
    def render(self, trace_id: str, vpm: Any) -> tuple[str, str, str]:
        """
        Renders field, strip, and legend PNGs from a VPM tensor.
        This is a legacy method. Do not use in new code.

        Args:
            trace_id: Run identifier.
            vpm: Must be a numpy.ndarray of shape (n, 4) with channels R,G,B,A.

        Returns:
            Tuple of (field_path, strip_path, legend_path)
        """
        if not self._visual:
            raise RuntimeError("EGVisualService not initialized")

        # Validate input
        if not isinstance(vpm, (list, tuple)) and hasattr(vpm, 'shape') and len(vpm.shape) == 2 and vpm.shape[1] == 4:
            pass
        else:
            # Attempt to convert from dict or list if needed
            if isinstance(vpm, dict) and all(k in vpm for k in ['R', 'G', 'B', 'A']):
                import numpy as np
                R = np.array(vpm['R'])
                G = np.array(vpm['G'])
                B = np.array(vpm['B'])
                A = np.array(vpm['A'])
                vpm = np.stack([R, G, B, A], axis=1)
            else:
                raise ValueError("vpm must be numpy.ndarray of shape (n, 4) or dict with keys R,G,B,A")

        field_path, strip_path, legend_path = self._visual.render(trace_id, vpm)
        self._logger.debug(
            f"EGVisualService rendered legacy artifacts for {trace_id}",
            extra={
                "field": field_path,
                "strip": strip_path,
                "legend": legend_path
            }
        )
        return field_path, strip_path, legend_path