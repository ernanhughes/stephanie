# stephanie/tools/vpm_tool.py
from __future__ import annotations

import logging
from typing import Any, Dict, List

from stephanie.tools.base_tool import BaseTool

log = logging.getLogger(__name__)


class VpmTool(BaseTool):
    """
    Tool wrapper for ZeroModel VPM generation.
    Converts metrics → VPM (CHW uint8) + metadata.
    """

    def __init__(self, cfg: Dict[str, Any], memory: Any, container: Any, logger: Any):
        super().__init__(cfg, memory, container, logger)

        # ZeroModel service (must support vpm_from_scorable)
        self.zm = container.get("zeromodel")

        self.require_metrics = bool(
            self.cfg.get("require_metrics_for_vpm", True)
        )

    # ----------------------------------------------------------
    async def apply(
        self,
        scorable,
        metrics_columns: List[str],
        metrics_values: List[float],
    ) -> Dict[str, Any]:
        """
        Generate VPM from metrics. Returns:
            {
              "vision_signals": np.ndarray(C,H,W),
              "vision_signals_meta": dict
            }
        """

        if not self.zm:
            log.debug("[VPMTool] zero model service missing → skip")
            return {}

        if not (metrics_columns and metrics_values):
            msg = "[VPMTool] metrics missing; cannot compute VPM"
            if self.require_metrics:
                log.error(msg)
                raise RuntimeError("VPM requested but metrics missing")
            else:
                log.debug(msg + " (config allow skip)")
                return {}

        log.debug(
            "[VPMTool] rendering VPM id=%s cols=%d",
            scorable.id, len(metrics_columns)
        )

        vpm_u8_chw, meta = await self.zm.vpm_from_scorable(
            scorable,
            metrics_values=metrics_values,
            metrics_columns=metrics_columns,
        )

        return {
            "vision_signals": vpm_u8_chw,
            "vision_signals_meta": meta or {},
        }
