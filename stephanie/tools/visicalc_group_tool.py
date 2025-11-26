# stephanie/tools/visicalc_group_tool.py
from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

from stephanie.tools.visicalc_tool import VisiCalcTool

log = logging.getLogger(__name__)

class VisiCalcGroupTool:
    name = "visicalc_group"

    def __init__(self, cfg: Dict, memory, container, logger):
        self.cfg = cfg or {}
        self.memory = memory
        self.container = container
        self.logger = logger
        self.vc = VisiCalcTool(cfg, memory, container, logger)

    async def apply(self, rows: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not rows:
            return rows

        out = self.vc.apply_rows(
            episode_id=context.get("pipeline_run_id", "visicalc:default"),
            rows=rows,
            meta={"n_rows": len(rows)},
        )

        feat_names = out["feature_names"]
        feats = out["features"]
        rep = out["report"].to_dict()

        # Attach per-row features; keep big arrays on disk if needed
        for i, r in enumerate(rows):
            r.setdefault("visicalc_feature_names", feat_names)
            if isinstance(feats, np.ndarray):
                r["visicalc_features"] = feats[i].tolist()
            r.setdefault("visicalc_report", rep)

        # out["vpm"] is cohort-level; consider saving and storing a path
        return rows
