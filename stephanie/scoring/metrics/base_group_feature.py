# stephanie/scoring/metrics/base_group_feature.py
from __future__ import annotations
from typing import Any, Dict, List

class BaseGroupFeature:
    """
    A group feature wraps a BaseGroupTool and is called once per batch.
    """
    name = "group_feature"
    requires: list[str] = []  # declare upstream dependencies by feature name if you like

    def __init__(self, cfg: Dict, memory, container, logger):
        self.cfg = cfg or {}
        self.memory = memory
        self.container = container
        self.logger = logger
        self.enabled = bool(self.cfg.get("enabled", True))

    async def apply(self, rows: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self.enabled:
            return rows
        return rows
