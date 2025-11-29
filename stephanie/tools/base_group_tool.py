# stephanie/tools/base_group_tool.py
from __future__ import annotations

from typing import Any, Dict, List


class BaseGroupTool:
    """
    Stateless, idempotent tool applied to a cohort (list of rows).
    Contract: rows in → rows out (may mutate in place, but must return rows).
    """
    name = "group_tool"

    def __init__(self, cfg: Dict, memory, container, logger):
        self.cfg = cfg or {}
        self.memory = memory
        self.container = container
        self.logger = logger

    async def apply(self, rows: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        raise NotImplementedError
