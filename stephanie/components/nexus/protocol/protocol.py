# stephanie/components/nexus/protocol/protocol.py
from __future__ import annotations

from __future__ import annotations
from typing import Dict, Any

from ..services.graph_layout import NexusService

class NexusProtocol:
    def __init__(self, cfg: Dict, memory, bus=None) -> None:
        self.svc = NexusService(cfg, memory)
        self.bus = bus

    async def on_run(self, run_id: str, start_node_id: str, goal_vec=None, goal_text: str = "") -> Dict[str, Any]:
        out = self.svc.find_path(start_node_id, goal_vec=goal_vec, goal_text=goal_text)
        if self.bus:
            await self.bus.publish("nexus.events.path_found", out)
        return out
