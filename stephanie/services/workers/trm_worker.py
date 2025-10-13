# stephanie/services/workers/trm_worker.py
from __future__ import annotations
import asyncio
import json
import torch 

class TRMWorker:
    """
    Consumes 'arena.configsolve.request', performs recursive reasoning
    using TinyRecursionModel, and publishes 'arena.configsolve.ready'.
    """
    def __init__(self, cfg, memory, container, logger):
        self.cfg = cfg
        self.logger = logger
        self.trm = container.get("tiny_recursion_model")

    async def handle(self, msg):
        data = json.loads(msg.data)
        x, y0 = data["x"], data.get("y0")
        logits, halt_p, z = self.trm(x, y0, torch.zeros_like(x))
        result = dict(solution=torch.argmax(logits, -1).tolist(), confidence=float(halt_p))
        await self.memory.bus.publish("arena.configsolve.ready", json.dumps(result))
