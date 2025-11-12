# stephanie/services/bus/prompt_result_listener.py

from __future__ import annotations
import asyncio
from typing import Dict, Callable, List
from stephanie.services.bus_service import BusService

RESULTS_BASE = "results.prompt"

class PromptResultListener:
    def __init__(self):
        self._bus = None

    async def _bus_conn(self):
        if self._bus is None:
            self._bus = await BusService.ensure_connected()
        return self._bus

    async def wait_all(self, job_ids: List[str], timeout_s: float = 60.0) -> Dict[str, Dict]:
        """Collect exactly one message per job_id on results.prompt.<job_id>.
        Returns partial results on timeout.
        """
        bus = await self._bus_conn()
        results: Dict[str, Dict] = {}
        done = asyncio.Event()

        async def _sub(job_id: str):
            subject = f"{RESULTS_BASE}.{job_id}"
            async def _handler(msg):
                results[job_id] = msg.data
                if len(results) == len(job_ids):
                    done.set()
            await bus.subscribe_json(subject, _handler)

        await asyncio.gather(*[_sub(j) for j in job_ids])
        try:
            await asyncio.wait_for(done.wait(), timeout=timeout_s)
        except asyncio.TimeoutError:
            pass
        return results
