from __future__ import annotations
from typing import Iterable, List
from stephanie.data.prompt_job import PromptJob
from stephanie.services.bus_service import BusService

TOPICS = {
    "prompts_eval": "prompts.eval",
    "results_base": "results.prompt",
    "telemetry_dispatched": "telemetry.prompts.dispatched",
}

class PromptDispatcher:
    def __init__(self):
        self._bus = None

    async def _bus_conn(self):
        if self._bus is None:
            self._bus = await BusService.ensure_connected()
        return self._bus

    async def dispatch_many(self, jobs: Iterable[PromptJob]) -> List[str]:
        bus = await self._bus_conn()
        job_ids: List[str] = []
        for job in jobs:
            job.return_topic = job.return_topic or f"{TOPICS['results_base']}.{job.job_id}"
            await bus.publish_json(TOPICS["prompts_eval"], job.dict())
            await bus.publish_json(TOPICS["telemetry_dispatched"], {
                "job_id": job.job_id,
                "scorable_id": job.scorable_id,
                "model": job.model,
                "created_ts": job.created_ts,
            })
            job_ids.append(job.job_id)
        return job_ids

