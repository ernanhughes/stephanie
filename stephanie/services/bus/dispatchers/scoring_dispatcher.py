from __future__ import annotations

from typing import Iterable, List

from stephanie.events.scoring_job import ScoringJob

PROMPT = "scoring.eval"
RESULT_BASE = "results.scoring"
TEL = "telemetry.scoring.dispatched"

class ScoringDispatcher:
    def __init__(self, memory):
        self.memory = memory


    async def dispatch_many(self, jobs: Iterable[ScoringJob]) -> List[str]:
        ids: List[str] = []
        for j in jobs:
            j.return_topic = j.return_topic or f"{RESULT_BASE}.{j.job_id}"
            await self.memory.bus.publish_json(PROMPT, j.dict())
            await self.memory.bus.publish_json(TEL, {"job_id": j.job_id, "scorable_id": j.scorable.get("id")})
            ids.append(j.job_id)
        return ids
