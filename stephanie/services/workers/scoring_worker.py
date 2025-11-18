from __future__ import annotations

import time

from stephanie.services.bus.events.scoring_job import ScoringJob

PROMPT = "scoring.eval"
RESULT_BASE = "results.scoring"
TEL_OK = "telemetry.scoring.completed"
TEL_FAIL = "telemetry.scoring.failed"

class ScoringWorker:
    def __init__(self, memory, container):
        self.memory = memory
        self.bus = memory.bus
        self.container = container

    async def run(self):
        await self.bus.subscribe_json(PROMPT, self._handle)

    async def _handle(self, msg):
        job = ScoringJob(**msg.data)
        t0 = time.time()
        try:
            scoring = self.container.get("scoring")
            from stephanie.scoring.scorable import Scorable
            scorable = Scorable.from_dict(job.scorable)
            res = scoring.evaluate_state(
                scorable=scorable,
                context=job.context,
                scorers=job.scorers,
                dimensions=job.dimensions,
                scorer_weights=job.scorer_weights,
                dimension_weights=job.dimension_weights,
                include_llm_heuristic=job.include_llm_heuristic,
                include_vpm_phi=job.include_vpm_phi,
                fuse_mode=job.fuse_mode,
                clamp_01=job.clamp_01,
            )
            payload = {"job_id": job.job_id, "scorable_id": job.scorable.get("id"), "result": res, "latency_ms": (time.time()-t0)*1000}
            await self.bus.publish_json(job.return_topic or f"{RESULT_BASE}.{job.job_id}", payload)
            await self.bus.publish_json(TEL_OK, payload)
        except Exception as e:
            payload = {"job_id": job.job_id, "scorable_id": job.scorable.get("id"), "error": str(e)}
            await self.bus.publish_json(job.return_topic or f"{RESULT_BASE}.{job.job_id}", payload)
            await self.bus.publish_json(TEL_FAIL, payload)
