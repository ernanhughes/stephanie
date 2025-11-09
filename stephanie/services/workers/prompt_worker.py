from __future__ import annotations
import time
from stephanie.data.events import PromptJob
from stephanie.services.llm_service import LLMService  # expects .get(model).generate(text)

PROMPTS_EVAL = "prompts.eval"
RESULTS_BASE = "results.prompt"
TELEMETRY_COMPLETED = "telemetry.prompts.completed"
TELEMETRY_FAILED    = "telemetry.prompts.failed"

class PromptWorkerLocal:

    def __init__(self, memory):
        self._memory = memory
        self._bus = memory.bus

    async def run(self):
        await self._bus.subscribe_json(PROMPTS_EVAL, self._handle)

    async def _handle(self, msg):
        job = PromptJob(**msg.data)
        start = time.time()
        try:
            llm = LLMService.get(job.model)
            text = await llm.generate(job.prompt_text)
            rtt = (time.time() - start) * 1000
            payload = {
                "job_id": job.job_id,
                "scorable_id": job.scorable_id,
                "model": job.model,
                "result": text,
                "latency_ms": rtt,
                "metadata": job.metadata,
            }
            await self._bus.publish_json(job.return_topic or f"{RESULTS_BASE}.{job.job_id}", payload)
            await self._bus.publish_json(TELEMETRY_COMPLETED, payload)
        except Exception as e:
            payload = {
                "job_id": job.job_id,
                "scorable_id": job.scorable_id,
                "model": job.model,
                "error": str(e),
                "metadata": job.metadata,
            }
            await self._bus.publish_json(job.return_topic or f"{RESULTS_BASE}.{job.job_id}", payload)
            await self._bus.publish_json(TELEMETRY_FAILED, payload)

