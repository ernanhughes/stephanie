# stephanie/services/bus/bus_llm.py
from __future__ import annotations

import uuid

from stephanie.prompts.dispatcher import PromptDispatcher
from stephanie.prompts.result_listener import ResultListener
from stephanie.prompts.targets_bootstrap import build_default_registry
from stephanie.services.bus.events.prompt_job import PromptJob


class BusLLM:
    def __init__(self, model="gpt-4o-mini", target="auto", priority="normal"):
        self.registry = build_default_registry()
        self.dispatcher = PromptDispatcher(self.registry)
        self.model, self.target, self.priority = model, target, priority

    async def generate_many(self, scorable_id: str, prompts: list[str], meta=None, timeout_s=90):
        jobs = []
        job_ids = []
        for p in prompts:
            jid = str(uuid.uuid4())
            job_ids.append(jid)
            jobs.append(PromptJob(
                job_id=jid, scorable_id=scorable_id, prompt_text=p,
                model=self.model, target=self.target, priority=self.priority,
                metadata=meta or {}
            ))
        await self.dispatcher.dispatch_many(jobs)
        listener = ResultListener()
        results = await listener.wait_all(job_ids, timeout_s=timeout_s)
        # return in original order, fallback to None on misses
        out = []
        for jid in job_ids:
            payload = results.get(jid)
            out.append((jid, (payload or {}).get("result")))
        return out
