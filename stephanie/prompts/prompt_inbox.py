# stephanie/prompts/prompt_inbox.py
from __future__ import annotations

from typing import Dict, Iterable

from stephanie.memory.prompt_job_store import PromptJobStore


class PromptInbox:
    """
    Lightweight fetcher that checks the DB for finished jobs.
    Swap to a bus-listener if you prefer push.
    """
    def __init__(self, store: PromptJobStore):
        self.store = store

    def gather_ready(self, job_ids: Iterable[str]) -> Dict[str, Dict]:
        out: Dict[str, Dict] = {}
        for jid in job_ids:
            row = self.store.get_by_job_id(jid)
            if row and row.status == "succeeded":
                out[jid] = {
                    "job_id": jid,
                    "result_text": row.result_text,
                    "result_json": row.result_json,
                    "latency_ms": row.latency_ms,
                    "provider": row.provider,
                    "cache_hit": row.cache_hit,
                }
        return out
