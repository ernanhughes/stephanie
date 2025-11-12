from __future__ import annotations
import asyncio
import json
import time
import logging
from typing import Any, Dict, List, Optional, Tuple
from stephanie.services.bus.events.prompt_job import PromptJob, Priority

log = logging.getLogger(__name__)

SUBMIT_SUBJECT = "stephanie.prompts.submit"
RESULT_WC = "stephanie.results.prompts.*"


class PromptClient:
    """
    Offload prompts to the bus, and (optionally) wait for results.
    """

    def __init__(self, cfg, memory, container, logger):
        self.cfg = cfg
        self.memory = memory
        self.container = container
        self.logger = logger
        self._inbox: Dict[str, str] = {}
        self._sub_ready = False

    async def offload_many(
        self,
        *,
        scorable_id: str,
        prompts: List[Dict[str, Any]],
        model: str,
        target_pool: str = "auto",
        priority: Priority = Priority.normal,
        group_key: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        response_format: str = "text",
    ) -> List[Tuple[str, str]]:
        tickets: List[Tuple[str, str]] = []
        for p in prompts:
            job = PromptJob(
                scorable_id=scorable_id,
                prompt_text=p.get("prompt_text"),
                messages=p.get("messages"),
                system=p.get("system"),
                model=model,
                target="auto",
                priority=priority,
                route={"target_pool": target_pool, "group_key": group_key},
                response_format=response_format,
                metadata=meta or {},
            )
            job.finalize_before_publish()
            if not job.return_topic:
                job.return_topic = f"stephanie.results.prompts.{job.job_id}"
            log.info(
                "Offloading prompt",
                extra={"job_id": job.job_id, "subject": SUBMIT_SUBJECT},
            )
            await self.memory.bus.publish(
                SUBMIT_SUBJECT, job.model_dump()
            )  # Pydantic v2
            tickets.append((job.job_id, job.return_topic))
        return tickets

    async def wait_many(
        self,
        tickets: List[Dict[str, str] | Tuple[str, str]],
        *,
        timeout_s: float,
    ) -> List[str]:
        await self._ensure_result_sub()
        want = []
        order = []
        for t in tickets:
            if isinstance(t, dict):
                want.append(t["job_id"])
                order.append((t["job_id"], int(t.get("k_index", 0))))
            else:
                want.append(t[0])
                order.append((t[0], 0))
        want_set = set(want)

        deadline = time.time() + float(timeout_s)
        out: Dict[str, str] = {}

        while want_set and time.time() < deadline:
            for jid in list(want_set):
                if jid in self._inbox:
                    out[jid] = self._inbox.pop(jid)
                    want_set.remove(jid)
            if want_set:
                await asyncio.sleep(0.05)

        # return texts in k_index order where available
        order.sort(key=lambda x: x[1])
        return [out[jid] for (jid, _) in order if jid in out]

    async def try_get(self, job_id: str) -> Optional[str]:
        await self._ensure_result_sub()
        return self._inbox.pop(job_id, None)

    async def _ensure_result_sub(self) -> None:
        if self._sub_ready:
            return

        async def _cb(msg):
            try:
                data = (
                    msg.data
                    if isinstance(msg.data, dict)
                    else json.loads(msg.data.decode("utf-8"))
                )
            except Exception:
                return
            jid = data.get("job_id")
            result = data.get("result", {})
            # Try common shapes
            text = (
                result.get("text")
                or result.get("content")
                or (
                    result.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content")
                    if isinstance(result.get("choices"), list)
                    else None
                )
                or result.get("output")
            )
            if text is None and "error" in data:
                text = f"[error] {data['error']}"
            if jid and isinstance(text, str):
                self._inbox[jid] = text
            if hasattr(msg, "ack"):
                try:
                    await msg.ack()
                except Exception:
                    pass

        await self.memory.bus.subscribe(subject=RESULT_WC, queue=None, handler=_cb)
        self._sub_ready = True
