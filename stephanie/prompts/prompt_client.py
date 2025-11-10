# stephanie/prompts/prompt_client.py
from __future__ import annotations
import uuid
from typing import Any, Dict, Iterable, List, Optional, Tuple

from stephanie.services.bus.events.prompt_job import PromptJob, Priority
from stephanie.prompts.registry import TargetRegistry  # your existing router/registry (or stub)


class PromptClient:
    """
    Fire-and-forget publisher for prompts. Never waits for results.
    Returns (job_id, return_topic) tickets you can stash in context.
    """

    def __init__(self, cfg, memory, container, logger):
        self.cfg = cfg
        self.memory = memory
        self.container = container
        self.logger = logger

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
        """
        prompts: list of {prompt_text|messages, system?, params?}
        returns: [(job_id, return_topic), ...]
        """
        bus = await self._bus()
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
            # dispatcher subject (adjust to your topic)
            subject = "prompts.submit"
            # results subject
            if not job.return_topic:
                job.return_topic = f"results.prompts.{job.job_id}"
            await bus.publish_json(subject, job.dict())
            tickets.append((job.job_id, job.return_topic))
        return tickets
