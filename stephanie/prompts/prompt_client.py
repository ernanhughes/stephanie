# stephanie/services/bus/prompt_client.py
from __future__ import annotations

import asyncio
import json
import time
import logging
from typing import Any, Dict, List, Optional, Tuple
import uuid
import contextlib

from stephanie.services.bus.events.prompt_job import PromptJob, Priority
from stephanie.constants import (
    PROMPT_RESULT,
    PROMPT_RESULT_WC,
    PROMPT_SUBMIT,
    PROMPT_DLQ,
)

log = logging.getLogger(__name__)


def _coerce_dict(obj: Any) -> Dict[str, Any]:
    """Accept dict / JSON str / bytes and return a dict."""
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, (bytes, bytearray)):
        return json.loads(obj.decode("utf-8"))
    if isinstance(obj, str):
        # If it's already a JSON string, parse; else wrap it
        try:
            return json.loads(obj)
        except Exception:
            return {"__raw__": obj}
    # last resort
    return {"__raw__": obj}


def _normalize_priority(p: Any) -> Priority:
    if isinstance(p, Priority):
        return p
    if isinstance(p, str):
        s = p.strip().lower()
        if s in ("hi", "high"):
            return Priority.high
        if s in ("lo", "low"):
            return Priority.low
        if s in ("norm", "normal", "med", "medium"):
            return Priority.normal
    return Priority.normal


# -----------------------------------------------------------------------------
def _extract_text(payload: Dict[str, Any]) -> Optional[str]:
    """
    Extract a useful text string from a variety of result shapes.
    Accepted forms (in order):
      {"result":{"text"|"content"|"output"|choices[0].message.content}}
      {"text"|"content"|"output"|"response"}  (flat forms)
      {"error": "..."}                        -> "[error] ..."
    """
    r = payload.get("result")
    if isinstance(r, dict):
        t = (
            r.get("text")
            or r.get("content")
            or (
                r.get("choices", [{}])[0].get("message", {}).get("content")
                if isinstance(r.get("choices"), list)
                else None
            )
            or r.get("output")
        )
        if isinstance(t, str) and t:
            return t

    # Flat fallbacks
    for k in ("text", "content", "output", "response"):
        v = payload.get(k)
        if isinstance(v, str) and v:
            return v

    # Error fallbacks
    if isinstance(r, dict) and "error" in r:
        return f"[error] {r['error']}"
    if "error" in payload:
        return f"[error] {payload['error']}"
    return None


# Fields that strongly indicate we are looking at a *job* (submit echo),
# not a *result* message.
KNOWN_JOB_KEYS = {
    "messages",
    "prompt_text",
    "response_format",
    "return_topic",
    "priority",
    "route",
    "model",
    "tools",
    "attachments",
    "target",
}


def _looks_like_job_echo(d: Dict[str, Any]) -> bool:
    """Job-like payload that has no 'result' → almost certainly a submit echo."""
    return ("result" not in d) and any(k in d for k in KNOWN_JOB_KEYS)


class PromptClient:
    """
    Offload prompts to the bus, and (optionally) wait for results.
    Compatible with InProc, NATS, and ZMQ (topic or RPC).
    """

    def __init__(self, cfg, memory, container, logger):
        self.cfg = cfg
        self.memory = memory
        self.container = container
        self.logger = logger
        self._inbox: Dict[str, str] = {}
        self._sub_ready = False

    # -------------------------------------------------------------------------
    # Publish APIs
    # -------------------------------------------------------------------------

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
        Publish multiple PromptJob messages to the submit subject.
        Returns a list of (job_id, return_topic) tickets.
        """
        await self.memory.bus.wait_ready()
        await self._ensure_result_sub()

        tickets: List[Tuple[str, str]] = []
        pri = _normalize_priority(priority)

        for p in prompts:
            job = PromptJob(
                scorable_id=scorable_id,
                prompt_text=p.get("prompt_text"),
                messages=p.get("messages"),
                system=p.get("system"),
                model=model,
                target="auto",
                priority=pri,
                route={"target_pool": target_pool, "group_key": group_key},
                response_format=response_format,
                metadata=meta or {},
            )
            # --- in offload_many(), just before publish ---
            job.finalize_before_publish()

            # Coerce to dict (to_json may be a string)
            payload = _coerce_dict(job.to_json())

            # ZMQ compatibility: mirror return_topic -> result_subject (fanout topic)
            rt = (
                payload.get("return_topic")
                or payload.get("return")
                or payload.get("ret")
            )
            if rt and "result_subject" not in payload:
                payload["result_subject"] = rt

            log.info(
                "PromptClient -> job=%s subject=%s ret=%s",
                job.job_id,
                PROMPT_SUBMIT,
                payload.get("result_subject") or job.return_topic,
            )

            await self.memory.bus.wait_ready()
            await self.memory.bus.publish(PROMPT_SUBMIT, payload)
            tickets.append((job.job_id, job.return_topic))

        return tickets

    async def offload_one(
        self,
        *,
        scorable_id: str,
        prompt_text: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        model: str = "ollama/qwen:0.5b",
        **kwargs,
    ) -> Tuple[str, str]:
        """
        Convenience: offload a single prompt. Returns (job_id, return_topic).
        """
        tickets = await self.offload_many(
            scorable_id=scorable_id,
            prompts=[{"prompt_text": prompt_text, "messages": messages}],
            model=model,
            **{
                k: v
                for k, v in kwargs.items()
                if k
                in {
                    "target_pool",
                    "priority",
                    "group_key",
                    "meta",
                    "response_format",
                }
            },
        )
        return tickets[0]

    # -------------------------------------------------------------------------
    # Rendezvous APIs
    # -------------------------------------------------------------------------

    async def wait_many(
        self,
        tickets: List[Dict[str, str] | Tuple[str, str]],
        *,
        timeout_s: float,
    ) -> List[str]:
        await self._ensure_result_sub()

        want, order = [], []
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

        order.sort(key=lambda x: x[1])
        return [out[jid] for (jid, _) in order if jid in out]

    async def try_get(self, job_id: str) -> Optional[str]:
        await self._ensure_result_sub()
        return self._inbox.pop(job_id, None)

    # -------------------------------------------------------------------------
    # Subscriptions
    # -------------------------------------------------------------------------

    async def _ensure_result_sub(self) -> None:
        if getattr(self, "_sub_ready", False):
            return
        # Subscribe once to the wildcard
        await self.memory.bus.subscribe(
            subject=PROMPT_RESULT_WC,
            handler=self._on_result_msg,
            queue=None,  # want all results (wildcard fanout)
        )
        self._sub_ready = True
        self._inbox = {}
        self._last_result_at = 0.0
        backend = getattr(self.memory.bus, "get_backend", lambda: "?")()
        log.info(
            "PromptClient.Subscribed results_wildcard=%s (backend=%s)",
            PROMPT_RESULT_WC,
            backend,
        )

    async def _on_result_msg(self, msg):
        """Accepts dict / bytes / bus message; stores under job_id."""
        try:
            subject = getattr(msg, "subject", "") or ""
            # decode to dict
            if isinstance(msg, dict):
                data = msg
            elif hasattr(msg, "data"):
                raw = msg.data
                data = (
                    raw
                    if isinstance(raw, dict)
                    else json.loads(raw.decode("utf-8"))
                )
            elif isinstance(msg, (bytes, bytearray)):
                data = json.loads(msg.decode("utf-8"))
            else:
                log.warning("PromptClient.ResultUnparsable type=%r", type(msg))
                if hasattr(msg, "ack"):
                    with contextlib.suppress(Exception):
                        await msg.ack()
                return

            # job id
            jid = data.get("job_id") or data.get("id") or data.get("job")
            if not jid and subject.startswith("results.prompts."):
                with contextlib.suppress(Exception):
                    jid = subject.split("results.prompts.", 1)[1]

            if not jid:
                log.warning(
                    "PromptClient.ResultNoJobId subj=%s keys=%s data_preview=%s",
                    subject,
                    list(data.keys()),
                    str(data)[:160],
                )
                if hasattr(msg, "ack"):
                    with contextlib.suppress(Exception):
                        await msg.ack()
                return

            # text extraction
            text = _extract_text(data)
            if text is None:
                text = json.dumps(data, ensure_ascii=False)

            self._inbox[jid] = text
            self._last_result_at = time.time()
            log.info(
                "PromptClient.ResultStored job=%s len=%d subj=%s",
                jid,
                len(text),
                subject,
            )

            if hasattr(msg, "ack"):
                with contextlib.suppress(Exception):
                    await msg.ack()

        except Exception as e:
            log.exception("PromptClient.ResultHandlerError: %s", e)
            if hasattr(msg, "ack"):
                with contextlib.suppress(Exception):
                    await msg.ack()
