from __future__ import annotations
import asyncio
import json
import logging
from typing import Any, Dict

from stephanie.services.prompt_service import PromptService
from stephanie.constants import PROMPT_SUBMIT, BUS_STREAM
from stephanie.services.bus.events.prompt_job import PromptJob, ResponseFormat
from stephanie.utils.json_sanitize import dumps_safe

log = logging.getLogger(__name__)


class PromptDispatcherWorker:
    """
    Bus worker that consumes PromptJob messages and returns results to job.return_topic.
    Works with HybridBus (NATS or ZMQ backends). No acks are required for ZMQ; HybridBus handles differences.
    """

    def __init__(self, cfg, memory, container, logger):
        self.cfg = cfg
        self.memory = memory
        self.container = container
        self.logger = logger
        self._subscribed = False

        # Optional: provider registry; for now, pick simple adapter
        self._provider = self._detect_provider()
        self.prompt: PromptService = self.container.get("prompt")

    # ---- public --------------------------------------------------------------

    async def start(self) -> None:
        bus = self.memory.bus
        self.logger.info("[PromptWorker] ensure_stream → %s", BUS_STREAM)
        # Also register the results wildcard so publish() never hits a missing subject on NATS
        await bus.ensure_stream(BUS_STREAM, subjects=[PROMPT_SUBMIT, "results.prompts.>"])

        self.logger.info("[PromptWorker] subscribing (before) → %s", PROMPT_SUBMIT)
        await bus.subscribe(subject=PROMPT_SUBMIT, handler=self._on_submit, queue="q_prompt_workers")
        self._subscribed = True
        self.logger.info("[PromptWorker] subscribed (after)  → %s", PROMPT_SUBMIT)    
        
    async def stop(self) -> None:
        # If your bus exposes unsubscribe/close, call it; ZMQ path may be no-op.
        log.info("[PromptWorker] stop() called")

    # ---- handlers ------------------------------------------------------------

    async def _on_submit(self, msg: Any) -> None:
        try:
            self.logger.info("[PromptWorker] _on_submit received msg=%r", getattr(msg, "data", msg))
            data = await self._decode(msg)
            self.logger.info("[PromptWorker] decoded keys=%s", list(data.keys()))
        except Exception as e:
            self.logger.exception("[PromptWorker] DecodeError: %s", e)
            await self._safe_ack(msg)
            return

        try:
            # IMPORTANT: sanitize unknown keys (e.g., result_subject) before model_validate()
            job_dict = self._sanitize_prompt_job_dict(data)
            job = PromptJob.model_validate(job_dict)
        except Exception as e:
            self.logger.exception("[PromptWorker] InvalidPromptJob: %s", e)
            await self._publish_error(data, error=str(e))
            await self._safe_ack(msg)
            return

        try:
            result = await self._execute_job(job)
        except Exception as e:
            self.logger.exception("[PromptWorker] ExecuteError job=%s: %s", job.job_id, e)
            await self._publish_error(job, error=str(e))
            await self._safe_ack(msg)
            return

        try:
            await self._publish_result(job, result)
        except Exception as e:
            self.logger.exception("[PromptWorker] PublishResultError job=%s: %s", job.job_id, e)

        await self._safe_ack(msg)

    # ---- execution -----------------------------------------------------------

    async def _execute_job(self, job: PromptJob) -> Dict[str, Any]:
        """
        Run the LLM call and normalize to a dict with keys we know how to extract:
          {
            "job_id": ..., "result": {"text": "...", ...},
            "meta": {"worker": "prompt_dispatcher"}
          }
        """
          # Simple policy: prefer messages path
        payload = job.to_openai_payload()

           # Route by model name; this can be extended into a registry
        prompt_text = payload.get("messages")[1].get("content") if payload.get("messages") and len(payload.get("messages")) > 1 else payload.get("prompt")
        text = await self.prompt.run_prompt(
                prompt_text=prompt_text or "",
                context={},
                model=payload.get("model"),
            )
        # Wrap to normalized shape
        out: Dict[str, Any] = {
            "job_id": job.job_id,
            "status": "ok",
            "result": {"text": text},
            "meta": {"worker": "prompt_dispatcher"},
        }

        # Optional: validate JSON/object formats
        if job.response_format == ResponseFormat.json_object:
            try:
                json.loads(text)  # must be valid JSON object
            except Exception:
                out["status"] = "format_error"
                out["result"] = {"text": text}
                out["error"] = "invalid_json_object"

        return out


    def _sanitize_prompt_job_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Drop unknown fields so PromptJob(extra='forbid') passes cleanly."""
        allowed = set(PromptJob.model_fields.keys())
        out = {k: v for k, v in d.items() if k in allowed}
        # Pydantic will coerce enum strings; keep them as-is.
        return out

    async def _publish_result(self, job: PromptJob, result: Dict[str, Any]) -> None:
        bus = self.memory.bus
        subject = job.return_topic or f"results.prompts.{job.job_id}"
        await bus.ensure_stream(BUS_STREAM, subjects=[subject])  # NATS-safe; ZMQ no-op

        payload = dumps_safe(result, ensure_ascii=False)
        # Use the same calling convention your HybridBus uses elsewhere:
        await bus.publish(subject, payload)

        self.logger.info("[PromptWorker] result published job=%s → %s (len=%d)", job.job_id, subject, len(payload))


    async def _run_ollama(self, job: PromptJob, payload: Dict[str, Any]) -> str:
        """
        Minimal local Ollama call. If you already have an Ollama client in container, use that.
        We keep it resilient (no internet).
        """
        import aiohttp
        base = (job.model or {}).get("api_base") or "http://localhost:11434"
        url = f"{base.rstrip('/')}/v1/chat/completions"

        body = {
            "model": (job.model or {}).get("name", "qwen2.5:3b"),
            "messages": payload["messages"],
            "temperature": payload.get("temperature", 0.2),
            "max_tokens": payload.get("max_tokens"),
            "top_p": payload.get("top_p"),
            "stop": payload.get("stop"),
            "stream": False,
        }
        # remove None
        body = {k: v for k, v in body.items() if v is not None}

        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.post(url, json=body, timeout=120) as resp:
                    j = await resp.json()
            # OpenAI-compatible shape
            text = (
                j.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            return text or ""
        except Exception as e:
            log.warning("[PromptWorker] Ollama call failed: %s", e)
            return ""

    # ---- publish helpers -----------------------------------------------------

    async def _publish_result(self, job: PromptJob, result: Dict[str, Any]) -> None:
        bus = self.memory.bus
        subject = job.return_topic or f"results.prompts.{job.job_id}"
        # Ensure stream exists for the dynamic subject on NATS; ZMQ path is fine
        await bus.ensure_stream(BUS_STREAM, subjects=[subject])

        payload = dumps_safe(result, ensure_ascii=False)
        await bus.publish(subject, payload)

        log.info(
            "[PromptWorker] result published job=%s → %s (len=%d)",
            job.job_id, subject, len(payload),
        )

    async def _publish_error(self, job_or_data: Any, *, error: str) -> None:
        try:
            job_id = getattr(job_or_data, "job_id", None) or job_or_data.get("job_id") or "unknown"
            subject = getattr(job_or_data, "return_topic", None) or f"results.prompts.{job_id}"
        except Exception:
            subject, job_id = "results.prompts.unknown", "unknown"

        payload = {
            "job_id": job_id,
            "status": "error",
            "error": str(error),
            "meta": {"worker": "prompt_dispatcher"},
        }
        await self.memory.bus.publish(subject, dumps_safe(payload, ensure_ascii=False))
        log.info("[PromptWorker] error published job=%s → %s : %s", job_id, subject, error)

    # ---- utils ---------------------------------------------------------------

    async def _decode(self, msg: Any) -> Dict[str, Any]:
        if isinstance(msg, dict):
            return msg
        if hasattr(msg, "data"):
            raw = msg.data
            if isinstance(raw, (bytes, bytearray)):
                return json.loads(raw.decode("utf-8"))
            if isinstance(raw, dict):
                return raw
        if isinstance(msg, (bytes, bytearray)):
            return json.loads(msg.decode("utf-8"))
        # Some buses may pass already-serialized JSON strings
        if isinstance(msg, str):
            return json.loads(msg)
        raise ValueError(f"Unsupported message type: {type(msg)}")

    async def _safe_ack(self, msg: Any) -> None:
        if hasattr(msg, "ack"):
            try:
                await msg.ack()
            except Exception:
                pass

    def _looks_like_submit_echo(self, d: Dict[str, Any]) -> bool:
        # If a publisher echoed submit payloads back onto the same subject
        # (shouldn’t happen in your pipeline, but be defensive)
        return (
            ("prompt_text" in d or "messages" in d) and
            "result" not in d and
            "status" not in d
        )

    def _detect_provider(self) -> str:
        # Placeholder for future: inspect cfg or container
        return "auto"
