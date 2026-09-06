# stephanie/portfolio/experiment/ollama_provider.py
"""Experiment-local Ollama transport (scaffolding, not canonical).

Stage 1 Phase 4 (Writer provider import) has not happened yet, so the
experiment talks to local Ollama directly. Temperature 0 for arm fairness.
"""
from __future__ import annotations

import json
import time
import urllib.request
from datetime import datetime
from uuid import uuid4
from stephanie.models.model import Model
from stephanie.models.provider import ModelProvider
from stephanie.models.request import ModelRequest
from stephanie.models.response import ModelResponse
from stephanie.models.usage import ModelUsage


class OllamaChatProvider(ModelProvider):
    def __init__(self, base_url: str = "http://localhost:11434",
                 num_predict: int = 256, timeout_s: int = 600):
        self.base_url = base_url.rstrip("/")
        self.num_predict = num_predict
        self.timeout_s = timeout_s

    def supports(self, model: Model) -> bool:
        return model.provider == "ollama"

    async def invoke(self, request: ModelRequest) -> ModelResponse:
        import asyncio

        return await asyncio.to_thread(self._invoke_sync, request)

    def _invoke_sync(self, request: ModelRequest) -> ModelResponse:
        model = request.model if isinstance(request.model, Model) else Model.from_ref(str(request.model))
        payload = {
            "model": model.name,
            "messages": request.to_messages(),
            "stream": False,
            # Thinking blocks bury task output and burn the token budget;
            # classification-style review tasks run with thinking disabled.
            "think": False,
            "options": {"temperature": 0, "num_predict": self.num_predict},
        }
        body = json.dumps(payload).encode()
        started = time.perf_counter()
        http_request = urllib.request.Request(
            f"{self.base_url}/api/chat", data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(http_request, timeout=self.timeout_s) as response:
            data = json.loads(response.read().decode())
        latency_ms = (time.perf_counter() - started) * 1000.0
        text = (data.get("message") or {}).get("content", "")
        try:
            from stephanie.utils.llm_utils import remove_think_blocks

            text = remove_think_blocks(text)
        except Exception:
            pass
        usage = ModelUsage(
            input_tokens=data.get("prompt_eval_count"),
            output_tokens=data.get("eval_count"),
            latency_ms=latency_ms,
        )
        if usage.total_tokens is None and usage.input_tokens is not None:
            usage.total_tokens = (usage.input_tokens or 0) + (usage.output_tokens or 0)
        return ModelResponse(
            request_id=str(uuid4()),
            model_id=model.id,
            provider="ollama",
            output_text=text,
            usage=usage,
            latency_ms=latency_ms,
            finish_reason=data.get("done_reason") or "stop",
            created_at=datetime.utcnow(),
            metadata={"trace_id": request.trace_id} if request.trace_id else {},
        )
