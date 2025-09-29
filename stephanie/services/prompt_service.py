from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Dict, Optional

import logging

from stephanie.services.service_protocol import Service
# stephanie/agents/prompt_runner_agent.py
from stephanie.agents.base_agent import BaseAgent

_logger = logging.getLogger(__name__)

config = """
prompt_service:
  name: prompt_service
  save_prompt: true
  save_context: false
  skip_if_completed: false
  model:
    name: ollama/qwen3
    api_base: http://localhost:11434
    api_key: null
  input_keys: ["goal"]   # add nodes
  output_key: prompt_service
  prompt_mode: context
"""

class PromptRunnerAgent(BaseAgent):
    def __init__(self, memory, container, logger):
        super().__init__(config, memory, container, logger)

    async def run(self, context: dict) -> dict:
        prompt = self.prompt_loader.from_context(self.cfg, context=context)
        _logger.info(f"PromptService prompt: {prompt}")
        # Generate response from LLM
        response = self.call_llm(prompt, context=context)
        _logger.info(f"PromptService response: {response}")
        # Store response in context
        context[self.output_key] = response
        return context

class PromptService(Service):
    """
    Service that listens for prompt jobs on the bus, runs them via the LLM backend,
    and publishes results back. Designed to decouple prompt execution from Arena loops.
    """

    def __init__(self, cfg: Dict, memory, container, logger):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger
        self.container = container
        self.semaphore = asyncio.Semaphore(self.cfg.get("max_concurrent", 8))
        self._initialized = False
        self._consumer_tasks: list[asyncio.Task] = []
        self.prompt_runner = PromptRunnerAgent(memory, container, logger)

    # === Service Protocol ===
    def initialize(self, **kwargs) -> None:
        if self._initialized:
            return
        loop = asyncio.get_event_loop()
        # subscribe to bus subjects
        loop.create_task(self._subscribe())
        self._initialized = True

    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy" if self._initialized else "unhealthy",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "metrics": {
                "in_flight": self.semaphore._value,
                "consumer_tasks": len(self._consumer_tasks),
            },
            "dependencies": {"llm_client": "attached" if self.llm_client else "missing"},
        }

    def shutdown(self) -> None:
        for t in self._consumer_tasks:
            t.cancel()
        self._consumer_tasks.clear()
        self._initialized = False

    @property
    def name(self) -> str:
        return "prompt-service-v1"

    # === Bus Subscription ===
    async def _subscribe(self):
        await self.subscribe("prompts.run.request", self._handle_request)

    async def _handle_request(self, payload: Dict[str, Any]):
        """
        Payload example:
        {
            "prompt_id": "...",
            "text": "full prompt text",
            "meta": {...}
        }
        """
        prompt_id = payload.get("prompt_id") or str(uuid.uuid4())
        text = payload.get("text") or ""

        async with self.semaphore:
            try:
                resp = await self.llm_client.call(text)
                result = {
                    "prompt_id": prompt_id,
                    "response": resp,
                    "meta": payload.get("meta") or {},
                }
                await self.publish("prompts.run.result", result)
            except Exception as e:
                err = {
                    "prompt_id": prompt_id,
                    "error": str(e),
                    "meta": payload.get("meta") or {},
                }
                await self.publish("prompts.run.result", err)
