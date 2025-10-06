# stephanie/agents/prompt_runner_agent.py
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

import yaml

from stephanie.agents.base_agent import BaseAgent
from stephanie.services.service_protocol import Service

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
  input_keys: ["goal"]
  output_key: prompt_service
  prompt_mode: context
  max_concurrent: 8
  timeout: 300  # 5 minutes
"""

class PromptRunnerAgent(BaseAgent):
    def __init__(self, memory, container, logger):
        cfg = yaml.safe_load(config)["prompt_service"]
        super().__init__(cfg, memory, container, logger)

    async def run(self, context: dict) -> dict:
        prompt = context.get("prompt_text")
        if not prompt:
            prompt = self.prompt_loader.from_context(self.cfg, context=context)

        response = await asyncio.wait_for(
            self.async_call_llm(prompt, context),
            timeout=self.cfg.get("timeout", 300)
        )
        context[self.output_key] = response
        return context

class PromptService(Service):
    """
    Fixed service that properly handles async LLM calls with timeout protection
    and proper thread management.
    """

    def __init__(self, cfg: Dict, memory, container, logger):
        self.cfg = cfg
        self.memory = memory
        self.container = container
        self.logger = logger

        self.max_concurrent = self.cfg.get("max_concurrent", 8)
        self.timeout = self.cfg.get("timeout", 300)
        self._semaphore = asyncio.Semaphore(self.max_concurrent)

        self.prompt_runner = PromptRunnerAgent(memory, container, logger)
        self._active_requests = 0
        self._initialized = False

    async def run_prompt(
        self,
        prompt_text: str,
        context: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> str:
        """
        Run a prompt directly (no bus).
        """
        context = context or {}
        context["prompt_text"] = prompt_text
        request_timeout = timeout or self.timeout

        async with self._semaphore:
            self._active_requests += 1
            try:
                _logger.info(f"Running direct prompt (active={self._active_requests})")
                result_ctx = await asyncio.wait_for(
                    self.prompt_runner.run(context),
                    timeout=request_timeout,
                )
                return result_ctx.get(self.prompt_runner.output_key, "")
            except asyncio.TimeoutError:
                _logger.warning(f"Prompt timed out after {request_timeout}s")
                raise RuntimeError(f"Prompt timed out after {request_timeout}s")
            finally:
                self._active_requests = max(0, self._active_requests - 1)

    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "metrics": {
                "in_flight": self.max_concurrent - self._semaphore._value,
                "active_requests": self._active_requests,
                "max_concurrent": self.max_concurrent,
                "timeout": self.timeout,
            },
        }

    def shutdown(self) -> None:
        if hasattr(self.prompt_runner, "executor"):
            self.prompt_runner.executor.shutdown(wait=False)
        _logger.info("PromptServiceDirect shutdown complete")

    # === Service Protocol ===
    def initialize(self, **kwargs) -> None:
        _logger.info("PromptService initialized and subscribed to bus")


    @property
    def name(self) -> str:
        return "prompt-service"  # Version bump to indicate fixed version
