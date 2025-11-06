from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import litellm
import yaml

from stephanie.agents.base_agent import BaseAgent
from stephanie.services.service_protocol import Service
from stephanie.utils.llm_utils import remove_think_blocks

log = logging.getLogger(__name__)

config = """
prompt_service:
  name: prompt_service
  save_prompt: true
  save_context: false
  skip_if_completed: false
  model:
    name: ollama/qwen:0.5b
    api_base: http://localhost:11434
    api_key: null
    params: {temperature: 0.2}
  input_keys: ["goal"]
  output_key: prompt_service
  prompt_mode: context
  max_concurrent: 8
  timeout: 300
"""

# ---------------- Model spec ----------------

@dataclass
class ModelSpec:
    name: str
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    params: Optional[Dict[str, Any]] = None

    @staticmethod
    def from_cfg(default_cfg: Dict[str, Any], override: Optional[Union[str, Dict[str, Any]]] = None) -> "ModelSpec":
        if override is None:
            base = default_cfg or {}
            m = base.get("model", {}) if "model" in base else base
            return ModelSpec(
                name=m.get("name") or "ollama/qwen:0.5b",
                api_base=m.get("api_base"),
                api_key=m.get("api_key"),
                params=(m.get("params") or {})
            )
        if isinstance(override, str):
            return ModelSpec(name=override)
        # override is a dict
        return ModelSpec(
            name=override.get("name"),
            api_base=override.get("api_base"),
            api_key=override.get("api_key"),
            params=(override.get("params") or {})
        )

# ---------------- Agent wrapper (unchanged behavior) ----------------

class PromptRunnerAgent(BaseAgent):
    def __init__(self, memory, container, logger):
        cfg = yaml.safe_load(config)["prompt_service"]
        super().__init__(cfg, memory, container, logger)

    async def run(self, context: dict) -> dict:
        prompt = context.get("prompt_text") or self.prompt_loader.from_context(self.cfg, context=context)
        # NOTE: This agent’s async_call_llm is used only by PromptService.run_prompt(...)
        response = await asyncio.wait_for(
            self.async_call_llm(prompt, context),
            timeout=self.cfg.get("timeout", 300)
        )
        context[self.output_key] = response
        return context

# ---------------- Service with multi-LLM + training events ----------------

class PromptService(Service):
    """
    Prompt execution with:
      - per-call model override
      - optional system preamble and params
      - multi-LLM competition (parallel)
      - TrainingEventStore logging (pointwise/pairwise)
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

        # default model from config
        self._default_model = ModelSpec.from_cfg(self.cfg)

    # ---- Low-level LLM call (async) ----
    async def _acomplete(
        self,
        *,
        prompt: str,
        model: ModelSpec,
        sys_preamble: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> str:
        messages = []
        if sys_preamble:
            messages.append({"role": "system", "content": sys_preamble})
        messages.append({"role": "user", "content": prompt})

        # merge params (call-time wins)
        call_params = dict((model.params or {}))
        if params:
            call_params.update(params)

        try:
            resp = await litellm.acompletion(
                model=model.name,
                messages=messages,
                api_base=model.api_base or "http://localhost:11434",
                api_key=model.api_key or "",
                **call_params,
            )
            out = resp["choices"][0]["message"]["content"]
            return remove_think_blocks(out)
        except Exception:
            log.exception("PromptService._acomplete failed", extra={"model": model.name})
            return ""

    # ---- Single model, per-call override ----
    async def run_prompt(
        self,
        prompt_text: str,
        context: Optional[Dict[str, Any]],
        *,
        model: Optional[Union[str, Dict[str, Any]]] = None,
        sys_preamble: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> str:
        """
        Execute a prompt with an optional per-call model override and preamble.
        """
        context = context or {}
        request_timeout = timeout or self.timeout
        model_spec = ModelSpec.from_cfg(self.cfg, model)

        async with self._semaphore:
            self._active_requests += 1
            try:
                log.debug(f"run_prompt(model={model_spec.name}) active={self._active_requests}")
                coro = self._acomplete(
                    prompt=prompt_text,
                    model=model_spec,
                    sys_preamble=sys_preamble,
                    params=params,
                )
                return await asyncio.wait_for(coro, timeout=request_timeout)
            except asyncio.TimeoutError:
                log.warning(f"Prompt timed out after {request_timeout}s (model={model_spec.name})")
                return ""
            finally:
                self._active_requests = max(0, self._active_requests - 1)

    # ---- Multi-LLM competition ----
    async def run_prompt_multi(
        self,
        prompt_text: str,
        *,
        models: List[Union[str, Dict[str, Any]]],
        judge: Optional[Callable[[Dict[str, str]], Tuple[str, Dict[str, float]]]] = None,
        sys_preamble: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
        # training events
        dimension: str = "prompt_quality",
        goal_id: Optional[str] = None,
        pipeline_run_id: Optional[int] = None,
        agent_name: str = "prompt_service",
    ) -> Dict[str, Any]:
        """
        Query multiple LLMs in parallel and (optionally) pick a winner via `judge`.

        Returns:
          {
            "outputs": {model_key: text, ...},
            "winner": <model_key or None>,
            "scores": {model_key: float_score, ...}  # if judge provided
          }
        Also logs pointwise/pairwise into TrainingEventStore when judge is provided.
        """
        context = context or {}
        request_timeout = timeout or self.timeout

        model_specs = [ModelSpec.from_cfg(self.cfg, m) for m in models]
        keys = [ms.name for ms in model_specs]

        async with self._semaphore:
            self._active_requests += 1
            try:
                tasks = [
                    asyncio.wait_for(
                        self._acomplete(
                            prompt=prompt_text, model=ms, sys_preamble=sys_preamble, params=params
                        ),
                        timeout=request_timeout
                    )
                    for ms in model_specs
                ]
                outs = await asyncio.gather(*tasks, return_exceptions=True)

                outputs: Dict[str, str] = {}
                for k, o in zip(keys, outs):
                    outputs[k] = ("" if isinstance(o, Exception) else (o or ""))

                winner = None
                scores: Dict[str, float] = {}
                if judge:
                    # judge returns (winner_key, scores_map) where scores_map maps model_key->score
                    winner, scores = judge(outputs)  # type: ignore

                    # ---- TrainingEventStore logging ----
                    tes = getattr(self.memory, "training_events", None)
                    if tes:
                        # pointwise (each output labeled by relative score vs winner)
                        for k, txt in outputs.items():
                            tes.insert_pointwise({
                                "model_key": k,
                                "dimension": dimension,
                                "query_text": prompt_text,
                                "cand_text": txt,
                                "label": 1 if (winner and k == winner) else 0,
                                "weight": 1.0,
                                "trust": float(scores.get(k, 0.0)),
                                "goal_id": goal_id,
                                "pipeline_run_id": pipeline_run_id,
                                "agent_name": agent_name,
                                "source": "prompt_compete",
                                "meta": {"sys_preamble": bool(sys_preamble)}
                            })
                        # pairwise (winner vs others)
                        if winner:
                            pos = outputs[winner]
                            for k, txt in outputs.items():
                                if k == winner:
                                    continue
                                tes.insert_pairwise({
                                    "model_key": winner,              # the "policy" that won
                                    "dimension": dimension,
                                    "query_text": prompt_text,
                                    "pos_text": pos,
                                    "neg_text": txt,
                                    "weight": 1.0,
                                    "trust": float(scores.get(winner, 0.0)),
                                    "goal_id": goal_id,
                                    "pipeline_run_id": pipeline_run_id,
                                    "agent_name": agent_name,
                                    "source": "prompt_compete",
                                    "meta": {"loser": k}
                                })

                return {"outputs": outputs, "winner": winner, "scores": scores}

            except asyncio.TimeoutError:
                log.warning(f"Multi-prompt timed out after {request_timeout}s (models={keys})")
                return {"outputs": {k: "" for k in keys}, "winner": None, "scores": {}}
            finally:
                self._active_requests = max(0, self._active_requests - 1)

    # ---- Legacy sync helper (kept for compatibility) ----
    def call_llm(self, prompt: str, model="ollama/qwen3") -> str:
        # Prefer run_prompt() in async contexts; keep this for legacy callers.
        messages = [{"role": "user", "content": prompt}]
        try:
            response = litellm.completion(
                model=model,
                messages=messages,
                api_base="http://localhost:11434",
                api_key="",
            )
            output = response["choices"][0]["message"]["content"]
            return remove_think_blocks(output)
        except Exception:
            log.exception("PromptService.call_llm failed")
            return ""

    # ---- Service protocol ----
    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "metrics": {
                "in_flight": self.max_concurrent - self._semaphore._value,
                "active_requests": self._active_requests,
                "max_concurrent": self.max_concurrent,
                "timeout": self.timeout,
                "default_model": self._default_model.name,
            },
        }

    def shutdown(self) -> None:
        if hasattr(self.prompt_runner, "executor"):
            self.prompt_runner.executor.shutdown(wait=False)
        log.debug("PromptService shutdown complete")

    def initialize(self, **kwargs) -> None:
        log.debug("PromptService initialized")

    @property
    def name(self) -> str:
        return "prompt-service"
