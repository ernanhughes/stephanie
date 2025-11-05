# legion_service.py

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import litellm
from litellm import ModelResponse

from stephanie.agents.base_agent import BaseAgent
from stephanie.services.service_protocol import Service
from stephanie.utils.llm_utils import remove_think_blocks

log = logging.getLogger(__name__)

# ---------------- ModelSpec (same as before) ----------------

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
        return ModelSpec(
            name=override.get("name"),
            api_base=override.get("api_base"),
            api_key=override.get("api_key"),
            params=(override.get("params") or {})
        )

# Type alias for judge functions
JudgeFunction = Callable[[Dict[str, str]], Tuple[Optional[str], Dict[str, float]]]


# legion_service.py (continued)

class LegionService(Service):
    """
    LegionService: Execute one prompt across many models concurrently.
    
    Features:
      - Parallel execution across N models
      - Per-call model overrides
      - Optional judging (rule-based or LLM-powered)
      - Training event logging (pointwise/pairwise)
      - Semaphore-limited concurrency
      - Timeout protection
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        memory,
        container,
        logger,
    ):
        self.cfg = cfg
        self.memory = memory
        self.container = container
        self.logger = logger or log

        # Concurrency control
        self.max_concurrent = self.cfg.get("max_concurrent", 8)
        self.timeout = self.cfg.get("timeout", 300)
        self._semaphore = asyncio.Semaphore(self.max_concurrent)
        self._active_requests = 0

        # Default fallback model spec
        self._default_model = ModelSpec.from_cfg(self.cfg)

        self._initialized = True

    # ---- Core: Multi-Model Prompt Execution ----

    async def run_legion(
        self,
        prompt: str,
        *,
        models: List[Union[str, Dict[str, Any]]],  # list of model names or specs
        judge: Optional[JudgeFunction] = None,
        sys_preamble: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
        # Logging metadata
        dimension: str = "prompt_quality",
        goal_id: Optional[str] = None,
        pipeline_run_id: Optional[int] = None,
        agent_name: str = "legion_agent",
    ) -> Dict[str, Any]:
        """
        Run the same prompt through multiple models in parallel.
        
        Returns:
          {
            "outputs": {model_name: response_text},
            "winner": model_name or None,
            "scores": {model_name: float_score},
            "timing": float_ms
          }
        """
        start_time = time.time()
        request_timeout = timeout or self.timeout
        context = context or {}

        # Resolve model specs
        model_specs = [ModelSpec.from_cfg(self.cfg, m) for m in models]
        model_keys = [ms.name for ms in model_specs]

        log.info(f"Running legion on {len(model_keys)} models: {model_keys}")

        async with self._semaphore:
            self._active_requests += 1
            try:
                # Launch all completions concurrently
                tasks = [
                    self._call_single_model(
                        prompt=prompt,
                        model_spec=ms,
                        sys_preamble=sys_preamble,
                        params=params,
                        timeout=request_timeout,
                    )
                    for ms in model_specs
                ]
                raw_outputs = await asyncio.gather(*tasks, return_exceptions=True)

                # Process results
                outputs: Dict[str, str] = {}
                errors = []
                for key, out in zip(model_keys, raw_outputs):
                    if isinstance(out, Exception):
                        log.error(f"Model {key} failed: {out}")
                        outputs[key] = ""
                        errors.append({"model": key, "error": str(out)})
                    else:
                        outputs[key] = out or ""

                # Apply judge if provided
                winner: Optional[str] = None
                scores: Dict[str, float] = {}

                if judge:
                    try:
                        winner, scores = judge(outputs)
                        log.info(f"Judge selected winner: {winner}")
                    except Exception as e:
                        log.exception("Judge function failed")
                        winner, scores = None, {}

                    # Log training events
                    await self._log_training_events(
                        prompt=prompt,
                        outputs=outputs,
                        winner=winner,
                        scores=scores,
                        dimension=dimension,
                        goal_id=goal_id,
                        pipeline_run_id=pipeline_run_id,
                        agent_name=agent_name,
                        sys_preamble=bool(sys_preamble),
                    )

                result = {
                    "outputs": outputs,
                    "winner": winner,
                    "scores": scores,
                    "timing": time.time() - start_time,
                    "errors": errors,
                }

                return result

            except asyncio.TimeoutError:
                log.warning(f"Legion run timed out after {request_timeout}s")
                return {
                    "outputs": {k: "" for k in model_keys},
                    "winner": None,
                    "scores": {},
                    "timing": time.time() - start_time,
                    "errors": [{"model": "*", "error": "timeout"}],
                }
            finally:
                self._active_requests = max(0, self._active_requests - 1)

    # ---- Single Model Call (Internal) ----

    async def _call_single_model(
        self,
        *,
        prompt: str,
        model_spec: ModelSpec,
        sys_preamble: Optional[str],
        params: Optional[Dict[str, Any]],
        timeout: int,
    ) -> str:
        """Call a single model with timeout."""
        messages = []
        if sys_preamble:
            messages.append({"role": "system", "content": sys_preamble})
        messages.append({"role": "user", "content": prompt})

        call_params = {**(model_spec.params or {})}
        if params:
            call_params.update(params)

        try:
            response: ModelResponse = await asyncio.wait_for(
                litellm.acompletion(
                    model=model_spec.name,
                    messages=messages,
                    api_base=model_spec.api_base or "http://localhost:11434",
                    api_key=model_spec.api_key or "",
                    **call_params,
                ),
                timeout=timeout,
            )
            content = response.choices[0].message.content
            return remove_think_blocks(content)
        except Exception as e:
            raise RuntimeError(f"LLM call failed for {model_spec.name}: {str(e)}")

    # ---- Training Event Logging ----

    async def _log_training_events(
        self,
        *,
        prompt: str,
        outputs: Dict[str, str],
        winner: Optional[str],
        scores: Dict[str, float],
        dimension: str,
        goal_id: Optional[str],
        pipeline_run_id: Optional[int],
        agent_name: str,
        sys_preamble: bool,
    ):
        """Log pointwise and pairwise comparisons to TrainingEventStore."""
        tes = getattr(self.memory, "training_events", None)
        if not tes:
            return

        # Pointwise: each output scored relative to best
        for model_key, text in outputs.items():
            label = 1 if (winner and model_key == winner) else 0
            trust = float(scores.get(model_key, 0.0))

            tes.insert_pointwise({
                "model_key": model_key,
                "dimension": dimension,
                "query_text": prompt,
                "cand_text": text,
                "label": label,
                "weight": 1.0,
                "trust": trust,
                "goal_id": goal_id,
                "pipeline_run_id": pipeline_run_id,
                "agent_name": agent_name,
                "source": "legion_competition",
                "meta": {"sys_preamble": sys_preamble},
            })

        # Pairwise: winner vs losers
        if winner and winner in outputs:
            pos_text = outputs[winner]
            for model_key, neg_text in outputs.items():
                if model_key == winner:
                    continue
                tes.insert_pairwise({
                    "model_key": winner,
                    "dimension": dimension,
                    "query_text": prompt,
                    "pos_text": pos_text,
                    "neg_text": neg_text,
                    "weight": 1.0,
                    "trust": float(scores.get(winner, 0.0)),
                    "goal_id": goal_id,
                    "pipeline_run_id": pipeline_run_id,
                    "agent_name": agent_name,
                    "source": "legion_competition",
                    "meta": {"loser": model_key},
                })

    # ---- Health & Lifecycle ----

    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "service": "legion-service",
            "metrics": {
                "in_flight": self.max_concurrent - self._semaphore._value,
                "active_requests": self._active_requests,
                "max_concurrent": self.max_concurrent,
                "timeout": self.timeout,
                "default_model": self._default_model.name,
            },
        }

    def shutdown(self) -> None:
        log.debug("LegionService shutdown complete")

    def initialize(self, **kwargs) -> None:
        log.debug("LegionService initialized")

    @property
    def name(self) -> str:
        return "legion-service"