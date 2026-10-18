# stephanie/services/llm_service.py
from __future__ import annotations

import re
import time
from typing import Any, Dict, Optional

import litellm

from stephanie.constants import (API_BASE, API_KEY, NAME, PROMPT_PATH,
                                 SAVE_PROMPT, STRATEGY)
from stephanie.services.service_protocol import Service
from stephanie.utils.llm_utils import remove_think_blocks
from stephanie.utils.secret_utils import redact_secrets


class LLMService(Service):
    """
    Centralized LLM access for agents.
    - Applies symbolic rules (agent/prompt) before each call (if RulesService present)
    - Optional cache via memory.prompts.find_similar_prompt
    - Optional persistence via memory.prompts.save (prompt + response)
    - Uniform error logging

    Public API:
      complete(prompt, *, context, agent_cfg, agent_name, remove_think=True, llm_cfg_override=None)
      chat(messages,  *, context, agent_cfg, agent_name, remove_think=True, llm_cfg_override=None)
    """

    def __init__(self, cfg: Dict, memory, container, logger):
        self.cfg = cfg or {}
        self.memory = memory
        self.container = container
        self.logger = logger
        self._initialized = False
        # Optional Worker-backed model routing (free OpenCode first, then a
        # capped paid opencode-go route). Disabled by default; see
        # config/config.yaml `model_defaults` and docs/worker-free-models.md.
        self._model_defaults: Dict[str, Any] = dict(self.cfg.get("model_defaults") or {})
        self._worker_bridge = None

        # Optional rules service
        try:
            self.rules = container.get("rules")
        except Exception:
            self.rules = None

    @property
    def name(self) -> str:
        return "llm"

    def initialize(self, **kwargs) -> None:
        self._initialized = True
        self.logger.log("LLMServiceInitialized", {"enabled": True})

    def health_check(self) -> Dict[str, Any]:
        ok = True
        try:
            _ = litellm  # import succeeded
        except Exception:
            ok = False
        return {
            "status": "healthy" if (self._initialized and ok) else "uninitialized",
            "has_rules": bool(self.rules),
            "has_prompts_store": hasattr(self.memory, "prompts"),
        }

    def shutdown(self) -> None:
        self._initialized = False

    # --------------------- Optional Worker-backed routing ---------------------

    def _bridge(self):
        if self._worker_bridge is None:
            from stephanie.services.worker_model_bridge import WorkerModelBridge

            self._worker_bridge = WorkerModelBridge(
                base_url=str(self._model_defaults.get("worker_base_url") or "http://127.0.0.1:7317")
            )
        return self._worker_bridge

    def _try_worker_model_defaults(self, prompt: str) -> Optional[str]:
        """Return generated text from a free/paid Worker route, or ``None`` to fall back.

        Never raises: any problem (Worker down, no eligible route, invocation
        error) returns ``None`` so the caller uses its normal provider.
        """
        from stephanie.services.worker_model_bridge import (
            DEFAULT_MAX_PRICE_USD_PER_MILLION,
            DEFAULT_PAID_PROVIDER,
        )

        try:
            bridge = self._bridge()
            route = bridge.select_route(
                allow_free=bool(self._model_defaults.get("allow_free", True)),
                allow_paid=bool(self._model_defaults.get("allow_paid", False)),
                provider=str(self._model_defaults.get("paid_provider") or DEFAULT_PAID_PROVIDER),
                max_price_usd_per_million=float(
                    self._model_defaults.get("max_price_usd_per_million", DEFAULT_MAX_PRICE_USD_PER_MILLION)
                ),
            )
            if route is None:
                return None
            self._model_defaults["_last_route"] = route.id
            self.logger.log("LLMWorkerRoute", {"route": route.id, "tier": route.tier})
            return bridge.submit_prompt(prompt, route=route)
        except Exception as exc:
            self.logger.log("LLMWorkerRouteFailed", {"error_type": type(exc).__name__})
            return None

    # --------------------- Public API ---------------------

    def complete(
        self,
        prompt: str,
        *,
        context: Dict[str, Any],
        agent_cfg: Dict[str, Any],
        agent_name: str,
        remove_think: bool = True,
        llm_cfg_override: Optional[Dict[str, Any]] = None,
        add_to_history: bool = True,
        preferences: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Single turn completion (string prompt). Returns dict:
        {
          "text": str,
          "model_name": str | None,
          "cached": bool
        }
        """
        resolved_cfg = self._resolve_cfg(agent_cfg, context)
        props = llm_cfg_override or self._llm_props(resolved_cfg)

        # Cache fast-path (optional)
        cached_text = self._maybe_fetch_cached_response(
            agent_name=agent_name,
            prompt=prompt,
            agent_cfg=resolved_cfg,
        )
        if cached_text is not None:
            out = remove_think_blocks(cached_text) if remove_think else cached_text
            if add_to_history:
                self._append_prompt_history(context, agent_name, prompt, {"response": out}, preferences)
            return {"text": out, "model_name": props.get(NAME), "cached": True}

        # Optional Worker-backed routing: free OpenCode route if one is admissible,
        # else the cheapest paid opencode-go route within the cap when allowed.
        # Any failure falls through to the normal provider below.
        if self._model_defaults.get("enabled", False):
            worker_text = self._try_worker_model_defaults(prompt)
            if worker_text is not None:
                if add_to_history:
                    self._append_prompt_history(context, agent_name, prompt, {"response": worker_text}, preferences)
                return {
                    "text": worker_text,
                    "model_name": self._model_defaults.get("_last_route") or "worker",
                    "cached": False,
                }

        # Call LLM
        try:
            messages = [{"role": "user", "content": prompt}]
            call_params = {
                "model": props[NAME],
                "messages": messages,
                "api_base": props[API_BASE],
            }
            if props.get(API_KEY):
                call_params["api_key"] = props[API_KEY]
            resp = litellm.completion(**call_params)
            output = resp["choices"][0]["message"]["content"]
            text = remove_think_blocks(output) if remove_think else output

            # Persist prompt/response if enabled
            self._maybe_persist_prompt(
                text=text,
                prompt=prompt,
                agent_name=agent_name,
                agent_cfg=resolved_cfg,
                context=context,
            )

            if add_to_history:
                self._append_prompt_history(context, agent_name, prompt, {"response": text}, preferences)

            return {"text": text, "model_name": props.get(NAME), "cached": False}

        except Exception as e:
            self.logger.log("LLMCallError", {"error_type": type(e).__name__})
            raise

    def chat(
        self,
        messages: list[dict],
        *,
        context: Dict[str, Any],
        agent_cfg: Dict[str, Any],
        agent_name: str,
        remove_think: bool = True,
        llm_cfg_override: Optional[Dict[str, Any]] = None,
        add_to_history: bool = False,
        preferences: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Multi-turn chat (messages). No caching by default.
        Returns same shape as `complete`.
        """
        resolved_cfg = self._resolve_cfg(agent_cfg, context)
        props = llm_cfg_override or self._llm_props(resolved_cfg)

        started = time.perf_counter()
        try:
            resp = litellm.completion(**self._chat_call_params(props, messages))
            result = self._chat_result(resp, props, remove_think, started)
            if add_to_history:
                # Store only the last user prompt + output pair
                last_user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
                self._append_prompt_history(context, agent_name, last_user, {"response": result["text"]}, preferences)

            # No automatic DB persistence for chat; callers can do it if needed
            return result

        except Exception as e:
            self.logger.log("LLMChatError", {"error_type": type(e).__name__})
            raise

    async def achat(
        self,
        messages: list[dict],
        *,
        context: Dict[str, Any],
        agent_cfg: Dict[str, Any],
        agent_name: str,
        remove_think: bool = True,
        llm_cfg_override: Optional[Dict[str, Any]] = None,
        add_to_history: bool = False,
        preferences: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Async counterpart to ``chat`` so request cancellation reaches LiteLLM."""
        resolved_cfg = self._resolve_cfg(agent_cfg, context)
        props = llm_cfg_override or self._llm_props(resolved_cfg)
        started = time.perf_counter()
        try:
            response = await litellm.acompletion(
                **self._chat_call_params(props, messages)
            )
            result = self._chat_result(response, props, remove_think, started)
            if add_to_history:
                last_user = next(
                    (m["content"] for m in reversed(messages) if m.get("role") == "user"),
                    "",
                )
                self._append_prompt_history(
                    context,
                    agent_name,
                    last_user,
                    {"response": result["text"]},
                    preferences,
                )
            return result
        except Exception as e:
            self.logger.log("LLMChatError", {"error_type": type(e).__name__})
            raise

    @staticmethod
    def _chat_call_params(props: Dict[str, Any], messages: list[dict]) -> dict:
        call_params = {
            key: props[key]
            for key in ("timeout", "max_tokens", "temperature", "top_p")
            if props.get(key) is not None
        }
        call_params.update(
            {"model": props[NAME], "messages": messages, "api_base": props[API_BASE]}
        )
        if props.get(API_KEY):
            call_params["api_key"] = props[API_KEY]
        return call_params

    @classmethod
    def _chat_result(cls, response, props, remove_think: bool, started: float) -> dict:
        output = response["choices"][0]["message"]["content"]
        text = remove_think_blocks(output) if remove_think else output
        response_model = cls._response_value(response, "model") or props.get(NAME)
        usage = cls._response_value(response, "usage")
        if hasattr(usage, "model_dump"):
            usage = usage.model_dump()
        elif hasattr(usage, "dict"):
            usage = usage.dict()
        if not isinstance(usage, dict):
            usage = None
        response_id = cls._response_value(response, "id")
        choice = response["choices"][0]
        finish_reason = (
            choice.get("finish_reason")
            if isinstance(choice, dict)
            else getattr(choice, "finish_reason", None)
        )
        configured_model = str(props.get(NAME) or "")
        provider = cls._response_value(response, "provider") or (
            configured_model.split("/", 1)[0] if "/" in configured_model else None
        )
        return {
            "text": text,
            "model_name": response_model,
            "provider": provider,
            "response_id": response_id,
            "usage": usage,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "finish_reason": finish_reason,
            "cached": False,
        }

    @staticmethod
    def _response_value(response, key: str):
        try:
            if isinstance(response, dict):
                return response.get(key)
            return getattr(response, key, None)
        except Exception:
            return None

    # --------------------- Internals ---------------------

    def _resolve_cfg(self, agent_cfg: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply symbolic rules to the agent cfg (agent + prompt layers), return resolved cfg.
        """
        cfg = dict(agent_cfg or {})
        try:
            if self.rules:
                cfg = self.rules.apply_to_agent(cfg, context)  # agent-level overrides (model, adapters, etc.)
                # We let prompt rules be applied by the prompt loader normally; keeping this here for symmetry:
                cfg = self.rules.apply_to_prompt(cfg, context)
        except Exception as e:
            self.logger.log("LLMResolveCfgWarn", {"error": redact_secrets(str(e))})
        return cfg

    def _llm_props(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        model_cfg = (cfg or {}).get("model", {}) or {}
        if not model_cfg:
            # Reasonable defaults if omitted
            return {
                NAME: "ollama/qwen3",
                API_BASE: "http://localhost:11434",
                API_KEY: None,
                "timeout": 60,
            }
        return {
            NAME: model_cfg.get(NAME, "ollama/qwen3"),
            API_BASE: model_cfg.get(API_BASE, "http://localhost:11434"),
            API_KEY: model_cfg.get(API_KEY),
            "timeout": model_cfg.get("timeout", 60),
            "max_tokens": model_cfg.get("max_tokens"),
            "temperature": model_cfg.get("temperature"),
            "top_p": model_cfg.get("top_p"),
        }

    def _maybe_fetch_cached_response(self, *, agent_name: str, prompt: str, agent_cfg: Dict[str, Any]) -> Optional[str]:
        try:
            use_cache = bool(agent_cfg.get("use_memory_for_fast_prompts", False))
            if not (use_cache and hasattr(self.memory, "prompts")):
                return None
            prior = self.memory.prompts.find_similar_prompt(
                agent_name=agent_name,
                prompt_text=prompt,
                strategy=agent_cfg.get(STRATEGY, ""),
                similarity_threshold=0.8,
            )
            if prior:
                chosen = prior[0]
                self.logger.log(
                    "LLMCacheHit",
                    {
                        "agent": agent_name,
                        "strategy": agent_cfg.get(STRATEGY, ""),
                        "prompt_key": agent_cfg.get(PROMPT_PATH, ""),
                        "count": len(prior),
                        "emoji": "📦🔁💬",
                    },
                )
                return chosen.get("response_text")
        except Exception as e:
            self.logger.log("LLMCacheWarn", {"error": str(e)})
        return None

    def _maybe_persist_prompt(
        self,
        *,
        text: str,
        prompt: str,
        agent_name: str,
        agent_cfg: Dict[str, Any],
        context: Dict[str, Any],
    ) -> None:
        try:
            if not (agent_cfg.get(SAVE_PROMPT, False) and hasattr(self.memory, "prompts")):
                return
            self.memory.prompts.save(
                context.get("goal"),
                agent_name=agent_name,
                prompt_key=agent_cfg.get(PROMPT_PATH, ""),
                prompt_text=prompt,
                response=text,
                strategy=agent_cfg.get(STRATEGY, ""),
                pipeline_run_id=context.get("pipeline_run_id"),
                version=agent_cfg.get("version", 1),
            )
        except Exception as e:
            self.logger.log("LLMPersistWarn", {"error": str(e)})

    def _append_prompt_history(
        self,
        context: Dict[str, Any],
        agent: str,
        prompt: str,
        metadata: Dict[str, Any] | None,
        preferences: Dict[str, Any] | None,
    ) -> None:
        try:
            if "prompt_history" not in context:
                context["prompt_history"] = {}
            if agent not in context["prompt_history"]:
                context["prompt_history"][agent] = []
            entry = {
                "prompt": prompt,
                "agent": agent,
                "preferences": preferences or {},
            }
            if metadata:
                entry.update(metadata)
            context["prompt_history"][agent].append(entry)
        except Exception:
            # history is non-critical; never fail the run
            pass
