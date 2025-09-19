# stephanie/services/llm_service.py
from __future__ import annotations

import re
import traceback
from typing import Any, Dict, Optional

import litellm

from stephanie.constants import (API_BASE, API_KEY, NAME, PROMPT_PATH,
                                 SAVE_PROMPT, STRATEGY)
from stephanie.services.service_protocol import Service


def _remove_think_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


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
            out = _remove_think_blocks(cached_text) if remove_think else cached_text
            if add_to_history:
                self._append_prompt_history(context, agent_name, prompt, {"response": out}, preferences)
            return {"text": out, "model_name": props.get(NAME), "cached": True}

        # Call LLM
        try:
            messages = [{"role": "user", "content": prompt}]
            resp = litellm.completion(
                model=props[NAME],
                messages=messages,
                api_base=props[API_BASE],
                api_key=props.get(API_KEY, ""),
            )
            output = resp["choices"][0]["message"]["content"]
            text = _remove_think_blocks(output) if remove_think else output

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
            self.logger.log("LLMCallError", {"error": str(e), "traceback": traceback.format_exc()})
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

        try:
            resp = litellm.completion(
                model=props[NAME],
                messages=messages,
                api_base=props[API_BASE],
                api_key=props.get(API_KEY, ""),
            )
            output = resp["choices"][0]["message"]["content"]
            text = _remove_think_blocks(output) if remove_think else output

            if add_to_history:
                # Store only the last user prompt + output pair
                last_user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
                self._append_prompt_history(context, agent_name, last_user, {"response": text}, preferences)

            # No automatic DB persistence for chat; callers can do it if needed
            return {"text": text, "model_name": props.get(NAME), "cached": False}

        except Exception as e:
            self.logger.log("LLMChatError", {"error": str(e), "traceback": traceback.format_exc()})
            raise

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
            self.logger.log("LLMResolveCfgWarn", {"error": str(e)})
        return cfg

    def _llm_props(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        model_cfg = (cfg or {}).get("model", {}) or {}
        if not model_cfg:
            # Reasonable defaults if omitted
            return {
                NAME: "ollama/qwen3",
                API_BASE: "http://localhost:11434",
                API_KEY: None,
            }
        return {
            NAME: model_cfg.get(NAME, "ollama/qwen3"),
            API_BASE: model_cfg.get(API_BASE, "http://localhost:11434"),
            API_KEY: model_cfg.get(API_KEY),
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
