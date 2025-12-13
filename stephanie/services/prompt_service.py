from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import litellm
import yaml

from stephanie.agents.base_agent import BaseAgent
from stephanie.constants import (BUS_STREAM, SUBJ_RESULT_LEG_W,
                                 SUBJ_RESULT_NS_T, SUBJ_RESULT_NS_W,
                                 SUBJ_SUBMIT, SUBJ_SUBMIT_LEG, SUBJ_SUBMIT_NS)
from stephanie.services.bus.events.prompt_job import PromptJob
from stephanie.services.service_protocol import Service
from stephanie.types.model import ModelSpec
from stephanie.utils.llm_utils import remove_think_blocks

log = logging.getLogger(__name__)


class LLMRole(str, Enum):
    CRITIC = "critic"
    PLANNER = "planner"
    IDEATOR = "ideator"
    SUMMARIZER = "summarizer"
    Years = "explainer"
    CRITIC_FEASIBILITY = "critic_feasibility"
    CRITIC_RISK = "critic_risk"
    EXECUTOR = "executor"
    EXPLAINER = "explainer"
    REFINER = "refiner"
    BLOG = "blog"

DEFAULT_ROLE_SYSTEM_PROMPTS: Dict[LLMRole, str] = {
    LLMRole.CRITIC_FEASIBILITY: (
        "You are an AI research feasibility auditor. Your ONLY task is to score how "
        "technically feasible and testable a research idea is with CURRENT tools and resources. "
        "\n\n"
        "SCORING RULES (STRICT):\n"
        "- 1.0 = Trivial (uses standard libraries, public datasets, <1hr runtime)\n"
        "- 0.7 = Straightforward (known methods, public data, <1 day runtime)\n"
        "- 0.4 = Challenging (requires novel engineering, may need data collection)\n"
        "- 0.1 = Nearly impossible (requires breakthroughs, proprietary data, or massive compute)\n"
        "- 0.0 = Impossible (violates known physics/math)\n"
        "\n"
        "ASSESS THESE FACTORS:\n"
        "1. Technical plausibility with current AI tools (PyTorch, Hugging Face, etc.)\n"
        "2. Testability with PUBLIC datasets or easily collectible data\n"
        "3. Clear, measurable success/failure criteria\n"
        "4. Realistic compute requirements (<4 A100 GPUs, <1 week runtime)\n"
        "\n"
        "BE CONSERVATIVE. When in doubt, score LOWER. "
        "DO NOT consider novelty, impact, or elegance—ONLY feasibility.\n"
        "\n"
        "OUTPUT FORMAT: A SINGLE FLOAT between 0.0 and 1.0. NO TEXT, NO EXPLANATION."
    ),
    LLMRole.CRITIC_RISK: (
        "You are an AI research safety auditor. Your ONLY task is to flag ideas that pose "
        "significant safety, ethics, or reproducibility risks.\n\n"
        "RED FLAG CATEGORIES (score 1.0 if ANY apply):\n"
        "- Human subjects without IRB approval\n"
        "- Bio/chem safety hazards (pathogens, toxins)\n"
        "- Weaponization potential\n"
        "- Privacy violations (non-public personal data)\n"
        "- Irreproducible claims (no public code/data)\n"
        "- Unverifiable assertions (no clear success metric)\n"
        "\n"
        "OUTPUT FORMAT: 1.0 if HIGH RISK, 0.0 otherwise. NO TEXT, NO EXPLANATION."
    ),
    LLMRole.EXECUTOR: (
        "You are an AI research code generator. Your ONLY task is to convert research methods "
        "into executable Python code using standard libraries (PyTorch, scikit-learn, etc.).\n\n"
        "RULES:\n"
        "- Use ONLY publicly available datasets (CIFAR-10, MNIST, GLUE, etc.)\n"
        "- Include clear success metrics (accuracy, F1, runtime)\n"
        "- Add safety checks (memory limits, timeout guards)\n"
        "- Output COMPLETE, runnable code in one block\n"
        "- NO hypothetical functions or placeholders\n"
        "\n"
        "If the method cannot be implemented with public tools/data, output:\n"
        "'''INFEASIBLE: [reason]'''"
    ),
    LLMRole.REFINER: (
        "You are an AI research collaborator refining ideas WITH HUMAN FEEDBACK. "
        "The human has provided constraints via sliders:\n"
        "- Novelty: {novelty_level} (0=minimal, 1=radical)\n"
        "- Feasibility: {feasibility_level} (0=moonshot, 1=straightforward)\n"
        "- Risk tolerance: {risk_level} (0=none, 1=high)\n"
        "\n"
        "Your job: Rewrite the idea to satisfy these constraints while preserving core value.\n"
        "Output ONLY the refined idea in this JSON format:\n"
        "{\n"
        '  "title": "Short title",\n'
        '  "hypothesis": "Clear claim",\n'
        '  "method": "Concrete validation approach",\n'
        '  "constraints_applied": ["list", "of", "changes"]\n'
        "}"
    ),
    # Keep your existing roles below this line
    LLMRole.CRITIC: (
        "You are a rigorous but constructive AI research critic. "
        "Your job is to find weaknesses, missing assumptions, and failure modes "
        "in ideas, arguments, or experimental designs. Be precise and concrete."
    ),
    LLMRole.PLANNER: (
        "You are an AI experiment and project planner. "
        "Given a research idea or goal, produce a concrete, step-by-step plan "
        "with milestones, resources, risks, and success criteria."
    ),
    LLMRole.IDEATOR: (
        "You are a creative but technically grounded AI research collaborator. "
        "Your job is to propose novel, plausible research ideas that connect "
        "the provided concepts or findings."
    ),
    LLMRole.SUMMARIZER: (
        "You are an expert summarizer for AI/ML research. "
        "Produce clear, faithful summaries optimized for later reuse as knowledge."
    ),
    LLMRole.EXPLAINER: (
        "You are a teacher explaining AI/ML concepts to an informed engineer. "
        "Use concise, accurate language and concrete examples."
    ),
    LLMRole.BLOG: (
        "You are a technical blog writer for AI/ML research papers. "
        "Your job is to turn paper summaries, section notes, and graph context into "
        "clear, trustworthy blog sections for an informed engineer. "
        "You avoid hype, stay faithful to the provided context, and explain ideas with "
        "concrete intuition, short paragraphs, and occasional bullet lists."
    ),
}

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


class PromptRunnerAgent(BaseAgent):
    def __init__(self, memory, container, logger):
        cfg = yaml.safe_load(config)["prompt_service"]
        super().__init__(cfg, memory, container, logger)

    async def run(self, context: dict) -> dict:
        prompt = context.get("prompt_text") or self.prompt_loader.from_context(
            self.cfg, context=context
        )
        response = await asyncio.wait_for(
            self.async_call_llm(prompt, context),
            timeout=self.cfg.get("timeout", 300),
        )
        context[self.output_key] = response
        return context


# ---------------- Service with multi-LLM + training events ----------------


class PromptService(Service):
    """
    Prompt execution service with an integrated bus worker:
      - Consumes PromptJob from bus subjects
      - Calls LLM (messages or plain prompt)
      - Publishes results to return topic
      - Supports direct API use: run_prompt, run_prompt_multi
      - Optional: wait_many / try_get for callers that want result rendezvous
    """

    def __init__(self, cfg: Dict, memory, container, logger):
        self.cfg = cfg
        self.memory = memory
        self.container = container
        self.logger = logger

        self.max_concurrent = int(self.cfg.get("max_concurrent", 8))
        self.timeout = int(self.cfg.get("timeout", 300))
        self._semaphore = asyncio.Semaphore(self.max_concurrent)

        # default model from config
        self._default_model = ModelSpec.from_cfg(self.cfg)

        # runtime / worker state
        self._active_requests = 0
        self._worker_started = False
        self._result_inbox: Dict[str, str] = {}  # job_id -> text
        self._subs_ready = False
        self._task_group: set[asyncio.Task] = set()


    # ---- Low-level LLM calls -------------------------------------------------

    def _build_sys_preamble(
        self,
        role: Optional[LLMRole],
        sys_preamble: Optional[str],
    ) -> Optional[str]:
        """
        Combine a role-based system prompt (if any) with an explicit sys_preamble.

        Priority:
          - If no role: just return sys_preamble.
          - If role set but no sys_preamble: return the default role prompt.
          - If both set: role prompt first, then user sys_preamble.
        """
        if role is None:
            return sys_preamble

        base = DEFAULT_ROLE_SYSTEM_PROMPTS.get(role)
        if not base:
            return sys_preamble

        if sys_preamble:
            return f"{base}\n\n{sys_preamble}"
        return base

    async def _acomplete_messages(
        self,
        *,
        model: ModelSpec,
        messages: List[Dict[str, Any]],
        params: Optional[Dict[str, Any]] = None,
    ) -> str:
        call_params = dict(model.params or {})
        if params:
            call_params.update(params)
        try:
            resp = await litellm.acompletion(
                model=model.name,
                messages=messages,
                api_base=(model.api_base or "http://localhost:11434"),
                api_key=(model.api_key or ""),
                **call_params,
            )
            out = resp["choices"][0]["message"]["content"]
            log.info(f"PS success: model={model.name} {out[:60]!r}...")
            return remove_think_blocks(out)
        except Exception:
            log.exception("PS failed extra=%s", model.name)
            return ""

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
        return await self._acomplete_messages(
            model=model, messages=messages, params=params
        )

    # ---- Public: single model ------------------------------------------------

    async def run_prompt(
        self,
        prompt_text: str,
        context: Optional[Dict[str, Any]],
        *,
        model: Optional[Union[str, Dict[str, Any]]] = None,
        role: Optional[LLMRole] = None,
        sys_preamble: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> str:
        request_timeout = int(timeout or self.timeout)
        model_spec = ModelSpec.from_cfg(self.cfg, model)
        async with self._semaphore:
            self._active_requests += 1
            try:
                log.debug(
                    f"run_prompt(model={model_spec.name}) active={self._active_requests}"
                )
                effective_sys = self._build_sys_preamble(role, sys_preamble)
                coro = self._acomplete(
                    prompt=prompt_text,
                    model=model_spec,
                    sys_preamble=effective_sys,
                    params=params,
                )
                return await asyncio.wait_for(coro, timeout=request_timeout)
            except asyncio.TimeoutError:
                log.warning(
                    f"Prompt timed out after {request_timeout}s (model={model_spec.name})"
                )
                return ""
            finally:
                self._active_requests = max(0, self._active_requests - 1)

    # ---- Public: multi-LLM competition --------------------------------------

    async def run_prompt_multi(
        self,
        prompt_text: str,
        *,
        models: List[Union[str, Dict[str, Any]]],
        judge: Optional[callable] = None,
        role: Optional[LLMRole] = None,  # <-- NEW
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
        context = context or {}
        request_timeout = int(timeout or self.timeout)

        model_specs = [ModelSpec.from_cfg(self.cfg, m) for m in models]
        keys = [ms.name for ms in model_specs]
        effective_sys = self._build_sys_preamble(role, sys_preamble)

        async with self._semaphore:
            self._active_requests += 1
            try:
                tasks = [
                    asyncio.wait_for(
                        self._acomplete(
                            prompt=prompt_text,
                            model=ms,
                            sys_preamble=effective_sys,
                            params=params,
                        ),
                        timeout=request_timeout,
                    )
                    for ms in model_specs
                ]
                outs = await asyncio.gather(*tasks, return_exceptions=True)
                outputs = {
                    k: ("" if isinstance(o, Exception) else (o or ""))
                    for k, o in zip(keys, outs)
                }

                winner, scores = None, {}
                if judge:
                    winner, scores = judge(outputs)  # type: ignore

                    tes = getattr(self.memory, "training_events", None)
                    if tes:
                        for k, txt in outputs.items():
                            tes.insert_pointwise(
                                {
                                    "model_key": k,
                                    "dimension": dimension,
                                    "query_text": prompt_text,
                                    "cand_text": txt,
                                    "label": 1
                                    if (winner and k == winner)
                                    else 0,
                                    "weight": 1.0,
                                    "trust": float(scores.get(k, 0.0)),
                                    "goal_id": goal_id,
                                    "pipeline_run_id": pipeline_run_id,
                                    "agent_name": agent_name,
                                    "source": "prompt_compete",
                                    "meta": {
                                        "sys_preamble": bool(sys_preamble)
                                    },
                                }
                            )
                        if winner:
                            pos = outputs[winner]
                            for k, txt in outputs.items():
                                if k == winner:
                                    continue
                                tes.insert_pairwise(
                                    {
                                        "model_key": winner,
                                        "dimension": dimension,
                                        "query_text": prompt_text,
                                        "pos_text": pos,
                                        "neg_text": txt,
                                        "weight": 1.0,
                                        "trust": float(
                                            scores.get(winner, 0.0)
                                        ),
                                        "goal_id": goal_id,
                                        "pipeline_run_id": pipeline_run_id,
                                        "agent_name": agent_name,
                                        "source": "prompt_compete",
                                        "meta": {"loser": k},
                                    }
                                )
                return {"outputs": outputs, "winner": winner, "scores": scores}
            except asyncio.TimeoutError:
                log.warning(
                    f"Multi-prompt timed out after {request_timeout}s (models={keys})"
                )
                return {
                    "outputs": {k: "" for k in keys},
                    "winner": None,
                    "scores": {},
                }
            finally:
                self._active_requests = max(0, self._active_requests - 1)

    # ---- Worker: bus integration --------------------------------------------

    async def _ensure_bus_bindings(self) -> None:
        bus = self.memory.bus
        try:
            await bus.ensure_stream(
                BUS_STREAM,
                [
                    f"{BUS_STREAM}.>",
                    SUBJ_SUBMIT,
                    SUBJ_SUBMIT_NS,
                    SUBJ_RESULT_NS_W,
                    SUBJ_SUBMIT_LEG,
                    SUBJ_RESULT_LEG_W,
                ],
            )
        except Exception as e:
            log.warning("PS _ensure_bus_bindings failed: %s", e)

        if not self._subs_ready:
            # Subscribe to both submit subjects; process in background
            async def _submit_cb(msg):
                self._spawn(self._handle_submit_msg(msg))

            # Listen to results wildcards to populate inbox (for wait_many/try_get)
            async def _result_cb(msg):
                try:
                    data = (
                        msg.data
                        if isinstance(msg.data, dict)
                        else json.loads(msg.data.decode("utf-8"))
                    )
                except Exception:
                    return
                jid = data.get("job_id")
                result = data.get("result", {})
                text = (
                    result.get("text")
                    or result.get("content")
                    or (
                        result.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content")
                        if isinstance(result.get("choices"), list)
                        else None
                    )
                    or result.get("output")
                )
                if text is None and "error" in data:
                    text = f"[error] {data['error']}"
                if jid and isinstance(text, str):
                    self._result_inbox[jid] = text
                if hasattr(msg, "ack"):
                    try:
                        await msg.ack()
                    except Exception:
                        pass

            # await bus.subscribe(
            #     subject=SUBJ_SUBMIT, queue="prompt-workers", handler=_submit_cb
            # )
            # await bus.subscribe(
            #     subject=SUBJ_RESULT_WC, queue=None, handler=_result_cb
            # )

            self._subs_ready = True
            log.info("PromptService bus bindings ready")

    def _spawn(self, coro: "asyncio.Future[Any]"):
        t = asyncio.create_task(coro)
        self._task_group.add(t)
        t.add_done_callback(lambda tt: self._task_group.discard(tt))

    async def _handle_submit_msg(self, msg) -> None:
        # Parse
        try:
            log.info("PS ReceivedJob: %s", msg)
            raw = (
                msg.data
                if isinstance(msg.data, dict)
                else json.loads(msg.data.decode("utf-8"))
            )
        except Exception as e:
            log.exception("PS %s", e)
            if hasattr(msg, "nak"):
                try:
                    await msg.nak()
                except Exception:
                    pass
            return

        # Validate
        try:
            log.info("PS ValidatingJob: %s", raw)
            job = PromptJob.model_validate(raw)  # Pydantic v2
        except Exception as e:
            log.error("PS ValidationError %s", e)
            if hasattr(msg, "nak"):
                try:
                    await msg.nak()
                except Exception:
                    pass
            return

        # Execute (messages or prompt_text)
        try:
            log.info("PromptService.ExecutingJob: %s", job.job_id)
            model_spec = ModelSpec.from_cfg(self.cfg, job.model)
            params = {}
            # Normalized exec path with concurrency limits
            async with self._semaphore:
                self._active_requests += 1
                try:
                    if job.messages:
                        # inject system if provided but missing
                        messages = list(job.messages)
                        if job.system:
                            if (
                                not messages
                                or messages[0].get("role") != "system"
                            ):
                                messages = [
                                    {"role": "system", "content": job.system}
                                ] + messages
                        text = await self._acomplete_messages(
                            model=model_spec, messages=messages, params=params
                        )
                        log.info(f"PS success: {text[:60]}...")
                    else:
                        text = await self._acomplete(
                            prompt=job.prompt_text or "",
                            model=model_spec,
                            sys_preamble=job.system,
                            params=params,
                        )
                        log.info(f"PS success: {text[:60]!r}...")
                finally:
                    self._active_requests = max(0, self._active_requests - 1)

            # Publish result
            ret = job.return_topic or SUBJ_RESULT_NS_T.format(job=job.job_id)
            payload = {
                "job_id": job.job_id,
                "scorable_id": job.scorable_id,
                "result": {"text": text},
            }
            log.info("PS PublishingResult: %s", ret)
            await self.memory.bus.publish(ret, payload)
            if hasattr(msg, "ack"):
                try:
                    await msg.ack()
                except Exception:
                    pass
            log.info("PS ResultSent %s %s", job.job_id, ret)
        except Exception as e:
            log.exception("PS ExecuteError %s", str(e))
            ret = job.return_topic or SUBJ_RESULT_NS_T.format(job=job.job_id)
            try:
                await self.memory.bus.publish(
                    ret, {"job_id": job.job_id, "error": str(e)}
                )
            except Exception:
                pass
            if hasattr(msg, "nak"):
                try:
                    await msg.nak()
                except Exception:
                    pass

    # ---- Optional rendezvous helpers (used by pollinator fast path) ----------

    async def wait_many(
        self,
        tickets: List[Dict[str, str] | Tuple[str, str]],
        *,
        timeout_s: float,
    ) -> List[str]:
        await self._ensure_bus_bindings()
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
                if jid in self._result_inbox:
                    out[jid] = self._result_inbox.pop(jid)
                    want_set.remove(jid)
            if want_set:
                await asyncio.sleep(0.05)

        order.sort(key=lambda x: x[1])
        return [out[jid] for (jid, _) in order if jid in out]

    async def try_get(self, job_id: str) -> Optional[str]:
        await self._ensure_bus_bindings()
        return self._result_inbox.pop(job_id, None)

    # ---- Service protocol ----------------------------------------------------

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
                "worker_started": self._worker_started,
                "subs_ready": self._subs_ready,
            },
        }

    def shutdown(self) -> None:
        for t in list(self._task_group):
            t.cancel()
        log.debug("PromptService shutdown complete")

    def initialize(self, **kwargs) -> None:
        # Idempotent start of the worker + result wildcard subscriptions
        if self._worker_started:
            return
        self._worker_started = True
        self._spawn(self._ensure_bus_bindings())
        log.debug("PromptService initialized")

    @property
    def name(self) -> str:
        return "prompt-service"
