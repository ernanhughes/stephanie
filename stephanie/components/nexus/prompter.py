# stephanie/components/nexus/prompter.py
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from stephanie.services.bus.prompt_client import PromptClient
from stephanie.services.bus.events.prompt_job import Priority

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Config & data types                                                         #
# --------------------------------------------------------------------------- #

@dataclass(slots=True)
class NexusPromptConfig:
    """
    Default configuration for how NexusPrompter talks to the prompt bus.

    This is intentionally small; all the heavy routing (pools, sharding, etc.)
    lives in PromptClient + PromptJob.
    """
    model: str = "auto"              # let gateway pick, or explicit model name
    target_pool: str = "auto"
    priority: Priority = Priority.normal
    response_format: str = "text"

    timeout_s: float = 90.0          # per batch of k expansions

    # Basic sampling params; passed through in prompt "params"
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512
    seed: Optional[int] = None

    # Default number of candidates per parent if caller doesn't override
    default_k: int = 4


@dataclass(slots=True)
class NexusPromptTemplates:
    """
    Simple templates for thought expansion.

    You can override these per-call if a specific agent wants a different style.
    """
    system: str = (
        "You are a careful thought refiner inside a cognitive graph called Nexus. "
        "The overall goal is:\n\n{goal}\n\n"
        "You will create alternative or improved versions of a given thought, "
        "keeping it on-goal, faithful, and clear."
    )
    user: str = (
        "Original thought:\n\n"
        "{thought}\n\n"
        "You are generating candidate #{index}.\n"
        "Rewrite or extend this thought so that it is clearer, more useful, and "
        "better aligned with the goal. Keep it concise and self-contained."
    )


@dataclass(slots=True)
class NexusPromptTicket:
    """
    Fire-and-forget handle for an offloaded expansion batch.

    This is useful if an agent wants to schedule expansions and come back later.
    """
    job_id: str
    subject: str
    scorable_id: str


@dataclass(slots=True)
class NexusPromptResult:
    """
    Final result for one expansion.

    `raw` is the full payload from the bus; `text` is the best-effort extraction
    of the candidate text that higher-level agents (Pollinator, Blossom) will use.
    """
    job_id: str
    scorable_id: str
    text: Optional[str]
    raw: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# NexusPrompter                                                               #
# --------------------------------------------------------------------------- #

class NexusPrompter:
    """
    High-level LLM expansion helper for Nexus.

    Responsibilities:
      - Given a scorable (id + text) and a goal, build k prompts.
      - Offload them via PromptClient to the ZeroMQ-backed bus.
      - Optionally wait for results and extract candidate texts.

    This is the only thing Pollinator / Blossom should use for "ask the LLM to
    grow this thought".
    """

    def __init__(
        self,
        cfg: NexusPromptConfig,
        prompt_client: PromptClient,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.cfg = cfg
        self.prompt_client = prompt_client
        self.log = logger or log
        self._default_templates = NexusPromptTemplates()

    # ------------------------------------------------------------------ public

    async def expand_scorable(
        self,
        *,
        scorable_id: str,
        scorable_text: str,
        goal_text: str,
        k: Optional[int] = None,
        templates: Optional[NexusPromptTemplates] = None,
        model: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        fire_and_forget: bool = False,
    ) -> List[NexusPromptResult] | List[NexusPromptTicket]:
        """
        Expand a single scorable into k candidate texts.

        If fire_and_forget=True, returns tickets (job_id, subject, scorable_id)
        and DOES NOT wait for responses.

        If fire_and_forget=False, waits for all results (or timeout) and returns
        NexusPromptResult objects, one per successfully completed job.
        """
        if k is None:
            k = self.cfg.default_k

        tpls = templates or self._default_templates

        prompts = self._build_prompts(
            scorable_id=scorable_id,
            scorable_text=scorable_text,
            goal_text=goal_text,
            k=k,
            templates=tpls,
        )

        # Publish k jobs via PromptClient
        tickets = await self.prompt_client.offload_many(
            scorable_id=scorable_id,
            prompts=prompts,
            model=model or self.cfg.model,
            target_pool=self.cfg.target_pool,
            priority=self.cfg.priority,
            group_key=meta.get("group_key") if meta else None,
            meta=meta,
            response_format=self.cfg.response_format,
        )

        if fire_and_forget:
            return [
                NexusPromptTicket(job_id=jid, subject=subj, scorable_id=scorable_id)
                for (jid, subj) in tickets
            ]

        # Synchronous "expand and wait" path
        if not hasattr(self.prompt_client, "wait_many"):
            raise RuntimeError(
                "PromptClient.wait_many(...) is required for "
                "NexusPrompter.expand_scorable(fire_and_forget=False). "
                "Either implement wait_many on PromptClient or call with "
                "fire_and_forget=True and consume results separately."
            )

        results_map = await self.prompt_client.wait_many(
            tickets,
            timeout_s=self.cfg.timeout_s,
        )

        return self._build_results_from_payloads(
            scorable_id=scorable_id,
            results_map=results_map,
        )

    # ------------------------------------------------------------------ helpers

    def _build_prompts(
        self,
        *,
        scorable_id: str,
        scorable_text: str,
        goal_text: str,
        k: int,
        templates: NexusPromptTemplates,
    ) -> List[Dict[str, Any]]:
        """
        Construct k prompt dicts for PromptClient.offload_many().
        """
        prompts: List[Dict[str, Any]] = []

        for idx in range(k):
            system_txt = templates.system.format(
                goal=goal_text,
                thought=scorable_text,
                index=idx + 1,
                scorable_id=scorable_id,
            )
            user_txt = templates.user.format(
                goal=goal_text,
                thought=scorable_text,
                index=idx + 1,
                scorable_id=scorable_id,
            )

            prompts.append(
                {
                    # Chat-style prompt; PromptJob will map this as needed.
                    "messages": [
                        {"role": "system", "content": system_txt},
                        {"role": "user", "content": user_txt},
                    ],
                    # Sampling params live inside the prompt payload so the
                    # gateway / router can forward them directly to the model.
                    "params": {
                        "temperature": self.cfg.temperature,
                        "top_p": self.cfg.top_p,
                        "max_tokens": self.cfg.max_tokens,
                        "seed": self.cfg.seed,
                    },
                    # Optional metadata; PromptClient will merge extra fields
                    # into PromptJob.meta where appropriate.
                    "meta": {
                        "kind": "nexus.expand",
                        "scorable_id": scorable_id,
                        "candidate_index": idx,
                    },
                }
            )

        return prompts

    def _build_results_from_payloads(
        self,
        *,
        scorable_id: str,
        results_map: Any,
    ) -> List[NexusPromptResult]:
        """
        Normalize whatever PromptClient.wait_many(...) returns into a list of
        NexusPromptResult objects.

        We assume wait_many returns either:
          - Dict[job_id, payload], or
          - List[Dict] where each dict has a 'job_id' field.
        """
        results: List[NexusPromptResult] = []

        if isinstance(results_map, dict):
            items = list(results_map.items())
        elif isinstance(results_map, list):
            items = []
            for item in results_map:
                if isinstance(item, dict) and "job_id" in item:
                    items.append((item["job_id"], item))
                else:
                    self.log.warning(
                        "NexusPrompter.wait_many got unrecognized item: %r", item
                    )
        else:
            self.log.warning(
                "NexusPrompter.wait_many returned unexpected type: %r",
                type(results_map),
            )
            return results

        for job_id, payload in items:
            text = self._extract_text(payload)
            results.append(
                NexusPromptResult(
                    job_id=str(job_id),
                    scorable_id=scorable_id,
                    text=text,
                    raw=payload if isinstance(payload, dict) else {"_raw": payload},
                )
            )

        return results

    @staticmethod
    def _extract_text(payload: Any) -> Optional[str]:
        """
        Best-effort extraction of the assistant text from a model payload.

        This is intentionally defensive and should be updated as your gateway's
        canonical response schema stabilizes.
        """
        # Bare string
        if isinstance(payload, str):
            return payload

        if not isinstance(payload, dict):
            return None

        # Common simple patterns
        for key in ("text", "output", "completion", "response"):
            val = payload.get(key)
            if isinstance(val, str):
                return val

        # OpenAI-style: choices[0].message.content or text
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            c0 = choices[0]
            if isinstance(c0, dict):
                msg = c0.get("message")
                if isinstance(msg, dict):
                    content = msg.get("content")
                    if isinstance(content, str):
                        return content
                txt = c0.get("text")
                if isinstance(txt, str):
                    return txt

        # Fallback: nothing obvious
        return None
