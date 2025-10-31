# stephanie/agents/ssp_mvp/solution_search.py
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from stephanie.components.tree.events import TreeEventEmitter
from stephanie.prompts.prompt_loader import PromptLoader

_logger = logging.getLogger(__name__)


class SolutionSearch:
    """
    LLM-backed retrieval shim for the SSP MVP.

    Usage:
      searcher = SolutionSearch(container, prompt_loader, cfg, seed_answer, prompt_name="solution_search")
      docs = searcher.search("why do glaciers accelerate when ...", k=3)

    Notes:
      - If you're already inside an asyncio loop, call `await asearch(...)` instead of `search(...)`.
      - Emits lightweight events (optional) for observability: search_query, search_results, error.
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        memory,
        container,
        logger,
        *,
        event_emitter: Optional[TreeEventEmitter] = None,
    ):
        self.cfg = cfg
        self.memory = memory
        self.container = container
        self.logger = logger
        self.events = event_emitter  # optional
        self.prompt_loader = PromptLoader(
            memory=self.memory, logger=self.logger
        )
        self.prompt_service = container.get("prompt")
        self.k = int(self.cfg.get("k", 3))
        self.max_chars_per_snippet = int(
            self.cfg.get("max_chars_per_snippet", 1000)
        )

    # ----------------------- public API -----------------------

    async def search(self, query: str, seed_answer: str, context: Dict[str, Any]) -> List[str]:
        """Primary async path."""
        t0 = time.time()
        self._emit("search_query", {"query": query, "k": self.k})

        merged_context = {  
            **context,
            "query": query,
            "seed_answer": seed_answer,
            "top_k": int(self.k),
            "now_ts": int(time.time()),
        }
        prompt = self.prompt_loader.from_text(PROMPT, merged_context)
        _logger.debug(
            "SolutionSearch: loaded prompt '%s.txt' (k=%d)",
            prompt,
            self.k,
        )

        try:
            response: str = await self.prompt_service.run_prompt(
                prompt, merged_context
            )
            _logger.debug(
                "SolutionSearch LLM response (first 120 chars): %s",
                response[:120].replace("\n", " "),
            )
            snippets = self._parse_response(response, self.k)
        except Exception as e:
            _logger.exception("SolutionSearch prompt call failed: %s", e)
            self._emit(
                "error",
                {"where": "solution_search.run_prompt", "error": str(e)},
            )
            snippets = self._fallback_snippets(query, self.k)

        snippets = self._postprocess(snippets, self.k)
        dt = round(time.time() - t0, 3)
        self._emit(
            "search_results",
            {
                "query": query,
                "k": self.k,
                "latency_sec": dt,
                "num": len(snippets),
            },
        )

        return snippets

    # ----------------------- internals -----------------------

    def _parse_response(self, response: str, k: int) -> List[str]:
        """
        Accepts either:
          1) JSON with keys: 'snippets' | 'docs' | 'evidence' | 'results'
          2) Markdown fenced JSON ```json ... ```
          3) Plain text bullet/line list
        Returns top-k strings.
        """
        # Try fenced JSON
        m = re.search(
            r"```json\s*(\{.*?\})\s*```", response, re.DOTALL | re.IGNORECASE
        )
        if m:
            try:
                obj = json.loads(m.group(1))
                lst = self._pluck_list(obj)
                if lst:
                    return lst[:k]
            except Exception:
                pass

        # Try bare JSON
        s = response.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                obj = json.loads(s)
                lst = self._pluck_list(obj)
                if lst:
                    return lst[:k]
            except Exception:
                pass

        # Fallback: split lines or bullets
        bullets = re.split(r"(?:\n|\r|\r\n)+", response)
        bullets = [b.strip(" -*•\t") for b in bullets if b.strip()]
        if bullets:
            return bullets[:k]

        return []

    def _pluck_list(self, obj: Dict[str, Any]) -> Optional[List[str]]:
        for key in ("snippets", "docs", "evidence", "results"):
            v = obj.get(key)
            if isinstance(v, list):
                # filter non-strings conservatively
                return [str(x) for x in v if isinstance(x, (str, int, float))]
        return None

    def _postprocess(self, snippets: List[str], k: int) -> List[str]:
        # Deduplicate, clip length, ensure non-empty
        seen = set()
        out: List[str] = []
        for s in snippets:
            s = str(s).strip()
            if not s:
                continue
            s = s[: self.max_chars_per_snippet]
            if s not in seen:
                seen.add(s)
                out.append(s)
            if len(out) >= k:
                break
        if not out:
            # last-ditch fallback mirrors the old dummy behaviour
            out = self._fallback_snippets("<unknown>", k)
        return out

    def _fallback_snippets(self, query: str, k: int) -> List[str]:
        base = (
            f"DOC: On '{query}', note that a key mechanism is: {self.seed}. "
            f"This may interact with other factors, but {self.seed} remains central."
        )
        return [base + f" [hit:{i}]" for i in range(k)]

    def _emit(self, event: str, payload: Dict[str, Any]) -> None:
        if not self.events:
            return
        try:
            # Use generic event names under the topic assigned to the emitter
            if event == "search_query":
                self.events.on_progress(
                    {"phase": "solution_search_query", **payload}
                )
            elif event == "search_results":
                self.events.on_progress(
                    {"phase": "solution_search_results", **payload}
                )
            elif event == "error":
                self.events.on_error(
                    payload.get("error", ""), "solution_search", payload
                )
            else:
                self.events.on_progress({"phase": event, **payload})
        except Exception:
            pass


PROMPT = """
Good SYSTEM:
You are a strict knowledge judge for a Search Self-Play (SSP) loop. Be concise and precise.

GOAL:
{{ goal_text }}

QUESTION:
{{ question_text }}

GROUND TRUTH (SEED_ANSWER):
{{ seed_answer }}

CANDIDATE ANSWER (to judge):
{{ assistant_text }}

{% if evidence and evidence|length > 0 %}
EVIDENCE SNIPPETS (may be noisy; use to check support):
{% for s in evidence %}
- {{ s }}
{% endfor %}
{% endif %}

JUDGING PRINCIPLES (SSP-specific):
- Treat SEED_ANSWER as the canonical ground truth for this episode.
- Score mainly on **correctness vs SEED_ANSWER** (70%), plus **faithfulness to evidence** (20%), and **goal-relevant clarity/specificity** (10%).
- Reward: directly states the correct mechanism/fact, matches SEED_ANSWER, and is supported (or at least not contradicted) by evidence.
- Penalize: contradictions to SEED_ANSWER, invented claims, vague/fluffy text, or ignoring the question’s focus.
- If the candidate is off-topic or unverifiable here, score low.

RETURN FORMAT — OUTPUT EXACTLY TWO LINES (no extra text, no code fences):
rationale: <1–3 sentences explaining the key reason for the score (match vs SEED_ANSWER, evidence support/contradiction)>
score: <integer 0–100>
"""
