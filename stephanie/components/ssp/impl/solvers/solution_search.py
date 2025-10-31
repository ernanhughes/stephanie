# stephanie/components/ssp/solution_search.py
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

# ------------------------ Retrieval prompt (line-by-line) ------------------------
PROMPT_SOLUTION_SEARCH_LINES = """
SYSTEM:
You retrieve SHORT evidence snippets that help answer a query about a canonical mechanism (SEED_ANSWER).

SEED_ANSWER:
{{ seed_answer }}

QUERY:
{{ query }}

CONSTRAINTS:
- Provide concise, factual snippets (1–2 sentences each).
- Prefer content that directly supports or explains SEED_ANSWER in relation to QUERY.
- No commentary or extra sections.

OUTPUT FORMAT — WRITE EXACTLY {{ top_k }} LINES:
snippet: <short evidence snippet>
"""

class SolutionSearch:
    """
    LLM-backed retrieval shim for the SSP MVP.

    Usage:
      searcher = SolutionSearch(cfg, memory, container, logger, event_emitter=emitter)
      docs = await searcher.search("why do glaciers accelerate ...", seed_answer, context)

    Notes:
      - Async-first.
      - Emits: solution_search_query, solution_search_results, error (via TreeEventEmitter).
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        memory: Any,
        container: Any,
        logger: Any,
        *,
        event_emitter: Optional[TreeEventEmitter] = None,
        prompt_text: Optional[str] = PROMPT_SOLUTION_SEARCH_LINES,
        prompt_name: Optional[str] = None,   # if set, load from file "<name>.txt"
        retries: int = 1,
        backoff_sec: float = 0.5,
    ):
        self.cfg = cfg or {}
        self.memory = memory
        self.container = container
        self.logger = logger or _logger
        self.events = event_emitter  # optional sink
        self.prompt_loader = PromptLoader(memory=self.memory, logger=self.logger)
        self.prompt_service = container.get("prompt")

        # knobs
        self.k = int(self.cfg.get("k", 3))
        self.max_chars = int(self.cfg.get("max_chars_per_snippet", 400))
        self.retries = max(0, int(retries))
        self.backoff = float(backoff_sec)

        # prompt sources
        self._prompt_text = prompt_text
        self._prompt_name = prompt_name

    # ----------------------- public API -----------------------

    async def search(self, query: str, seed_answer: str, context: Dict[str, Any]) -> List[str]:
        """Return up to k evidence snippets supporting seed_answer for query."""
        t0 = time.time()
        self._emit("search_query", {"query": query, "k": self.k})

        merged_context = {
            **(context or {}),
            "query": query,
            "seed_answer": seed_answer,
            "top_k": int(self.k),
            "now_ts": int(time.time()),
        }

        # Prompt: prefer file if prompt_name provided
        if self._prompt_name:
            prompt = self.prompt_loader.from_file(f"{self._prompt_name}.txt", self.cfg, merged_context)
            psrc = f"file:{self._prompt_name}.txt"
        else:
            prompt = self.prompt_loader.from_text(self._prompt_text or PROMPT_SOLUTION_SEARCH_LINES, merged_context)
            psrc = "inline"
        _logger.debug("SolutionSearch: loaded prompt (%s), k=%d", psrc, self.k)

        # Call model with light retry
        response: str = ""
        attempt = 0
        while True:
            try:
                response = await self.prompt_service.run_prompt(prompt, merged_context)
                break
            except Exception as e:
                attempt += 1
                self.logger.exception("SolutionSearch prompt call failed (attempt %d): %s", attempt, e)
                self._emit("error", {"where": "solution_search.run_prompt", "error": str(e), "attempt": attempt})
                if attempt > self.retries:
                    snippets = self._fallback_snippets(query, seed_answer, self.k)
                    dt = round(time.time() - t0, 3)
                    self._emit("search_results", {"query": query, "k": self.k, "latency_sec": dt, "num": len(snippets)})
                    return snippets
                await asyncio.sleep(self.backoff * attempt)

        _logger.debug("SolutionSearch LLM response (first 160 chars): %s", (response or "").replace("\n", " ")[:160])

        # Parse + postprocess
        snippets = self._parse_snippets(response, self.k)
        snippets = self._postprocess(snippets, self.k) or self._fallback_snippets(query, seed_answer, self.k)

        dt = round(time.time() - t0, 3)
        self._emit("search_results", {"query": query, "k": self.k, "latency_sec": dt, "num": len(snippets)})
        return snippets

    # ----------------------- internals -----------------------

    def _parse_snippets(self, response: str, k: int) -> List[str]:
        """
        Supported formats (in order of preference):
          1) Line-by-line:  lines starting with `snippet: ...`
          2) JSON:          keys 'snippets' | 'docs' | 'evidence' | 'results'
          3) Bullets/lines: split by newline, trim bullets
        """
        if not response:
            return []

        # 1) Explicit 'snippet:' lines (case/space tolerant)
        lines = [ln.strip() for ln in response.splitlines() if ln.strip()]
        snips: List[str] = []
        for ln in lines:
            m = re.match(r'(?i)^\s*(?:-|\d+[.)])?\s*snippet\s*[:=]\s*(.+?)\s*$', ln)
            if m:
                snips.append(m.group(1).strip())
        if snips:
            return snips[:k]

        # 2) JSON (fenced or bare)
        m = re.search(r"```json\s*(\{.*?\})\s*```", response, re.DOTALL | re.IGNORECASE)
        jtxt = m.group(1) if m else response.strip()
        if jtxt.startswith("{") and jtxt.endswith("}"):
            try:
                obj = json.loads(jtxt)
                lst = self._pluck_list(obj)
                if lst:
                    return lst[:k]
            except Exception:
                pass

        # 3) Fallback: plain lines/bullets
        bullets = [b.strip(" -*•\t") for b in lines]
        bullets = [b for b in bullets if b]
        return bullets[:k]

    def _pluck_list(self, obj: Dict[str, Any]) -> Optional[List[str]]:
        for key in ("snippets", "docs", "evidence", "results"):
            v = obj.get(key)
            if isinstance(v, list):
                # filter non-strings conservatively
                return [str(x) for x in v if isinstance(x, (str, int, float))]
        return None

    def _postprocess(self, snippets: List[str], k: int) -> List[str]:
        """Dedup, clip length, and ensure non-empty."""
        seen = set()
        out: List[str] = []
        for s in snippets:
            s = str(s).strip()
            if not s:
                continue
            if len(s) > self.max_chars:
                s = s[: self.max_chars].rstrip()
            if s not in seen:
                seen.add(s)
                out.append(s)
            if len(out) >= k:
                break
        return out

    def _fallback_snippets(self, query: str, seed_answer: str, k: int) -> List[str]:
        base = (
            f"DOC: On '{query}', note that a key mechanism is: {seed_answer}. "
            f"This may interact with other factors, but {seed_answer} remains central."
        )
        return [base + f" [hit:{i}]" for i in range(k)]

    def _emit(self, event: str, payload: Dict[str, Any]) -> None:
        if not self.events:
            return
        try:
            if event == "search_query":
                self.events.on_progress({"phase": "solution_search_query", **payload})
            elif event == "search_results":
                self.events.on_progress({"phase": "solution_search_results", **payload})
            elif event == "error":
                self.events.on_error(payload.get("error", ""), "solution_search", payload)
            else:
                self.events.on_progress({"phase": event, **payload})
        except Exception:
            pass
