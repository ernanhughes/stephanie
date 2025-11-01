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

# Try to use shared parser if available; otherwise use local fallback.
try:
    # Expected to return: {"rationale": str, "score": float|int, "result": str, "ok": bool, "raw": str}
    from stephanie.components.ssp.utils.parser import parse_three_lines as _parse_three_lines_ext  # type: ignore
except Exception:  # pragma: no cover
    _parse_three_lines_ext = None  # type: ignore


# ------------------------ Retrieval prompts ------------------------

# Primary (simple, reliable): exactly three lines, single snippet in `result:`
PROMPT_EVIDENCE_THREE = """
SYSTEM:
You produce ONE short evidence snippet that helps explain or support the SEED_ANSWER
with respect to the QUERY.

SEED_ANSWER:
{{ seed_answer }}

QUERY:
{{ query }}

CONSTRAINTS:
- Return exactly one short factual snippet (1–2 sentences).
- If unsure, fall back to: "{{ seed_answer }} is the key mechanism."
- No extra text, no markdown, no bullet points.

OUTPUT — EXACTLY THREE LINES:
rationale: <1 sentence on why this snippet is relevant>
score: <0-100 confidence you have in this snippet>
result: <the single snippet>
""".strip()

# Secondary (more permissive) if we ever request k>1: ask for explicit `snippet:` lines
PROMPT_EVIDENCE_LINES = """
SYSTEM:
You return SHORT evidence snippets that help explain or support the SEED_ANSWER
with respect to the QUERY.

SEED_ANSWER:
{{ seed_answer }}

QUERY:
{{ query }}

CONSTRAINTS:
- Provide concise, factual snippets (1–2 sentences each).
- No commentary or extra sections.

OUTPUT — WRITE EXACTLY {{ top_k }} LINES:
snippet: <short evidence snippet>
""".strip()


class SolutionSearch:
    """
    LLM-backed retrieval shim for SSP evidence gathering.

    Public API:
      - search(query, seed_answer, context, top_k=None) -> List[str]
      - find_snippets(query, top_k=None) -> List[str]   (compat shim; seedless)

    Notes:
      - Returns list[str] snippets (possibly length 1 if using the three-line prompt).
      - Emits TreeEventEmitter events (optional).
      - Uses PromptService.run_prompt(prompt_text, context) under the hood.
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        memory: Any,
        container: Any,
        logger: Any,
        *,
        event_emitter: Optional[TreeEventEmitter] = None,
        use_three_line_prompt: bool = True,
        retries: int = 1,
        backoff_sec: float = 0.5,
    ):
        self.cfg = cfg or {}
        self.memory = memory
        self.container = container
        self.logger = logger or _logger
        self.events = event_emitter

        self.prompt_loader = PromptLoader(memory=self.memory, logger=self.logger)
        self.prompt_service = container.get("prompt")

        # knobs
        self.k_default = int(self.cfg.get("k", 1))          # default to 1 per your “one result” rule
        self.max_chars = int(self.cfg.get("max_chars_per_snippet", 400))
        self.retries = max(0, int(retries))
        self.backoff = float(backoff_sec)
        self.use_three = bool(use_three_line_prompt)

    # ----------------------- public API -----------------------

    async def search(
        self,
        query: str,
        seed_answer: str,
        context: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
    ) -> List[str]:
        """
        Return up to k evidence snippets supporting `seed_answer` for `query`.
        If k==1 (default), we use the three-line prompt and return [result].
        For k>1, we ask for N 'snippet:' lines and parse accordingly.
        """
        k = int(top_k or self.k_default)
        t0 = time.time()
        self._emit("search_query", {"query": query, "k": k})

        merged = {
            **(context or {}),
            "query": query,
            "seed_answer": seed_answer,
            "top_k": k,
            "now_ts": int(time.time()),
        }

        # Choose prompt
        prompt_text = PROMPT_EVIDENCE_THREE if (self.use_three and k == 1) else PROMPT_EVIDENCE_LINES
        prompt = self.prompt_loader.from_text(prompt_text, merged)

        # Call model with light retry
        response: str = ""
        attempt = 0
        while True:
            try:
                response = await self.prompt_service.run_prompt(prompt_text=prompt, context=merged)
                break
            except Exception as e:
                attempt += 1
                _logger.warning("SolutionSearch prompt call failed (attempt %d): %s", attempt, e)
                self._emit("error", {"where": "solution_search.run_prompt", "error": str(e), "attempt": attempt})
                if attempt > self.retries:
                    snippets = self._fallback_snippets(query, seed_answer, k)
                    self._emit_results(query, k, t0, len(snippets))
                    return snippets
                await asyncio.sleep(self.backoff * attempt)

        _logger.info("SolutionSearch LLM response (first 160 chars): %s", (response or "").replace("\n", " ")[:160])

        # Parse
        if self.use_three and k == 1:
            # Expect exactly three lines with `result:` holding the snippet
            snippet = self._parse_three_result(response, seed_answer)
            snippets = self._postprocess([snippet], 1) if snippet else []
        else:
            # Expect k lines `snippet: ...`, with robust fallbacks
            snippets = self._parse_snippets(response, k)
            snippets = self._postprocess(snippets, k)

        # Final fallback
        if not snippets:
            snippets = self._fallback_snippets(query, seed_answer, k)

        self._emit_results(query, k, t0, len(snippets))
        return snippets

    async def find_snippets(self, query: str, top_k: Optional[int] = None) -> List[str]:
        """
        Compatibility shim (seedless). Uses the same engine but without seed conditioning.
        """
        seed = ""  # No seed answer
        return await self.search(query=query, seed_answer=seed, context={}, top_k=top_k or self.k_default)

    # ----------------------- internals -----------------------

    def _parse_three_result(self, text: str, seed_answer: str) -> str:
        """Parse our strict three-line format into a single snippet string."""
        parsed = self._call_parse_three(text, default_score=35.0)
        if parsed.get("ok") and parsed.get("result", "").strip():
            return parsed["result"].strip()

        # Hard fallback: never return empty
        seed = (seed_answer or "").strip()
        return f"{seed} is the key mechanism." if seed else "The mechanism is central to the process."

    def _call_parse_three(self, text: str, default_score: float = 35.0) -> Dict[str, Any]:
        """Use shared parser if present; otherwise local minimal parser."""
        if _parse_three_lines_ext is not None:
            try:
                return _parse_three_lines_ext(text, default_score=default_score)  # type: ignore
            except Exception:
                pass
        return self._parse_three_lines_local(text, default_score=default_score)

    @staticmethod
    def _parse_three_lines_local(text: str, default_score: float = 35.0) -> Dict[str, Any]:
        """
        Minimal local parser for the strict format:
          rationale: <...>
          score: <int 0-100>
          result: <...>
        """
        raw = (text or "").strip()
        out = {"rationale": "", "score": float(default_score), "result": "", "ok": False, "raw": raw}
        if not raw:
            return out

        rat, sco, res = None, None, None
        for ln in raw.splitlines():
            s = ln.strip()
            if not s:
                continue
            low = s.lower()
            if low.startswith("rationale:"):
                rat = s[len("rationale:") :].strip().strip(" \"'“”‘’")
            elif low.startswith("score:"):
                val = s[len("score:") :].strip()
                m = re.search(r"(-?\d+(\.\d+)?)", val)
                sco = float(m.group(1)) if m else default_score
            elif low.startswith("result:"):
                res = s[len("result:") :].strip().strip(" \"'“”‘’")

        out["rationale"] = rat or ""
        out["score"] = float(sco if sco is not None else default_score)
        out["result"] = res or ""
        out["ok"] = bool(out["result"])
        return out

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

    @staticmethod
    def _pluck_list(obj: Dict[str, Any]) -> Optional[List[str]]:
        for key in ("snippets", "docs", "evidence", "results"):
            v = obj.get(key)
            if isinstance(v, list):
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
        """Conservative, non-empty fallback."""
        base = (
            f"DOC: For '{query}', a central mechanism is: {seed_answer}. "
            f"This snippet highlights why {seed_answer} is relevant."
        )
        return [base + f" [hit:{i}]" for i in range(k)]

    # ----------------------- events -----------------------

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

    def _emit_results(self, query: str, k: int, t0: float, num: int) -> None:
        dt = round(time.time() - t0, 3)
        self._emit("search_results", {"query": query, "k": k, "latency_sec": dt, "num": num})
