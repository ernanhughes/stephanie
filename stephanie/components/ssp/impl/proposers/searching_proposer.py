# stephanie/components/ssp/impl/proposers/searching_proposer.py
from __future__ import annotations
import asyncio
import time
import logging
from typing import Any, Dict, Optional, Tuple
import re
from stephanie.components.ssp.core.roles.proposer import Proposer
from stephanie.prompts.prompt_loader import PromptLoader
from stephanie.components.tree.events import TreeEventEmitter  # optional sink

_logger = logging.getLogger(__name__)

# -------------------- parser (your original, kept) --------------------

_LINE_RE = re.compile(r'^\s*"?(?P<key>[A-Za-z_]+)"?\s*[:=]\s*(?P<val>.+?)\s*,?\s*$')

def _clean_text(val: str) -> str:
    s = val.strip()
    if (len(s) >= 2) and (
        (s[0] == s[-1] == '"')
        or (s[0] == s[-1] == "'")
        or (s[0] in "“”" and s[-1] in "“”")
        or (s[0] in "‘’" and s[-1] in "‘’")
    ):
        s = s[1:-1].strip()
    return s

def _int_in(val: str) -> Optional[int]:
    m = re.search(r"(-?\d+)", val)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None

def parse_proposer_lines(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    out: Dict[str, Any] = {
        "rationale": "",
        "difficulty": 0,
        "verifiability": 0,
        "question": "",
        "raw": raw,
        "ok": False,
    }
    if not raw:
        return out

    lines = [ln for ln in raw.splitlines() if ln.strip()]
    for ln in lines:
        m = _LINE_RE.match(ln)
        if not m:
            continue
        key = m.group("key").strip().lower()
        val = m.group("val").strip()

        if key == "rationale" and not out["rationale"]:
            out["rationale"] = _clean_text(val)
        elif key == "difficulty" and out["difficulty"] == 0:
            v = _int_in(val)
            if v is not None:
                out["difficulty"] = max(0, min(100, v))
        elif key == "verifiability" and out["verifiability"] == 0:
            v = _int_in(val)
            if v is not None:
                out["verifiability"] = max(0, min(100, v))
        elif key == "question" and not out["question"]:
            out["question"] = _clean_text(val)

    out["ok"] = bool(out["question"])
    return out

# -------------------- prompt (your current inline) --------------------
PROPOSER_PROMPT = """
SYSTEM:
You are building an SSP dataset. Given the canonical mechanism (SEED_ANSWER), write ONE precise, verifiable question whose correct answer is that mechanism.

SEED_ANSWER:
{{ answer }}

CONSTRAINTS:
- Ask for the mechanism directly (no trivia, no multi-part).
- Be specific and test factual understanding.
- No explanations or extra lines.

OUTPUT FORMAT — WRITE EXACTLY FOUR LINES, IN THIS ORDER, NO CODE FENCES:
rationale: <1–2 sentences on why this question targets the mechanism>
difficulty: <integer 0-100>
verifiability: <integer 0-100>
question: <the single best question>
"""

# -------------------- improved Proposer --------------------
class SearchingProposer:
    """
    SSP Proposer (async).
    - Loads a small line-by-line prompt (inline by default, or from file if you want).
    - Calls PromptService, parses robustly, emits telemetry.
    - Returns: (question, meta) where meta={rationale, difficulty, verifiability, raw_ok}.
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        memory: Any,
        container: Any,
        logger: Optional[logging.Logger] = None,
        *,
        event_emitter: Optional[TreeEventEmitter] = None,
        prompt_text: Optional[str] = PROPOSER_PROMPT,            # or None to force file
        prompt_name: Optional[str] = None,              # e.g. "ssp_proposer_lines"
        retries: int = 1,                               # light retry for transient errors
        backoff_sec: float = 0.5,
        question_max_chars: int = 300,
    ):
        self.cfg = cfg or {}
        self.memory = memory
        self.container = container
        self.logger = logger

        self.prompt_loader = PromptLoader(memory=self.memory, logger=self.logger)
        self.prompt_service = container.get("prompt")

        self.events = event_emitter or TreeEventEmitter(topic="ssp.proposer")

        # prompt source
        self._prompt_text = prompt_text
        self._prompt_name = prompt_name  # if set, load from file

        # runtime knobs
        self._retries = max(0, int(retries))
        self._backoff = float(backoff_sec)
        self._qmax = int(question_max_chars)

    async def propose(
        self, seed_answer: str, context: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        t0 = time.time()
        self._emit("start", {"seed_answer": seed_answer})

        merged_context = {
            **(context or {}),
            "answer": seed_answer,
            "now_ts": int(time.time()),
        }

        # Load prompt (prefer file if prompt_name provided)
        if self._prompt_name:
            prompt = self.prompt_loader.from_file(f"{self._prompt_name}.txt", self.cfg, merged_context)
            psrc = f"file:{self._prompt_name}.txt"
        else:
            prompt = self.prompt_loader.from_text(self._prompt_text or PROPOSER_PROMPT, merged_context)
            psrc = "inline"

        _logger.debug("Proposer: loaded prompt (%s)", psrc)

        # Call model with light retry
        response: str = ""
        attempt = 0
        while True:
            try:
                response = await self.prompt_service.run_prompt(prompt, merged_context)
                break
            except Exception as e:
                attempt += 1
                _logger.exception("Proposer prompt call failed (attempt %d): %s", attempt, e)
                self._emit("error", {"where": "proposer.run_prompt", "error": str(e), "attempt": attempt})
                if attempt > self._retries:
                    # Hard fail: return empty but consistent shape
                    dt = round(time.time() - t0, 3)
                    self._emit("results", {
                        "seed_answer": seed_answer, "ok": False, "latency_sec": dt, "response_len": 0
                    })
                    return "", {"rationale": "", "difficulty": 0, "verifiability": 0, "raw_ok": False}
                await asyncio.sleep(self._backoff * attempt)

        _logger.debug("Proposer LLM response (first 160 chars): %s", (response or "").replace("\n", " ")[:160])

        # Parse & normalize
        parsed = parse_proposer_lines(response)
        question = self._normalize_question(parsed.get("question", ""))

        meta = {
            "rationale": parsed.get("rationale", ""),
            "difficulty": int(parsed.get("difficulty", 0) or 0),
            "verifiability": int(parsed.get("verifiability", 0) or 0),
            "raw_ok": bool(parsed.get("ok", False)),
        }

        dt = round(time.time() - t0, 3)
        self._emit("results", {
            "seed_answer": seed_answer,
            "ok": bool(question),
            "latency_sec": dt,
            "response_len": len(response or ""),
            "question_len": len(question),
        })

        return question, meta

    # -------------------- helpers --------------------
    def _normalize_question(self, q: str) -> str:
        """Trim, collapse whitespace, strip quotes, enforce trailing '?', length cap."""
        q = (q or "").strip().strip('"\''"“”‘’").strip()
        q = re.sub(r"\s+", " ", q)
        if q and not q.endswith("?"):
            q += "?"
        if len(q) > self._qmax:
            q = q[: self._qmax].rstrip()
            if not q.endswith("?"):
                q += "?"
        return q

    def _emit(self, event: str, payload: Dict[str, Any]) -> None:
        """Non-fatal telemetry: emits via TreeEventEmitter if available; else logs."""
        try:
            if event == "error":
                self.events.on_error(payload.get("error", ""), "proposer", payload)
            elif event == "start":
                self.events.on_progress({"phase": "proposer_start", **payload})
            elif event == "results":
                self.events.on_progress({"phase": "proposer_results", **payload})
            else:
                self.events.on_progress({"phase": f"proposer_{event}", **payload})
        except Exception:
            # never crash on telemetry
            pass
