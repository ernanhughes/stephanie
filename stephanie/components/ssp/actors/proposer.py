from __future__ import annotations

import json
import re
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig

from stephanie.components.ssp.types import Proposal
from stephanie.components.ssp.util import PlanTrace_safe, get_trace_logger
from stephanie.services.service_container import ServiceContainer

# --------- Parsing helpers (YAML-ish, no external deps) ---------

_CODE_FENCE = re.compile(r"^```[^\n]*\n|\n```$", re.MULTILINE)

# Matches:
# query: ...
# verification: ...
# difficulty: 0.42
# connections:
# - tag one
# - tag two
_PLAIN_KV = re.compile(
    r"(?is)\bquery\s*:\s*(?P<query>.+?)\n+"
    r"\b(?:verification|verification_approach)\s*:\s*(?P<verif>.+?)\n+"
    r"\bdifficulty\s*:\s*(?P<diff>[-+]?\d*\.?\d+)\s*\n+"
    r"\bconnections\s*:\s*(?P<rest>.*)$"
)

# When connections are inline:
# connections: tag1, tag2, tag3
_INLINE_CONN = re.compile(r"(?i)\bconnections\s*:\s*(.+)")

# Bullet list items after "connections:" block
_BULLETS = re.compile(r"(?m)^\s*[-*+]\s*(.+?)\s*$")


def _strip_fences(text: str) -> str:
    if not text:
        return ""
    return _CODE_FENCE.sub("", text).strip()


def _parse_connections(block: str) -> List[str]:
    """
    Accept:
      - bullet list under a 'connections:' block
      - or a single line 'connections: a, b, c'
    """
    if not block:
        return []
    # First try bullet lines within the block
    bullets = _BULLETS.findall(block)
    if bullets:
        return [b.strip() for b in bullets if b.strip()]

    # Else try inline CSV after "connections:"
    m = _INLINE_CONN.search(block)
    if m:
        items = [x.strip() for x in m.group(1).split(",")]
        return [x for x in items if x]

    # Fallback: split on newlines/commas
    raw = [x.strip() for x in re.split(r"[,;\n]+", block)]
    return [x for x in raw if x]


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def parse_proposal_text(raw: str) -> dict:
    """
    Robustly parse the LLM response in this plain-text schema:

    query: <one line>
    verification: <1-3 sentences>
    difficulty: <0.0..1.0>
    connections:
    - tag1
    - tag2
    - tag3

    Also accepts:
      connections: tag1, tag2, tag3

    Raises ValueError on failure.
    """
    text = _strip_fences(raw)
    m = _PLAIN_KV.search(text)
    if not m:
        # Very last-ditch: try to salvage JSON if the model ignored instructions
        j = _salvage_json(raw)
        if j:
            return j
        raise ValueError("Could not parse proposer output")

    query = (m.group("query") or "").strip()
    verif = (m.group("verif") or "").strip()
    diff = float(m.group("diff"))
    rest = m.group("rest") or ""
    conns = _parse_connections(rest)

    return {
        "query": query,
        "verification_approach": verif,
        "difficulty": diff,
        "connections": conns,
    }


# Minimal JSON salvage if the model returns JSON anyway
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)
def _salvage_json(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except Exception:
        m = _JSON_BLOCK.search(text or "")
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None


# --------- Proposer actor ---------

FORMAT_INSTR = (
    "Return ONLY in this exact format (no extra text, no markdown):\n"
    "query: <one line question>\n"
    "verification: <how to verify—1-3 sentences>\n"
    "difficulty: <0.0..1.0>\n"
    "connections:\n"
    "- <tag1>\n"
    "- <tag2>\n"
)

class Proposer:
    """
    Generates novel, verifiable queries at the current curriculum difficulty.
    Uses PromptService via the ServiceContainer; avoids JSON in the LLM reply.
    """

    def __init__(self, cfg: DictConfig | dict, container: ServiceContainer):
        root = cfg
        self.root: DictConfig = root
        self.sp: DictConfig = root.self_play
        self.cfg: DictConfig = self.sp.proposer
        self.container = container

        self.prompt_service = self.container.get("prompt")  # PromptService
        self.trace_logger = get_trace_logger()

        # curriculum-driven difficulty (bounded by cfg)
        self.difficulty = float(self.sp.qmax.initial_difficulty)
        self._d_lo = float(self.sp.qmax.initial_difficulty)
        self._d_hi = float(self.sp.qmax.max_difficulty)

        # Optional: name/path for a template (if you later add a file)
        self.template_name = getattr(self.cfg, "template_name", None)


    # --- lifecycle hooks --------------------------------------------------

    def set_difficulty(self, value: float) -> None:
        self.difficulty = _clamp(float(value), self._d_lo, self._d_hi)

    def update_difficulty(self, success_rate: float) -> None:
        target = 0.7
        k_p = 0.5 * float(self.sp.qmax.difficulty_step)
        delta = k_p * (target - float(success_rate))
        self.set_difficulty(self.difficulty + delta)

    # --- prompting --------------------------------------------------------

    def _default_prompt(self, context: Optional[Dict[str, Any]]) -> str:
        # Short and strict → higher parse rate
        ctx = context or {}
        ctx_str = json.dumps(ctx, ensure_ascii=False)
        return (
            "You are a research proposal generator. Produce a novel, verifiable question "
            "appropriate for the requested difficulty and context.\n\n"
            f"Difficulty (0..1): {self.difficulty:.2f}\n"
            f"Context JSON: {ctx_str}\n\n"
            f"{FORMAT_INSTR}"
        )

    # --- public API -------------------------------------------------------

    def generate(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Asks the LLM for a proposal (plain text schema), parses it, clamps difficulty,
        returns a Proposal dict your trainer can consume.
        """
        ctx = context or {}
        prompt = self._default_prompt(ctx)

        # Give the model one shot; if parsing fails, tighten and retry once
        attempts = 0
        parsed = None
        last_raw = ""
        while attempts < 2 and parsed is None:
            last_raw = self.prompt_service.call_llm(prompt)
            try:
                data = parse_proposal_text(last_raw)
                parsed = data
            except Exception:
                # Tighten follow-up instruction
                prompt = FORMAT_INSTR
                attempts += 1

        if parsed is None:
            # Guaranteed fallback
            parsed = {
                "query": "What measurable improvement does the VPM evolver give to HRM on matched cases?",
                "verification_approach": "A/B compare HRM pre/post on aligned cases; bootstrap confidence intervals.",
                "difficulty": float(self.difficulty),
                "connections": ["VPM", "HRM"],
            }

        # Normalize + clamp difficulty, ensure list for connections
        try:
            diff = float(parsed.get("difficulty", self.difficulty))
        except Exception:
            diff = self.difficulty
        diff = _clamp(diff, self._d_lo, self._d_hi)

        conns = parsed.get("connections") or []
        if not isinstance(conns, list):
            conns = [str(conns)]

        prop = Proposal(
            query=parsed.get("query", "").strip(),
            verification_approach=parsed.get("verification_approach", "").strip(),
            difficulty=float(diff),
            connections=[str(c).strip() for c in conns if str(c).strip()],
            raw_response=_strip_fences(last_raw) if last_raw else json.dumps(parsed, ensure_ascii=False),
            metadata={"source": "proposer", "ctx": ctx},
        )

        # Trace
        self.trace_logger.log(PlanTrace_safe(
            trace_id=f"proposer-{int(time.time()*1000) % 1_000_000}",
            role="proposer",
            goal=prop.query,
            status="proposed",
            metadata={"difficulty": prop.difficulty, "connections": prop.connections},
            input=prompt,
            output=prop.raw_response,
            artifacts={},
        ))

        return asdict(prop)
