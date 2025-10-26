# stephanie/components/ssp/proposer.py
from __future__ import annotations

import json
import time
from dataclasses import asdict
from typing import Optional, Dict, Any
from stephanie.components.ssp.types import Proposal
from stephanie.components.ssp.util import get_model_safe, get_trace_logger, PlanTrace_safe

JSON_INSTR = """Return ONLY valid JSON with keys:
{"query": "...", "verification_approach": "...", "difficulty": 0.0, "connections": ["..."]}
No commentary, no markdown, no code fences.
"""

class Proposer:
    def __init__(self, cfg):
        self.cfg = cfg.self_play
        self.model = get_model_safe("proposer")
        self.trace_logger = get_trace_logger()
        self.difficulty = self.cfg.qmax.initial_difficulty

    def _json_or_retry(self, prompt: str, retries: int = 2) -> dict:
        for _ in range(retries + 1):
            resp = self.model(prompt + "\n\n" + JSON_INSTR)
            try:
                data = json.loads(resp)
                if not all(k in data for k in ("query","verification_approach","difficulty","connections")):
                    raise ValueError("missing keys")
                if not isinstance(data["connections"], list):
                    data["connections"] = []
                return data
            except Exception:
                prompt = "Your previous output was invalid JSON. " + JSON_INSTR
        return {
            "query": "What measurable improvement does VPM evolver give to HRM?",
            "verification_approach": "A/B compare HRM pre/post on matched cases; bootstrap CI.",
            "difficulty": float(self.difficulty),
            "connections": ["VPM","HRM"]
        }

    def build_prompt(self, context: Optional[Dict[str,Any]]) -> str:
        return f"""Generate a novel, verifiable research question at difficulty={self.difficulty:.2f}.
Context: {context or 'None'}
"""

    def generate(self, context: Optional[Dict[str,Any]] = None) -> Dict[str,Any]:
        prompt = self.build_prompt(context)
        data = self._json_or_retry(prompt)
        prop = Proposal(
            query=data["query"],
            verification_approach=data["verification_approach"],
            difficulty=float(data.get("difficulty", self.difficulty)),
            connections=data.get("connections", []),
            raw_response=json.dumps(data)
        )
        tr = PlanTrace_safe(
            trace_id=f"proposer-{int(time.time()*1000)%1000000}",
            role="proposer", goal=prop.query, status="proposed",
            metadata={"difficulty": prop.difficulty, "connections": prop.connections},
            input=prompt, output=prop.raw_response, artifacts={}
        )
        self.trace_logger.log(tr)
        return asdict(prop)

    def update_difficulty(self, success_rate: float):
        target = 0.7
        k_p = 0.5 * self.cfg.qmax.difficulty_step
        delta = k_p * (target - success_rate)
        self.difficulty = float(max(self.cfg.qmax.initial_difficulty,
                                    min(self.cfg.qmax.max_difficulty,
                                        self.difficulty + delta)))
