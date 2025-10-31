# stephanie/components/ssp/actors/verifier.py
from __future__ import annotations

import hashlib
from typing import Any, Dict, List

from stephanie.components.ssp.services.telemetry import SSPTelemetry


def _sid(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


class Verifier:
    """
    SSP Verifier actor (bus-events only).

    Responsibilities:
      - Score/verify candidate solutions and produce verdicts with confidence.
      - Emit structured telemetry to bus/Arena (no PlanTrace writes).

    The default implementation provides a lightweight heuristic confidence so you
    can wire dashboards immediately. Replace `_score()` with HRM/SICQL/EBT or LLM judges.
    """

    def __init__(self, cfg, memory, container, logger):
        self.cfg = cfg or {}
        self.memory = memory
        self.container = container
        self.logger = logger
        self.telemetry = SSPTelemetry(self.cfg.get("telemetry", {}), memory, logger, subject_root="ssp")

        # Heuristic scoring knobs
        self.len_min = int(self.cfg.get("len_min", 16))
        self.len_max = int(self.cfg.get("len_max", 2048))

    async def verify(self, context: Dict[str, Any], solutions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Produce verdicts:
          {
            "id": str,
            "solution_id": str,
            "ok": bool,
            "confidence": float,   # 0..1
            "meta": {...}
          }
        """
        async with self.telemetry.span(context={**context, "actor": "verifier"}, name="actor.verify"):
            verdicts: List[Dict[str, Any]] = []

            if not solutions:
                await self.telemetry.publish(
                    "ssp.actor.verify.result",
                    {"count": 0, "note": "no_solutions"},
                    context={**context, "actor": "verifier"},
                )
                return verdicts

            for s in solutions:
                text = (s.get("text") or "").strip()
                conf = self._score(text)
                ok = conf >= 0.5
                verdicts.append(
                    {
                        "id": _sid(f"{s.get('id','')}::{conf:.4f}"),
                        "solution_id": s.get("id"),
                        "ok": bool(ok),
                        "confidence": float(conf),
                        "meta": {
                            "ssp_actor": "verifier",
                            "goal_id": (context.get("goal") or {}).get("id"),
                            "heuristic": "length_window",
                            "len": len(text),
                            "len_min": self.len_min,
                            "len_max": self.len_max,
                        },
                    }
                )

            await self.telemetry.publish(
                "ssp.actor.verify.result",
                {
                    "count": len(verdicts),
                    "ok_rate": (sum(1 for v in verdicts if v["ok"]) / max(1, len(verdicts))),
                    "ids": [v["id"] for v in verdicts],
                },
                context={**context, "actor": "verifier"},
            )

            return verdicts

    def _score(self, text: str) -> float:
        """
        Tiny heuristic confidence in [0,1] based on length window.
        Replace with HRM/SICQL/EBT/LLM judge in production.
        """
        n = len(text)
        if n <= 0:
            return 0.0
        # Map len_min..len_max → 0.5..1.0; outside → decay toward 0.0
        if n < self.len_min:
            return max(0.0, 0.5 * (n / max(1, self.len_min)))
        if n > self.len_max:
            # linear decay beyond len_max down to ~0.25 (arbitrary cap)
            overflow = n - self.len_max
            return max(0.25, 1.0 - min(0.75, overflow / (self.len_max * 2.0)))
        # inside window → scale to [0.5, 1.0]
        span = max(1, self.len_max - self.len_min)
        return 0.5 + 0.5 * ((n - self.len_min) / span)
