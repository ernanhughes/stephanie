# stephanie/components/ssp/actors/solver.py
from __future__ import annotations

import hashlib
from typing import Any, Dict, List

from stephanie.components.ssp.services.telemetry import SSPTelemetry


def _sid(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


class Solver:
    """
    SSP Solver actor (bus-events only).

    Responsibilities:
      - Turn proposals into candidate solutions (fast heuristic or call out to tools/agents).
      - Emit structured telemetry to bus/Arena (no PlanTrace writes).
    """

    def __init__(self, cfg, memory, container, logger):
        self.cfg = cfg or {}
        self.memory = memory
        self.container = container
        self.logger = logger
        self.telemetry = SSPTelemetry(self.cfg.get("telemetry", {}), memory, logger, subject_root="ssp")

        # Optional knobs (placeholders to wire real tools later)
        self.echo_mode: bool = bool(self.cfg.get("echo_mode", True))  # If true, produce simple echo-style solution

    async def solve(self, context: Dict[str, Any], proposals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Turn proposals into solutions:
          { "id": str, "proposal_id": str, "text": str, "meta": {...} }
        """
        async with self.telemetry.span(context={**context, "actor": "solver"}, name="actor.solve"):
            solutions: List[Dict[str, Any]] = []

            if not proposals:
                await self.telemetry.publish(
                    "ssp.actor.solve.result",
                    {"count": 0, "note": "no_proposals"},
                    context={**context, "actor": "solver"},
                )
                return solutions

            # Simple baseline: echo the proposal text (replace with real logic/tools)
            for p in proposals:
                ptext = (p.get("text") or "").strip()
                solved_text = ptext if self.echo_mode else f"SOLUTION: {ptext}"
                solutions.append(
                    {
                        "id": _sid(f"{p.get('id','')}::{solved_text}"),
                        "proposal_id": p.get("id"),
                        "text": solved_text,
                        "meta": {
                            "ssp_actor": "solver",
                            "strategy": "echo" if self.echo_mode else "prefix",
                            "goal_id": (context.get("goal") or {}).get("id"),
                        },
                    }
                )

            await self.telemetry.publish(
                "ssp.actor.solve.result",
                {"count": len(solutions), "ids": [s["id"] for s in solutions]},
                context={**context, "actor": "solver"},
            )

            return solutions
