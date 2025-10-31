# stephanie/components/ssp/actors/proposer.py
from __future__ import annotations

import hashlib
import uuid
from typing import Any, Dict, List, Optional

from stephanie.components.ssp.services.telemetry import SSPTelemetry


def _sid(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


class Proposer:
    """
    SSP Proposer actor (bus-events only).

    Responsibilities:
      - Generate candidate proposals for the current goal/context.
      - Emit structured telemetry to bus/Arena (no PlanTrace writes).
    """

    def __init__(self, cfg, memory, container, logger):
        self.cfg = cfg or {}
        self.memory = memory
        self.container = container
        self.logger = logger
        self.telemetry = SSPTelemetry(self.cfg.get("telemetry", {}), memory, logger, subject_root="ssp")
        self.max_proposals: int = int(self.cfg.get("max_proposals", 6))

    async def propose(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Produce a list of proposal dicts:
          { "id": str, "text": str, "meta": {...} }
        """
        async with self.telemetry.span(context={**context, "actor": "proposer"}, name="actor.propose"):
            goal = (context.get("goal") or {})
            goal_text: str = (goal.get("goal_text") or "").strip()

            # Source material for proposals: goal_text + any hints supplied in context.
            hints: List[str] = []
            if isinstance(context.get("hints"), list):
                hints = [str(h) for h in context["hints"] if h]

            seeds: List[str] = []
            if goal_text:
                seeds.append(goal_text)
            seeds.extend(hints)

            # De-duplicate and cap by max_proposals
            seen, deduped = set(), []
            for s in seeds:
                s2 = s.strip()
                if not s2 or s2 in seen:
                    continue
                seen.add(s2)
                deduped.append(s2)
                if len(deduped) >= self.max_proposals:
                    break

            if not deduped and goal_text:
                deduped = [goal_text]

            proposals: List[Dict[str, Any]] = []
            for text in deduped:
                proposals.append(
                    {
                        "id": _sid(text),
                        "text": text,
                        "meta": {
                            "ssp_actor": "proposer",
                            "from": "goal_or_hints",
                            "goal_id": goal.get("id"),
                        },
                    }
                )

            await self.telemetry.publish(
                "ssp.actor.propose.result",
                {"count": len(proposals), "ids": [p["id"] for p in proposals]},
                context={**context, "actor": "proposer"},
            )

            return proposals
