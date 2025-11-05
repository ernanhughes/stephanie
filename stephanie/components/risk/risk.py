# stephanie/components/risk/risk.py
from __future__ import annotations

import logging
from typing import Any, Dict

from stephanie.agents.base_agent import BaseAgent
from stephanie.components.risk.orchestrator import RiskOrchestrator

log = logging.getLogger(__file__)


class RiskAgent(BaseAgent):
    """
    Risk Agent — entry point for per-reply risk assessment and badge generation.

    Responsibilities:
      - Read (goal, reply) from context
      - Delegate to RiskOrchestrator.evaluate(...)
      - Attach result under context['risk']

    Returns (example):
      {
        "risk": {
          "run_id": "...",
          "model_alias": "chat",
          "monitor_alias": "tiny",
          "metrics": {...},
          "decision": "OK" | "WATCH" | "RISK",
          "thresholds": {...},
          "reasons": {...},
          "badge_svg": "data:image/svg+xml;base64,..."
        }
      }
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        # Keep output key explicit so callers know where to find results
        self.orchestrator = RiskOrchestrator(cfg, memory, container, logger)
        self.policy_profile = cfg.get("policy_profile", "chat.standard")
        self.policy_overrides = cfg.get("policy_overrides") 
        self.default_model_alias = cfg.get("default_model_alias", "risk")
        self.default_monitor_alias = cfg.get("default_monitor_alias", "tiny")

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        
        context = await self.orchestrator.execute_assessment(context)

        return context
