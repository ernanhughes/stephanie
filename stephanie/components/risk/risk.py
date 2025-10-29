# stephanie/components/risk/risk.py
from __future__ import annotations

import logging
from typing import Any, Dict

from stephanie.agents.base_agent import BaseAgent
from stephanie.components.risk.orchestrator import RiskOrchestrator

_logger = logging.getLogger(__file__)


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
        scorables = self.get_scorables(context)
        pipeline_run_id = context.get("pipeline_run_id")
        results = []
        for scorable in scorables:
            goal = scorable.get("goal_ref", context.get("goal", {})).get("text", "")
            reply = scorable.get("text", "")

            _logger.debug(
                f"RiskAgent: evaluating run_id={pipeline_run_id} model={self.default_model_alias} monitor={self.default_monitor_alias}"
            )

            rec = await self.orchestrator._evaluate_one(
                run_id=str(pipeline_run_id),
                goal=goal,
                reply=reply,
                model_alias=self.default_model_alias,
                monitor_alias=self.default_monitor_alias,
                context=context,
            )
            results.append(rec)

        context[self.output_key] = results

        return context
