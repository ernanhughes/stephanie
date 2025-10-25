# stephanie/components/gap/risk/orchestrator.py
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml  # PyYAML

# --- Required sibling services (keep these names stable across files) -------
# Implemented in:
#   stephanie/components/gap/risk/monitor.py         -> class MonitorService
#   stephanie/components/gap/risk/aligner.py         -> class Aligner
#   stephanie/components/gap/risk/risk_engine.py     -> class RiskEngine
#   stephanie/components/gap/risk/badge_renderer.py  -> class BadgeRenderer
#   stephanie/components/gap/risk/provenance.py      -> class ProvenanceLogger
from .monitor import MonitorService
from .aligner import Aligner
from .risk_engine import RiskEngine
from .badge_renderer import BadgeRenderer
from .provenance import ProvenanceLogger


def _utc_now_run_id() -> str:
    # example: 2025-10-23T20:58:41Z/35841
    now = datetime.now(timezone.utc)
    return f"{now.strftime('%Y-%m-%dT%H:%M:%SZ')}/{now.strftime('%S%f')[-5:]}"


class GapRiskOrchestrator:
    """
    Glue service:
      goal, reply -> monitor(scores) -> aligner(normalize,+Δ)
                   -> risk(decide,hysteresis) -> badge(SVG)
                   -> provenance(log) -> JSON record for UI & storage.

    Policy profiles live in ./profiles/*.yaml, e.g.:
      chat.standard.yaml, research.wide.yaml, rag.strict.yaml

    Returns a record shaped like:

    {
      "run_id": "...",
      "model_alias": "chat-hrm",
      "monitor_alias": "tiny-monitor",
      "metrics": {
          "confidence01": 0.81,
          "faithfulness_risk01": 0.22,
          "ood_hat01": 0.10,
          "delta_gap01": 0.17
      },
      "decision": "OK",  # OK | WATCH | RISK
      "thresholds": { "faithfulness": 0.35, "uncertainty": 0.40, "ood": 0.30, "delta": 0.30 },
      "reasons": { "risk_faith": 0.22, "risk_ood": 0.10, "risk_delta": 0.17, "risk_unc": 0.19 },
      "badge_svg": "data:image/svg+xml;base64,..."
    }
    """

    def __init__(
        self,
        container: Any,
        *,
        policy_profile: str = "chat.standard",
        policy_overrides: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.container = container
        self.logger = logger or logging.getLogger(__name__)

        self.policy = self._load_policy(policy_profile)
        if policy_overrides:
            # shallow merge for convenience
            self.policy.update(policy_overrides)

        # Extract knobs with safe defaults
        thresholds = self.policy.get("thresholds", {})
        hysteresis = float(self.policy.get("hysteresis", 0.05))
        badge_size = int(self.policy.get("badge_size", 256))
        out_dir = str(self.policy.get("out_dir", "./runs/hallucinations"))
        calibration = self.policy.get("calibration_params", {}) or {}

        # Wire services
        self.monitor = MonitorService(container=container, logger=self.logger)
        self.aligner = Aligner(calibration_params=calibration, logger=self.logger)
        self.risk = RiskEngine(thresholds=thresholds, hysteresis=hysteresis, logger=self.logger)
        self.badge = BadgeRenderer(size=badge_size)
        self.prov = ProvenanceLogger(out_dir=out_dir, logger=self.logger)

        # Optional: event bus publisher hook if present on container
        self._publisher = getattr(container, "event_publisher", None)
        self._publish_topic = self.policy.get("publish_topic")  # e.g., "sis.hallucination.badge"

        self.logger.info(
            "GapRiskOrchestrator ready | profile=%s thresholds=%s hysteresis=%.3f out_dir=%s",
            policy_profile, json.dumps(thresholds), hysteresis, out_dir
        )

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    async def evaluate(
        self,
        goal: str,
        reply: str,
        *,
        model_alias: str = "chat",
        monitor_alias: str = "tiny",
        sparkline: Optional[list] = None,
        run_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,  # future: rag evidence, etc.
    ) -> Dict[str, Any]:
        """
        Main entrypoint: compute risk & badge for a finalized reply.
        Safe to call inside your chat pipeline after the model returns.
        """
        rid = run_id or _utc_now_run_id()

        # 1) Score via Tiny/Monitor (teacher-forced)
        raw_metrics = await self.monitor.score(
            goal=goal,
            reply=reply,
            model_alias=model_alias,
            monitor_alias=monitor_alias,
            context=context,
        )

        # 2) Normalize & map to SCM-friendly metrics (+ optional Δ features)
        metrics01 = self.aligner.normalize(raw_metrics, context=context)

        # 3) Decide OK/WATCH/RISK (includes hysteresis)
        decision, reasons = self.risk.decide(metrics01)

        # 4) Render 256x256 badge (data: URI)
        badge_uri = self.badge.render_data_uri(
            metrics01=metrics01,
            decision=decision,
            thresholds=self.risk.thresholds_dict(),
            sparkline=sparkline,
        )

        # 5) Assemble record
        record = {
            "run_id": rid,
            "model_alias": model_alias,
            "monitor_alias": monitor_alias,
            "metrics": metrics01,
            "decision": decision,
            "thresholds": self.risk.thresholds_dict(),
            "reasons": reasons,  # human/debug readable contributions
            "badge_svg": badge_uri,
        }

        # 6) Provenance (persist JSON + any attachments)
        try:
            self.prov.log(record=record, goal=goal, reply=reply, context=context)
        except Exception as e:
            self.logger.exception("Provenance logging failed: %s", e)

        # 7) Optional event publish (UI overlay)
        if self._publisher and self._publish_topic:
            try:
                self._publisher.publish(self._publish_topic, record)
            except Exception as e:
                self.logger.warning("Badge publish failed (topic=%s): %s", self._publish_topic, e)

        return record

    # --------------------------------------------------------------------- #
    # Internals
    # --------------------------------------------------------------------- #
    def _load_policy(self, profile: str) -> Dict[str, Any]:
        """
        Loads ./profiles/{profile}.yaml relative to this file.
        Provides safe defaults if not found.
        """
        base = Path(__file__).resolve().parent
        candidates = [
            base / "profiles" / f"{profile}.yaml",
            base / "profiles" / f"{profile}.yml",
        ]

        # Env override for custom policy dir
        policy_dir = os.getenv("GAP_RISK_PROFILE_DIR")
        if policy_dir:
            candidates.insert(0, Path(policy_dir) / f"{profile}.yaml")
            candidates.insert(1, Path(policy_dir) / f"{profile}.yml")

        for path in candidates:
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}

        # Safe defaults if no profile present
        return {
            "badge_size": 256,
            "hysteresis": 0.05,
            "out_dir": "./runs/hallucinations",
            "thresholds": {
                "faithfulness": 0.35,
                "uncertainty": 0.40,
                "ood": 0.30,
                "delta": 0.30,
            },
            "calibration_params": {},
            # Optional publish topic (UI bus)
            # "publish_topic": "sis.hallucination.badge",
        }


__all__ = ["GapRiskOrchestrator"]
