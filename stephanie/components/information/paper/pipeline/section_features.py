# stephanie/components/information/paper/pipeline/section_features.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from stephanie.constants import PIPELINE_RUN_ID
from stephanie.components.information.data import PaperSection

log = logging.getLogger(__name__)


@dataclass
class SectionFeatureDispatcherConfig:
    enabled: bool = True
    max_concurrency: int = 8


class SectionFeatureDispatcher:
    """
    Dispatches additional feature extraction jobs for sections.
    This is where you call other agents/tools/pipelines (summaries, triplets, NER, etc).
    In your original PaperPipelineAgent this was: _dispatch_feature_jobs(...)
    """

    def __init__(self, *, cfg: Dict[str, Any], memory: Any, logger: Any):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger

        fcfg = (cfg.get("features", {}) if isinstance(cfg, dict) else {}) or {}
        self.enabled = bool(fcfg.get("enabled", True))
        self.feature_names: List[str] = list(fcfg.get("feature_names", []))
        self.max_concurrency = int(fcfg.get("max_concurrency", 8))

    async def dispatch(self, *, sections: List[PaperSection], context: Dict[str, Any]) -> None:
        if not self.enabled or not self.feature_names:
            return

        # Keep this lightweight: your pipeline likely uses the Supervisor or Bus
        # to publish events consumed by feature workers.
        bus = self.memory.bus
        run_id = context.get(PIPELINE_RUN_ID)
        arxiv_id = context.get("arxiv_id")

        # Publish one event per section per feature (simple + explicit)
        # If you already have a batched worker, you can change to batch events.
        count = 0
        for sec in sections:
            scorable_id = getattr(sec, "scorable_id", None) or getattr(sec, "section_id", None) or getattr(sec, "id", None)
            if not scorable_id:
                continue

            for feat in self.feature_names:
                payload = {
                    "run_id": run_id,
                    "arxiv_id": arxiv_id,
                    "section_id": getattr(sec, "section_id", None) or getattr(sec, "id", None),
                    "scorable_id": str(scorable_id),
                }
                try:
                    await bus.publish("paper.section.feature", payload)                    
                    count += 1
                except Exception:
                    log.warning("SectionFeatureDispatcher: publish failed feature=%s scorable_id=%s", feat, scorable_id, exc_info=True)

        log.info("SectionFeatureDispatcher: dispatched %d feature events", count)
