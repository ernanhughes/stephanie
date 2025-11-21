# stephanie/agents/knowledge/scorable_annotate.py
"""
Scorable Annotation Agent (Processor-powered)

This agent enriches incoming scorables using the canonical ScorableProcessor.
It hydrates known features (DB), computes missing ones (domain/NER/embeddings,
vision), optionally attaches model scores, persists deltas, and reflects a thin
annotation back onto each scorable's 'meta' for downstream compatibility.

Key improvements:
- Pluggable, robust feature mediation via ScorableProcessor
- Batch processing with progress bars
- Idempotent persistence (writers only write deltas)
- Optional 'force' mode that recomputes without persisting to avoid dupes
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import List, Optional

from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.scorable_processor import ScorableProcessor

log = logging.getLogger(__name__)


class ScorableAnnotateAgent(BaseAgent):
    """
    Agent that places the ScorableProcessor at the front of the pipeline.
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)

        # Behavior knobs (sane defaults)
        self.progress_enabled: bool = bool(cfg.get("progress", True))
        self.filter_role: bool = bool(cfg.get("filter_role", False))
        self.scorable_role: str = cfg.get("scorable_role", "candidate")


        # Batch + scoring options
        self.batch_size: int = int(cfg.get("batch_size", 64))
        self.attach_scores: bool = bool(cfg.get("attach_scores", True))
        self.scoring_dims: Optional[List[str]] = cfg.get("scoring_dims")

        # progress/concurrency knobs
        self.max_concurrency: int = int(cfg.get("max_concurrency", 8))
        self.progress_log_every: int = int(cfg.get("progress_log_every", 25))
        self.progress_leave: bool = bool(cfg.get("progress_leave", True))
        self.progress_position: int = int(cfg.get("progress_position", 0))

        self.scorable_processor: ScorableProcessor = ScorableProcessor(
            self.cfg.get("processor", {}),
            memory,
            container,
            logger
        )

    # ---------- Public entry point ----------

    async def run(self, context: dict) -> dict:
        """
        Expects context['scorables'] = List[dict|Scorable].
        Produces:
          - context['scorable_features'] = List[dict] (canonical features rows)
          - updates each scorable.meta with short-form 'domains' and 'ner'
          - context['scorable_annotation_summary'] with stats
        """
        scorables = list(context.get(self.input_key) or [])

        rows = await self.scorable_processor.process_many(scorables, context=context)
    
        context[self.output_key] = rows

        return context

