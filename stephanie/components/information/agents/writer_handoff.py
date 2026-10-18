"""Hand the generated blog draft to Writer and record the outcome.

SWF01 phase 4. Runs after ``paper_blog_generator`` and submits the artifact it
published in ``context["paper_blog_artifact_path"]`` to Writer's supported
``validation.validate-file`` operation.

This stage is read-only with respect to the draft and never auto-accepts Writer's
decision: it records the verdict, the Writer report identity, the artifact hash
and whether a human must review before the draft advances. It never raises -- a
missing or failing Writer produces an explicit ``unavailable``/``error`` record.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from stephanie.agents.base_agent import BaseAgent
from stephanie.integrations.writer_handoff import (
    WriterDraftHandoff,
    WriterHandoffRequest,
)
from stephanie.integrations.writer_runtime import WriterRuntimeConfig

log = logging.getLogger(__name__)


class WriterHandoffAgent(BaseAgent):
    """Submits the blog artifact to Writer and records the result."""

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg=cfg, memory=memory, container=container, logger=logger)

    def _enabled(self) -> bool:
        value = self.cfg.get("enabled", True)
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if not self._enabled():
            context["writer_handoff"] = {"status": "disabled"}
            return context

        artifact = context.get("paper_blog_artifact_path")
        if not artifact:
            log.error(
                "WriterHandoffAgent: no paper_blog_artifact_path in context; "
                "nothing to hand off (draft was not published by the generator?)"
            )
            context["writer_handoff"] = {
                "status": "error",
                "error": "no draft artifact in context",
            }
            return context

        cfg = WriterRuntimeConfig(
            project_root=self.cfg.get("project_root"),
            writer_executable=self.cfg.get("writer_executable"),
            timeout_seconds=int(self.cfg.get("timeout_seconds", 60)),
        )
        handoff = WriterDraftHandoff(cfg=cfg, record_dir=self.cfg.get("record_dir"))

        meta = context.get("paper_blog_meta") or {}
        record = handoff.submit(
            WriterHandoffRequest(
                artifact_path=str(artifact),
                paper_arxiv_id=meta.get("arxiv_id") or context.get("arxiv_id"),
                research_run_id=str(context.get("pipeline_run_id") or ""),
                profile=str(self.cfg.get("profile", "fiction")),
                correlation_id=(
                    context.get("writer_handoff_correlation_id")
                    or self.cfg.get("correlation_id")
                ),
                provenance={
                    "stage": "writer_handoff",
                    "blog_config_id": context.get("blog_config_id"),
                    "pipeline_run_id": context.get("pipeline_run_id"),
                    "paper_blog_meta": meta,
                },
            )
        )

        context["writer_handoff"] = record.to_dict()
        return context
