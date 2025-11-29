from __future__ import annotations

import logging
from typing import Any, Dict

from stephanie.agents.base_agent import BaseAgent
from stephanie.components.vibe.agents.research_draft import ResearchDraftAgent
from stephanie.components.vibe.agents.research_refine import \
    ResearchRefineAgent
from stephanie.components.vibe.agents.writing_scorer import WritingScorerAgent

log = logging.getLogger(__name__)


class ResearchResponsePipelineAgent(BaseAgent):
    """
    End-to-end research response generator:

      1) Generate initial draft (ResearchDraftAgent)
      2) Score writing quality (WritingScorerAgent)
      3) Optionally refine in a loop (ResearchRefineAgent + scorer)

    Inputs in context:
      - research_question: str  (required)
      - research_notes: str     (optional; sources, scratch, etc.)
      - style_hint: str         (optional; default is Stephanie style)

    Outputs in context:
      - research_answer: str            # final chosen answer
      - research_answer_score: dict     # WritingScore dict
      - research_revisions: list        # history of revisions + scores
    """

    def __init__(self, cfg: Dict[str, Any], memory: Any, container: Any, logger: Any):
        super().__init__(cfg, memory, container, logger)

        self.cfg = cfg or {}
        self.max_iters: int = self.cfg.get("max_iters", 3)
        self.target_overall: float = self.cfg.get("target_overall", 85.0)

        # Sub-agents share the same memory/container/logger
        self.draft_agent = ResearchDraftAgent(cfg.get("draft", {}), memory, container, logger)
        self.scorer_agent = WritingScorerAgent(cfg.get("scorer", {}), memory, container, logger)
        self.refine_agent = ResearchRefineAgent(cfg.get("refine", {}), memory, container, logger)

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if not context.get("research_question"):
            raise ValueError("ResearchResponsePipelineAgent requires 'research_question' in context.")

        # 1) Initial draft
        context = await self.draft_agent.run(context)

        best_draft: str = context.get("research_draft", "")
        best_score: Dict[str, Any] = {}
        revisions = []

        # 2) Refinement loop
        for step in range(self.max_iters):
            # 2a) Score current draft
            #   WritingScorerAgent expects 'text' in context
            context["text"] = context.get("research_draft", "")
            context = await self.scorer_agent.run(context)
            score: Dict[str, Any] = context["writing_score"]

            overall = float(score.get("overall", 0.0))
            revisions.append(
                {
                    "step": step,
                    "draft": context.get("research_draft", ""),
                    "score": score,
                }
            )

            self.logger.info(
                "ResearchResponsePipeline step=%s overall=%.2f breakdown=%s",
                step,
                overall,
                score.get("breakdown"),
            )

            # Keep track of the best so far
            if not best_score or overall > float(best_score.get("overall", -1.0)):
                best_score = score
                best_draft = context.get("research_draft", "")

            # Early stop if good enough
            if overall >= self.target_overall:
                self.logger.info(
                    "Early stopping research refinement: overall=%.2f >= target=%.2f",
                    overall,
                    self.target_overall,
                )
                break

            # 2b) Refine draft using scores
            context["writing_score"] = score
            context = await self.refine_agent.run(context)

        # 3) Finalize outputs
        context["research_answer"] = best_draft
        context["research_answer_score"] = best_score
        context["research_revisions"] = revisions

        # Cleanup scratch keys if you like
        # context.pop("text", None)
        # context.pop("research_draft", None)

        return context
