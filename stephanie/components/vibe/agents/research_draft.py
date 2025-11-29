from __future__ import annotations

import logging
from typing import Any, Dict

from stephanie.agents.base_agent import BaseAgent

log = logging.getLogger(__name__)


class ResearchDraftAgent(BaseAgent):
    """
    Generates an initial research-style answer given:
      - research_question
      - optional sources / notes

    This is intentionally simple; you can later swap in your PlanTrace-based
    pipelines or Blossom runners.
    """

    def __init__(self, cfg: Dict[str, Any], memory: Any, container: Any, logger: Any):
        super().__init__(cfg, memory, container, logger)
        self.llm = container.get("llm_service")  # adapt to your stack
        self.max_tokens = cfg.get("max_tokens", 1024)

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        question: str = context.get("research_question", "")
        notes: str = context.get("research_notes", "")
        style_hint: str = context.get("style_hint", "technical, calm, precise")

        prompt = self._build_prompt(question, notes, style_hint)

        draft = await self.llm.complete(
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=self.cfg.get("temperature", 0.4),
        )

        context["research_draft"] = draft
        return context

    def _build_prompt(self, question: str, notes: str, style_hint: str) -> str:
        return (
            "You are Stephanie, a research partner helping the user understand and "
            "implement ideas in their AI system.\n\n"
            f"User question:\n{question}\n\n"
            "Relevant notes / sources (may be partial or messy):\n"
            f"{notes}\n\n"
            "Write a clear, technically accurate explanation in the following style:\n"
            f"- Tone: {style_hint}\n"
            "- Structure: short sections with headings where useful\n"
            "- Focus: explain what it is, why it matters, and how to implement it "
            "in the Stephanie system.\n\n"
            "Draft your answer below:\n"
        )
