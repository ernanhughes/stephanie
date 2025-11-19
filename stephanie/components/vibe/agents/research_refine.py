from __future__ import annotations

from typing import Any, Dict

import logging

from stephanie.agents.base_agent import BaseAgent

log = logging.getLogger(__name__)


class ResearchRefineAgent(BaseAgent):
    """
    Takes a draft + writing scores/breakdown and produces an improved version.

    You can later swap this for a PlanTrace-aware refiner or use HRM feedback too.
    """

    def __init__(self, cfg: Dict[str, Any], memory: Any, container: Any, logger: Any):
        super().__init__(cfg, memory, container, logger)
        self.llm = container.get("llm_service")
        self.max_tokens = cfg.get("max_tokens", 1024)

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        draft: str = context.get("research_draft", "")
        score: Dict[str, Any] = context.get("writing_score", {})
        rubric_breakdown: Dict[str, float] = score.get("breakdown", {})

        question: str = context.get("research_question", "")
        notes: str = context.get("research_notes", "")

        prompt = self._build_prompt(question, notes, draft, score, rubric_breakdown)

        improved = await self.llm.complete(
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=self.cfg.get("temperature", 0.3),
        )

        context["research_draft"] = improved
        context.setdefault("research_revisions", []).append(
            {
                "draft": improved,
                "score": score,
            }
        )
        return context

    def _build_prompt(
        self,
        question: str,
        notes: str,
        draft: str,
        score: Dict[str, Any],
        breakdown: Dict[str, float],
    ) -> str:
        return (
            "You are revising a research explanation.\n\n"
            f"User question:\n{question}\n\n"
            f"Relevant notes / sources:\n{notes}\n\n"
            "Current draft:\n"
            f"{draft}\n\n"
            "Automated rubric scores for this draft (0–100):\n"
            f"{breakdown}\n\n"
            "Your job:\n"
            "- Improve CLARITY and STRUCTURE.\n"
            "- Fix any TECHNICAL issues or ambiguities.\n"
            "- Increase DEPTH where helpful, but stay concise.\n"
            "- Make the answer more ACTIONABLE for implementing in Stephanie.\n"
            "- Keep the calm, precise 'Stephanie research partner' vibe.\n\n"
            "Rewrite the answer in full, applying these improvements. Do not mention "
            "scores or rubrics in the text. Output only the revised answer.\n"
        )
