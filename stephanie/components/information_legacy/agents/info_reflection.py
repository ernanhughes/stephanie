# stephanie/components/information/agents/info_reflection_agent.py
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from stephanie.agents.base_agent import BaseAgent
from stephanie.memory.reflection_store import ReflectionStore
from stephanie.services.prompt_service import LLMRole

log = logging.getLogger(__name__)


@dataclass
class MicroReflection:
    """Structured reflection output for a single draft."""
    task_id: str
    trace_id: int
    draft_text: str
    reference_text: str
    score: Optional[float]
    problems: List[Dict[str, str]]
    action_plan: List[str]
    raw_text: str


class InformationReflectionAgent(BaseAgent):
    """
    Minimal micro-reflection agent for the SMART pipeline.

    Responsibilities:
    - Take a draft blog post (+ optional source summary & score)
    - Ask an LLM to critique it and propose an action plan
    - Return a structured object that SmartPaperBuilder can feed into the second draft
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg=cfg, memory=memory, container=container, logger=logger)
        self.reflection_store: ReflectionStore = self.memory.reflections
        self.prompt = self.container.get("prompt")
        # You can add `self.reflection_store = memory.reflections` later if needed

    async def reflect_on_run(
        self,
        *,
        task_id: str,
        trace_id: int,
        draft_text: str,
        reference_text: Optional[str] = None,
        score: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Main entrypoint used by SmartPaperBuilderAgent.

        Returns:
        {
          "reflection_text": <raw LLM output>,
          "structured_output": {
            "problems": [...],
            "action_plan": [...]
          }
        }
        """
        prompt = self._build_micro_prompt(
            task_id=task_id,
            draft=draft_text,
            reference=reference_text,
            score=score,
        )

        try:
            raw = await self.prompt.run_prompt(
                prompt_text=prompt,
                context=None,
                role=LLMRole.REFLECTION,
            )
        except Exception as e:
            log.error(f"InformationReflectionAgent: prompt failed: {e}")
            raw = ""

        problems: List[Dict[str, str]] = []
        action_plan: List[str] = []

        if raw.strip():
            try:
                data = json.loads(raw)
                problems = data.get("problems", []) or []
                action_plan = data.get("action_plan", []) or []
            except json.JSONDecodeError:
                log.warning(
                    "InformationReflectionAgent: non-JSON reflection output, "
                    "falling back to single bullet"
                )
                action_plan = [raw.strip()[:300] + "..."]

        reflection = MicroReflection(
            task_id=task_id,
            trace_id=trace_id,
            draft_text=draft_text,
            reference_text=reference_text or "",
            score=score,
            problems=problems,
            action_plan=action_plan,
            raw_text=raw,
        )

        self.reflection_store.save_micro_reflection(
            task_id=task_id,
            trace_id=trace_id,
            draft_text=draft_text,
            reference_text=reference_text,
            score=score,
            problems=problems,
            action_plan=action_plan,
            raw_text=raw,
        )

        return {
            "reflection_text": reflection.raw_text,
            "structured_output": {
                "problems": reflection.problems,
                "action_plan": reflection.action_plan,
            },
        }

    def _build_micro_prompt(
        self,
        *,
        task_id: str,
        draft: str,
        reference: Optional[str],
        score: Optional[float],
    ) -> str:
        ref_block = f"\nREFERENCE (source summary):\n{reference}\n" if reference else ""
        score_block = f"(Previous automatic quality score: {score:.2f})\n" if score is not None else ""

        return f"""
You are an expert editor for AI research blog posts.

Your job is to:
1. Critique the draft.
2. Propose a concrete action plan for the next revision.

TASK ID: {task_id}
{score_block}

DRAFT:
<<<DRAFT>>>
{draft}
<<<END DRAFT>>>
{ref_block}

1. Identify specific PROBLEMS using categories like:
   - missing motivation
   - unclear explanation
   - inaccurate claim
   - poor structure
   - jargon-heavy
   - weak examples

For each problem, provide:
   - "category": short label
   - "description": one-sentence description of the issue
   - "evidence": a short quote or pointer from the draft

2. Then write an ACTION PLAN:
   - 3–5 bullet points
   - Each must be a specific edit instruction
   - Example: "Add a paragraph after the intro explaining why OOD robustness matters"

Respond in valid JSON:

{{
  "problems": [
    {{
      "category": "clarity",
      "description": "The introduction assumes the reader knows what OOD means.",
      "evidence": "In the first paragraph, 'OOD' is used without explanation."
    }}
  ],
  "action_plan": [
    "Add a short definition of OOD in the introduction.",
    "Give one concrete real-world example of OOD failure."
  ]
}}
"""
