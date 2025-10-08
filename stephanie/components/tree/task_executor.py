# stephanie/components/tree/task_executor.py
"""
A general-purpose execution layer for Stephanie's agentic search.
Instead of running code, this focuses on evaluating *textual tasks*
using the system's scorers (MRQ, SICQL, EBT, etc.) or external LLMs.
"""

from __future__ import annotations
import json
import hashlib
from typing import Any, Dict, Optional

from stephanie.services.scoring_service import ScoringService
from stephanie.scoring.scorable import Scorable, ScorableType
from stephanie.agents.base_agent import BaseAgent
from stephanie.components.tree.output_verifier import OutputVerifier


class TaskExecutor:
    def __init__(self, agent: BaseAgent, container, verifier: Optional[OutputVerifier] = None, timeout: int = 30):
        """
        Args:
            agent: reference to the parent agent (for async_call_llm etc.)
            container: DI container (must contain 'scoring' service)
            verifier: optional OutputVerifier for fallback metric extraction
            timeout: max seconds for long-running async calls
        """
        self.agent = agent
        self.container = container
        self.verifier = verifier or OutputVerifier()
        self.timeout = timeout
        self.scoring: ScoringService = container.get("scoring")

    # --------------------------------------------------------------- #
    async def execute_task(self, task_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute or evaluate a task, depending on context.
        Usually this means scoring a piece of text, not running code.
        """
        task_type = context.get("task_type", "prompt_improvement")

        try:
            scorable = Scorable(
                id=None,
                text=task_text,
                target_type=ScorableType.PROMPT if "prompt" in task_type else ScorableType.TEXT,
            )
            bundle = await self.scoring.score(
                scorer_name=context.get("scorer", "sicql"),
                scorable=scorable,
                context=context,
                dimensions=context.get("dimensions", ["alignment"]),
            )
            metric = float(bundle.aggregate())
            return {
                "metric": metric,
                "summary": task_text[:400],
                "merged_output": json.dumps({"score": metric, "text": task_text}),
                "stdout": json.dumps({"score": metric}),
                "stderr": "",
                "returncode": 0,
            }

        except Exception as e:
            # Fallback: try a verification pass if ScoringService fails
            return self.verifier.verify(stdout="", stderr=str(e), has_submission_file=False)

    # --------------------------------------------------------------- #
    async def evaluate_text(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Optional alias for semantic equivalence with old interfaces."""
        return await self.execute_task(text, context)

    # --------------------------------------------------------------- #
    async def llm_feedback(self, plan: str, context: Dict[str, Any]) -> str:
        """Obtain qualitative feedback from an LLM for improvement cycles."""
        prompt = f"""
Evaluate this plan against the stated goal.
Goal: {context.get('goal', {}).get('goal_text', 'N/A')}
Plan:
{plan}

Provide short structured feedback (1-3 sentences).
"""
        resp = await self.agent.async_call_llm(prompt, context=context)
        return resp.strip()

    # --------------------------------------------------------------- #
    def hash_text(self, text: str) -> str:
        """Stable hash for caching."""
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
