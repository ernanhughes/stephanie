# stephanie/components/tree/task_executor.py
"""
Fix (2025-10-30):
- On scoring failure, do not call OutputVerifier.verify with wrong kwargs.
- Return a well-formed error result, or pass a minimal dict into verifier.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional

from stephanie.agents.base_agent import BaseAgent
from stephanie.components.tree.output_verifier import OutputVerifier
from stephanie.scoring.scorable import Scorable, ScorableType
from stephanie.services.scoring_service import ScoringService


class TaskExecutor:
    def __init__(self, agent: BaseAgent, container, verifier: Optional[OutputVerifier] = None, timeout: int = 30):
        self.agent = agent
        self.container = container
        self.verifier = verifier or OutputVerifier()
        self.timeout = timeout
        self.scoring: ScoringService = container.get("scoring")

    async def execute_task(self, task_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        task_type = context.get("task_type", "prompt_improvement")

        try:
            scorable = Scorable(
                id=None,
                text=task_text,
                target_type=ScorableType.PROMPT if "prompt" in task_type else ScorableType.TEXT,
            )
            scorer_name = context.get("scorer", "sicql")

            bundle = self.scoring.score(
                scorer_name=scorer_name,
                scorable=scorable,
                context=context,
                dimensions=context.get("dimensions", ["alignment"]),
            )
            metric = float(bundle.aggregate())
            flat = bundle.flatten(
                include_scores=True,
                include_weights=False,
                include_sources=False,
                include_rationales=False,
                include_attributes=True,
                include_meta=False,
                numeric_only=True,
                sep=".",
                attr_prefix="attr",
            )
            vector: Dict[str, float] = {f"{scorer_name}.{k}": float(v) for k, v in flat.items()}
            vector[f"{scorer_name}.aggregate"] = metric

            return {
                "metric": metric,
                "summary": task_text[:400],
                "merged_output": json.dumps({"score": metric, "text": task_text}),
                "stdout": json.dumps({"score": metric}),
                "stderr": "",
                "returncode": 0,
                "vector": vector,
                "is_bug": False,
            }

        except Exception as e:
            # Proper, minimal error result; keep shapes expected by verifier
            err_result = {
                "metric": 0.0,
                "summary": f"Scoring failed: {e}",
                "merged_output": "",
                "vector": {},
            }
            # Optionally verify (will mark is_bug True if thresholds unmet)
            verified = self.verifier.verify(err_result, stderr=str(e), has_submission_file=False)
            verified.setdefault("is_bug", True)
            return verified

    async def evaluate_text(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        return await self.execute_task(text, context)

    async def llm_feedback(self, plan: str, context: Dict[str, Any]) -> str:
        prompt = f"""
Evaluate this plan against the stated goal.
Goal: {context.get('goal', {}).get('goal_text', 'N/A')}
Plan:
{plan}

Provide short structured feedback (1-3 sentences).
"""
        resp = await self.agent.async_call_llm(prompt, context=context)
        return resp.strip()

    def hash_text(self, text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
