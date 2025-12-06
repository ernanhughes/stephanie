# stephanie/components/information/agents/explainer_judge_agent.py
from __future__ import annotations

import json
import logging
from typing import Any, Dict

from stephanie.agents.base_agent import BaseAgent
from stephanie.services.prompt_service import LLMRole

log = logging.getLogger(__name__)


class ExplainerJudgeAgent(BaseAgent):
    """
    Measures whether reflection improved the explainer for a specific paper.
    Compares v1 vs v2 against a source summary and returns JSON scores.

    Output schema:
    {
      "paper_id": str,
      "task_id": str,
      "v1_scores": {"clarity": float, "faithfulness": float, "usefulness": float},
      "v2_scores": {"clarity": float, "faithfulness": float, "usefulness": float},
      "winner": "v1" | "v2" | "TIE" | "ERROR",
      "improvement": str,
      "raw_judge_text": str
    }
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg=cfg, memory=memory, container=container, logger=logger)
        self.prompt = self.container.get("prompt")
        self.min_improvement = float(cfg.get("min_improvement", 0.2))

    async def judge_pair(
        self,
        *,
        paper_id: str,
        task_id: str,
        paper_summary: str,
        draft_v1: str,
        draft_v2: str,
    ) -> Dict[str, Any]:
        prompt = self._build_prompt(
            paper_summary=paper_summary,
            draft_v1=draft_v1,
            draft_v2=draft_v2,
        )

        try:
            raw = await self.prompt.run_prompt(
                prompt_text=prompt,
                context=None,
                role=LLMRole.JUDGE_EXPLAINER,
            )
        except Exception as e:
            log.error(f"ExplainerJudgeAgent: prompt failed: {e}")
            return {
                "paper_id": paper_id,
                "task_id": task_id,
                "v1_scores": {"clarity": 0.0, "faithfulness": 0.0, "usefulness": 0.0},
                "v2_scores": {"clarity": 0.0, "faithfulness": 0.0, "usefulness": 0.0},
                "winner": "ERROR",
                "improvement": f"judge_pair failed: {e}",
                "raw_judge_text": "",
            }

        v1_scores = {"clarity": 0.0, "faithfulness": 0.0, "usefulness": 0.0}
        v2_scores = {"clarity": 0.0, "faithfulness": 0.0, "usefulness": 0.0}
        winner = "ERROR"
        improvement = ""

        if raw and raw.strip():
            try:
                data = json.loads(raw)
                v1_scores = data.get("v1_scores", v1_scores)
                v2_scores = data.get("v2_scores", v2_scores)
                winner = data.get("winner", "ERROR")
                improvement = data.get("improvement", "")
            except json.JSONDecodeError:
                log.warning("ExplainerJudgeAgent: non-JSON judge output; falling back")
                # Fallback: treat as freeform explanation
                improvement = f"[Non-JSON judge output] {raw.strip()[:300]}..."

        result = {
            "paper_id": paper_id,
            "task_id": task_id,
            "v1_scores": v1_scores,
            "v2_scores": v2_scores,
            "winner": winner,
            "improvement": improvement,
            "raw_judge_text": raw,
        }

        # ---- NEW: log training events if available ----
        tes = getattr(self.memory, "training_events", None)
        if tes and winner in ("v1", "v2"):
            try:
                dimension = "explainer_quality"
                query_text = paper_summary

                # Pointwise events
                tes.insert_pointwise(
                    {
                        "model_key": "smart_v1",
                        "dimension": dimension,
                        "query_text": query_text,
                        "cand_text": draft_v1,
                        "label": 1 if winner == "v1" else 0,
                        "weight": 1.0,
                        "trust": float(
                            sum(v1_scores.values()) / max(len(v1_scores), 1)
                        ),
                        "goal_id": paper_id,
                        "pipeline_run_id": task_id,
                        "agent_name": "explainer_judge",
                        "source": "explainer_pair_judge",
                        "meta": {"role": "v1"},
                    }
                )
                tes.insert_pointwise(
                    {
                        "model_key": "smart_v2",
                        "dimension": dimension,
                        "query_text": query_text,
                        "cand_text": draft_v2,
                        "label": 1 if winner == "v2" else 0,
                        "weight": 1.0,
                        "trust": float(
                            sum(v2_scores.values()) / max(len(v2_scores), 1)
                        ),
                        "goal_id": paper_id,
                        "pipeline_run_id": task_id,
                        "agent_name": "explainer_judge",
                        "source": "explainer_pair_judge",
                        "meta": {"role": "v2"},
                    }
                )

                # Pairwise preference event (winner vs loser)
                if winner == "v2":
                    pos_text, neg_text = draft_v2, draft_v1
                    pos_key, neg_key = "smart_v2", "smart_v1"
                elif winner == "v1":
                    pos_text, neg_text = draft_v1, draft_v2
                    pos_key, neg_key = "smart_v1", "smart_v2"
                else:
                    pos_text, neg_text, pos_key, neg_key = None, None, None, None

                if pos_text and neg_text:
                    tes.insert_pairwise(
                        {
                            "model_key": pos_key,
                            "dimension": dimension,
                            "query_text": query_text,
                            "pos_text": pos_text,
                            "neg_text": neg_text,
                            "weight": 1.0,
                            "trust": 1.0,
                            "goal_id": paper_id,
                            "pipeline_run_id": task_id,
                            "agent_name": "explainer_judge",
                            "source": "explainer_pair_judge",
                            "meta": {"loser": neg_key},
                        }
                    )
            except Exception as e:
                log.warning(f"ExplainerJudgeAgent: failed to log training events: {e}")

        return result

    def _build_prompt(
        self,
        *,
        paper_summary: str,
        draft_v1: str,
        draft_v2: str,
    ) -> str:
        return f"""
You are an expert evaluator of AI research explainers.

Your job is to compare two explainer drafts (V1 and V2) for the same ML paper.
You must be STRICT about correctness and usefulness for practitioners.

PAPER SUMMARY:
<<<PAPER>>>
{paper_summary}
<<<END PAPER>>>

DRAFT VERSION 1:
<<<V1>>>
{draft_v1}
<<<END V1>>>

DRAFT VERSION 2:
<<<V2>>>
{draft_v2}
<<<END V2>>>

For each draft, assign 1–5 scores on:
- CLARITY: How easy is it to understand the main ideas?
- FAITHFULNESS: How well does it match the paper's actual content?
- USEFULNESS: How helpful would this be for a practitioner who wants to apply the ideas?

Then determine:
- WINNER: Which draft is better overall? ("v1", "v2", or "TIE")
- IMPROVEMENT: A short natural language description of how the better draft improves over the other, or why they are tied.

Respond in valid JSON ONLY, with this structure:

{{
  "v1_scores": {{"clarity": 1.0, "faithfulness": 1.0, "usefulness": 1.0}},
  "v2_scores": {{"clarity": 1.0, "faithfulness": 1.0, "usefulness": 1.0}},
  "winner": "v2",
  "improvement": "V2 adds motivation and concrete examples."
}}
"""
