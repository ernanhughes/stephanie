#  stephanie/components/information/learning/blog_judge.py
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

DEFAULT_RUBRIC = {
    "clarity": "Is it easy to read and understand for a technical audience?",
    "faithfulness": "Does it stay true to the paper(s) and avoid hallucinations?",
    "structure": "Is the flow coherent with useful headings and narrative?",
    "usefulness": "Does it provide insights, intuition, and actionable understanding?",
}

@dataclass
class BlogJudgeResult:
    overall: float
    rubric: Dict[str, float]
    rationale: str
    issues: Optional[list[str]] = None
    suggestions: Optional[list[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall": self.overall,
            "rubric": self.rubric,
            "rationale": self.rationale,
            "issues": self.issues or [],
            "suggestions": self.suggestions or [],
        }


class BlogJudge:
    """
    LLM judge for blog quality. Produces stable JSON.
    Designed to be used *after* blog generation to label runs.
    """

    def __init__(self, container, memory, cfg: Optional[dict] = None):
        self.container = container
        self.memory = memory
        self.cfg = cfg or {}
        self.prompt_service = container.prompt_service  # your existing PromptService
        self.model = self.cfg.get("model", "ollama/llama3.1:8b")
        self.rubric = self.cfg.get("rubric", DEFAULT_RUBRIC)

    def _prompt_hash(self, prompt: str) -> str:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]

    async def score(
        self,
        *,
        blog_md: str,
        seed_title: str = "",
        seed_summary: str = "",
        neighbor_summary: str = "",
    ) -> Dict[str, Any]:
        rubric_json = json.dumps(self.rubric, ensure_ascii=False, indent=2)

        prompt = f"""
You are a strict evaluator of technical blog posts generated from research papers.

Return ONLY valid JSON, no markdown, no commentary.

Rubric (0-10 each):
{rubric_json}

Context:
- Seed title: {seed_title}
- Seed summary: {seed_summary}
- Neighbor summary (optional): {neighbor_summary}

Blog to judge:
<<<BLOG
{blog_md}
BLOG>>>

Output JSON schema:
{{
  "overall": <number 0-100>,
  "rubric": {{
    "clarity": <0-10>,
    "faithfulness": <0-10>,
    "structure": <0-10>,
    "usefulness": <0-10>
  }},
  "rationale": "<short paragraph>",
  "issues": ["..."],
  "suggestions": ["..."]
}}
""".strip()

        try:
            resp = await self.prompt_service.complete(
                model=self.model,
                prompt=prompt,
                temperature=self.cfg.get("temperature", 0.0),
            )
            text = (resp or "").strip()
            data = json.loads(text)

            # light validation / defaults
            overall = float(data.get("overall", 0.0))
            rubric = data.get("rubric", {}) or {}
            rationale = str(data.get("rationale", "")).strip()

            result = BlogJudgeResult(
                overall=overall,
                rubric={k: float(rubric.get(k, 0.0)) for k in self.rubric.keys()},
                rationale=rationale,
                issues=list(data.get("issues") or []),
                suggestions=list(data.get("suggestions") or []),
            )

            return {
                "result": result.to_dict(),
                "judge_model": self.model,
                "judge_prompt_hash": self._prompt_hash(prompt),
            }

        except Exception as e:
            log.warning("BlogJudge failed: %s", e)
            return {
                "result": BlogJudgeResult(overall=0.0, rubric={k: 0.0 for k in self.rubric}, rationale=str(e)).to_dict(),
                "judge_model": self.model,
                "judge_prompt_hash": None,
                "error": str(e),
            }
