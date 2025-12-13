# stephanie/components/information/agents/paper_blog_judge.py
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from stephanie.agents.base_agent import BaseAgent

log = logging.getLogger(__name__)


def _safe_read_text(path: str, max_chars: int) -> str:
    try:
        p = Path(path)
        if not p.exists():
            return ""
        txt = p.read_text(encoding="utf-8", errors="ignore")
        return txt[:max_chars]
    except Exception:
        return ""


def _extract_json_obj(raw: str) -> Optional[dict]:
    """
    Robust-ish JSON extraction:
    - If model wraps JSON in text, we try to grab first {...} block.
    """
    raw = raw.strip()
    if not raw:
        return None
    # Fast path
    try:
        return json.loads(raw)
    except Exception:
        pass

    # Bracket carve-out
    l = raw.find("{")
    r = raw.rfind("}")
    if l >= 0 and r > l:
        candidate = raw[l : r + 1]
        try:
            return json.loads(candidate)
        except Exception:
            return None
    return None


@dataclass
class BlogJudgeResult:
    overall_score: float
    dims: Dict[str, float]
    rationale: str
    critical_issues: list[str]
    rewrite_instructions: str
    confidence: float

    def to_dict(self) -> dict:
        return {
            "overall_score": self.overall_score,
            "dims": self.dims,
            "rationale": self.rationale,
            "critical_issues": self.critical_issues,
            "rewrite_instructions": self.rewrite_instructions,
            "confidence": self.confidence,
        }


class PaperBlogJudgeAgent(BaseAgent):
    """
    Stage: paper_blog_judge
    Reads the generated blog from disk (path in context), scores it with LLM,
    returns a compact JSON result, and (optionally) persists to PaperStore.
    """

    def __init__(self, cfg, memory, container, logger=None):
        super().__init__(cfg, memory, container, logger=logger)
        self.prompt = container.get("prompt")  # PromptService
        self.model_name = cfg.get("model_name", "ollama/llama3.1:8b")
        self.max_blog_chars = int(cfg.get("max_blog_chars", 18000))
        self.max_report_chars = int(cfg.get("max_report_chars", 6000))

    def _build_prompt(
        self,
        *,
        arxiv_id: str,
        title: str,
        paper_summary: str,
        blog_md: str,
        pipeline_report_md: str,
    ) -> str:
        # Keep it simple + strict JSON return
        return f"""
You are a strict evaluator for an AI-generated technical blog post.

Paper:
- arXiv: {arxiv_id}
- Title: {title}
- Summary (may be empty): {paper_summary}

Context report (may be empty, truncated):
{pipeline_report_md}

Blog markdown to evaluate (truncated):
{blog_md}

Rubric (0-10 each):
- clarity
- faithfulness (no hallucinated claims/citations)
- structure
- usefulness
- technical_accuracy

Return ONLY valid JSON with this schema:
{{
  "overall_score": <0-100 number>,
  "dims": {{
    "clarity": <0-10>,
    "faithfulness": <0-10>,
    "structure": <0-10>,
    "usefulness": <0-10>,
    "technical_accuracy": <0-10>
  }},
  "rationale": "<short explanation>",
  "critical_issues": ["..."],
  "rewrite_instructions": "<very actionable bullet-like instructions to improve the blog>",
  "confidence": <0.0-1.0>
}}
""".strip()

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        arxiv_id = context.get("arxiv_id") or context.get("paper_arxiv_id") or ""
        title = context.get("paper_title") or context.get("title") or arxiv_id
        paper_summary = context.get("paper_summary") or ""

        # Prefer file paths (keeps DB context small)
        blog_path = context.get("paper_blog_path") or context.get("blog_path")
        report_path = context.get("paper_pipeline_report_path") or context.get("pipeline_report_path")

        blog_md = ""
        if blog_path:
            blog_md = _safe_read_text(blog_path, self.max_blog_chars)
        else:
            # fallback (avoid huge)
            blog_md = (context.get("paper_blog_markdown") or "")[: self.max_blog_chars]

        pipeline_report_md = ""
        if report_path:
            pipeline_report_md = _safe_read_text(report_path, self.max_report_chars)

        if not blog_md.strip():
            log.warning("[PaperBlogJudgeAgent] No blog content found (path=%s).", blog_path)
            context["paper_blog_judge"] = {"error": "no_blog_content"}
            return context

        prompt = self._build_prompt(
            arxiv_id=arxiv_id,
            title=title,
            paper_summary=paper_summary,
            blog_md=blog_md,
            pipeline_report_md=pipeline_report_md,
        )

        raw = await self.prompt.complete(prompt=prompt, model=self.model_name)
        data = _extract_json_obj(raw) or {}

        # Safe defaults (backward compatible / never crash)
        res = BlogJudgeResult(
            overall_score=float(data.get("overall_score", 0.0) or 0.0),
            dims=dict(data.get("dims") or {}),
            rationale=str(data.get("rationale", "") or ""),
            critical_issues=list(data.get("critical_issues") or []),
            rewrite_instructions=str(data.get("rewrite_instructions", "") or ""),
            confidence=float(data.get("confidence", 0.0) or 0.0),
        )

        context["paper_blog_judge"] = res.to_dict()
        context["ai_blog_score"] = res.overall_score
        context["ai_blog_rationale"] = res.rationale
        context["blog_rewrite_instructions"] = res.rewrite_instructions

        log.info(
            "[PaperBlogJudgeAgent] Judged blog arxiv_id=%s overall=%.1f conf=%.2f",
            arxiv_id,
            res.overall_score,
            res.confidence,
        )

        # Optional persistence (only if your PaperStore has these methods)
        try:
            run_id = context.get("run_id") or context.get("pipeline_run_id")
            if getattr(self.memory, "papers", None) and run_id:
                # Pick one of these patterns depending on what you implemented:
                if hasattr(self.memory.papers, "save_run_event"):
                    self.memory.papers.save_run_event(
                        run_id=run_id,
                        paper_id=arxiv_id,
                        event_type="blog_judge",
                        payload=res.to_dict(),
                    )
                elif hasattr(self.memory.papers, "update_run"):
                    self.memory.papers.update_run(
                        run_id=run_id,
                        paper_id=arxiv_id,
                        patch={"ai_blog_score": res.overall_score, "ai_blog_rationale": res.rationale},
                    )
        except Exception as e:
            log.warning("[PaperBlogJudgeAgent] Persist failed (non-fatal): %s", e)

        return context
