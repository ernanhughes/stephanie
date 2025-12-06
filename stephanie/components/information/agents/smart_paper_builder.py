# stephanie/components/information/agents/smart_paper_builder.py
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from stephanie.agents.base_agent import BaseAgent
from stephanie.services.prompt_service import LLMRole
from stephanie.types.idea import Idea  # your Pydantic Idea model
import logging

log = logging.getLogger(__name__)

class SmartPaperBuilderAgent(BaseAgent):
    """
    SMART-style paper → blog section builder.

    Assumptions / Inputs (from earlier pipeline stages):
      - context["information"]:
          {
              "title": str,
              "abstract": str,
              "summary": str,             # high-level summary of the paper
              "key_points": List[str],    # optional
              "domain": str,              # optional
              "similar_papers": [...],    # optional
              ...
          }

      - context["ideas_accepted"]: List[Idea]
          High-scoring ideas from the idea engine (IdeaGenerationHead + IdeaCriticHead).

    Outputs:
      - context["smart_blog_draft"]: str
          A blog-style section explaining the paper and weaving in the best ideas.

      - context["smart_trace"]: Dict[str, Any]
          Lightweight trace capturing what went into this generation:
              {
                  "paper_title": ...,
                  "idea_ids": [...],
                  "prompt": "...",
                  "n_ideas_used": int,
              }

    This is intentionally **minimal but real**:
      - It uses your actual PromptService (new API).
      - It ties information + ideas into a single artefact you can show in a demo.
      - It is easy to extend later with more SMART-style loops (multiple drafts, critic, re-write, etc.).
    """

    def __init__(self, cfg, memory, container, logger) -> None:
        super().__init__(cfg=cfg, memory=memory, container=container, logger=logger)
        self.prompt_service = self.container.get("prompt")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        information: Dict[str, Any] = context.get("information") or {}
        ideas: List[Idea] = context.get("ideas_accepted") or []

        if not information:
            log.warning(
                "SmartPaperBuilderAgent: context['information'] missing or empty; "
                "falling back to minimal prompt."
            )

        if not ideas:
            log.warning(
                "SmartPaperBuilderAgent: context['ideas_accepted'] is empty; "
                "blog draft will not include novel idea directions."
            )

        # 1) Select top-k ideas to weave into the blog post
        top_ideas = self._select_top_ideas(ideas, k=self.cfg.get("max_ideas", 3))

        # 2) Build a single, explicit prompt
        prompt = self._build_blog_prompt(information, top_ideas)

        # 3) Call PromptService (new signature) to generate the blog draft
        blog_text = await self.prompt_service.run_prompt(
            prompt_text=prompt,
            context=context,           # pass pipeline context in case roles/sys need it
            role=LLMRole.EXPLAINER,    # or a dedicated BLOG_WRITER role if you add one
            params={
                "temperature": self.cfg.get("temperature", 0.7),
                "max_tokens": self.cfg.get("max_tokens", 1200),
            },
        )

        # 4) Store outputs + a tiny SMART-style trace
        context["smart_blog_draft"] = blog_text
        context["smart_trace"] = self._build_trace(information, top_ideas, prompt)

        log.info(
            "SmartPaperBuilderAgent: generated blog draft "
            f"(len={len(blog_text)} chars, ideas_used={len(top_ideas)})"
        )
        return context

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _select_top_ideas(self, ideas: List[Idea], k: int = 3) -> List[Idea]:
        """
        Select the top-k ideas by r_final. This is your 'SMART' selection step:
        you are explicitly using the critic's multi-objective score to decide
        what makes it into the narrative.
        """
        if not ideas:
            return []

        sorted_ideas = sorted(
            ideas,
            key=lambda i: float(getattr(i, "r_final", 0.0)),
            reverse=True,
        )
        return sorted_ideas[:k]

    def _build_blog_prompt(
        self,
        information: Dict[str, Any],
        ideas: List[Idea],
    ) -> str:
        """
        Compose a single rich prompt that:
          - Explains the paper.
          - Weaves in the selected ideas as 'our hypotheses' / 'future work'.
        """
        title = information.get("title", "Unknown Title")
        abstract = (information.get("abstract") or "").strip()
        summary = (information.get("summary") or "").strip()
        domain = information.get("domain") or "AI / machine learning"

        key_points = information.get("key_points") or []
        key_points_text = ""
        if key_points:
            bullet_lines = "\n".join(f"- {kp}" for kp in key_points[:6])
            key_points_text = f"\nKey points extracted from the paper:\n{bullet_lines}\n"

        # Format ideas into a readable block
        if ideas:
            idea_lines = []
            for idx, idea in enumerate(ideas, start=1):
                idea_lines.append(
                    f"{idx}. {idea.title}\n"
                    f"   Hypothesis: {idea.hypothesis}\n"
                    f"   Method (sketch): {idea.method}"
                )
            ideas_block = "\n".join(idea_lines)
        else:
            ideas_block = "None (you may still suggest plausible extensions)."

        prompt = f"""
You are helping write a blog post for technically literate readers
(e.g., ML engineers, researchers, and advanced students).

We have a **research paper** and a set of **novel research directions** that
our idea engine has generated.

---

PAPER METADATA
--------------
Title: {title}

Domain: {domain}

Abstract:
{abstract or "(no abstract available)"}

High-level summary:
{summary or "(no high-level summary available)"}
{key_points_text}

---

NOVEL RESEARCH DIRECTIONS (OUR OWN IDEAS, NOT CLAIMED BY THE PAPER)
-------------------------------------------------------------------
{ideas_block}

Each idea above has already been scored for:
- novelty
- feasibility
- utility
- risk

Assume these ideas are *worthwhile to discuss*, especially the first ones.

---

YOUR TASK
---------
Write a **single coherent blog post section** (approximately 800–1200 words) that:

1. Clearly explains what the paper is about and why it matters.
2. Draws out one or more limitations, gaps, or open questions in the work.
3. Introduces **our best new ideas** (from the list above) as:
   - potential extensions,
   - follow-up experiments, or
   - alternative framings of the problem.
4. Makes it obvious to the reader which ideas are from the *original paper*
   and which ideas are *our own proposals*.
5. Is written in a conversational but precise tone, suitable for a research blog
   (e.g., programmer.ie).

Structure the output roughly as:

- Short hook / intro
- "What this paper actually does"
- "Where the gaps or limitations are"
- "Where we think this could go next" (integrating our ideas)
- Light concluding paragraph

Do NOT use bullet lists in the final output. Write continuous prose.
Do NOT fabricate experimental results; when you speculate, say so explicitly.
"""
        return prompt.strip()

    def _build_trace(
        self,
        information: Dict[str, Any],
        ideas: List[Idea],
        prompt: str,
    ) -> Dict[str, Any]:
        """
        Minimal SMART-style trace: enough to debug + show in a demo.
        You can later evolve this into a full PlanTrace.
        """
        return {
            "paper_title": information.get("title"),
            "paper_domain": information.get("domain"),
            "idea_ids": [getattr(i, "id", None) for i in ideas],
            "n_ideas_used": len(ideas),
            "prompt": prompt,
        }
