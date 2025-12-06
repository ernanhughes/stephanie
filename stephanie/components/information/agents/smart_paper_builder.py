# stephanie/components/information/agents/smart_paper_builder.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
import asyncio
import logging

from stephanie.agents.base_agent import BaseAgent
from stephanie.services.prompt_service import LLMRole

from stephanie.types.idea import Idea  # your Pydantic model
from stephanie.memory.idea_store import IdeaStore
from stephanie.memory.reflection_store import ReflectionStore

log = logging.getLogger(__name__)


class SmartPaperBuilderAgent(BaseAgent):
    """
    End-to-end SMART paper builder:
    - Fetch similar papers
    - Ingest + structure
    - Generate explainer draft
    - Reflect & improve (SAMULE-style micro reflection)
    - Generate & critique research ideas
    - Assemble final blog artifact
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg=cfg, memory=memory, container=container, logger=logger)

        # Core services / tools
        self.prompt = self.container.get("prompt_service")
        self.similar_tool = self.container.get("similar_papers_tool")  # wrapper around recommend_similar_papers
        self.doc_ingestor = self.container.get("document_ingest_agent")
        self.info_reflection = self.container.get("info_reflection_agent")

        # Idea engine
        self.idea_gen = self.container.get("idea_generation_head")
        self.idea_critic = self.container.get("idea_critic_head")

        # Stores
        self.idea_store: IdeaStore = self.memory.ideas
        self.reflection_store: ReflectionStore = self.memory.reflections

        # Scoring
        self.scoring_agent = self.container.get("info_quality_scorer")

        # Thresholds
        self.min_quality = float(cfg.get("min_quality", 0.75))
        self.min_idea_score = float(cfg.get("min_idea_r_final", 0.65))
        self.max_ideas = int(cfg.get("max_ideas", 5))

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        paper_url: str = context["paper_url"]
        audience: str = context.get("audience", "practitioner")
        topic: str = context.get("topic", paper_url)

        # 1) Find similar papers
        similar = await self._find_similar_papers(paper_url)
        self.logger.info(f"Found {len(similar)} similar papers for {paper_url}")

        # 2) Ingest + structure (main + similar papers)
        ingest_result = await self._ingest_papers(paper_url, similar)
        main_doc_id = ingest_result["main_doc_id"]
        similar_doc_ids = ingest_result["similar_doc_ids"]

        # 3) Initial explainer draft
        draft_v1 = await self._generate_blog_draft(
            main_doc_id,
            similar_doc_ids,
            audience=audience,
            reflection_hint=None,
        )

        # 4) Score draft
        score_v1 = await self._score_draft(draft_v1, main_doc_id)
        self.logger.info(f"SMART draft_v1 score={score_v1:.3f}")

        draft_final = draft_v1
        reflection = None

        # 5) If below threshold: reflect & improve
        if score_v1 < self.min_quality:
            self.logger.warning("Draft below threshold; triggering reflection loop")
            reflection = await self._reflect_on_draft(
                topic=topic,
                draft=draft_v1,
                score=score_v1,
                main_doc_id=main_doc_id,
            )

            draft_final = await self._generate_blog_draft(
                main_doc_id,
                similar_doc_ids,
                audience=audience,
                reflection_hint=reflection.get("action_plan") if reflection else None,
            )
            score_v2 = await self._score_draft(draft_final, main_doc_id)
            self.logger.info(f"SMART draft_v2 score={score_v2:.3f}")
        else:
            self.logger.info("Draft above threshold; skipping reflection loop")

        # 6) Generate & critique research ideas
        ideas, accepted_ideas = await self._generate_and_score_ideas()
        self.logger.info(f"Generated {len(ideas)} ideas, accepted {len(accepted_ideas)}")

        if accepted_ideas:
            # Persist ideas
            await self._store_ideas(accepted_ideas)

        # 7) Assemble final blog artifact
        blog = self._assemble_blog(
            draft_final=draft_final,
            similar=similar,
            top_ideas=accepted_ideas,
            audience=audience,
        )

        context.update(
            {
                "blog_markdown": blog,
                "draft_v1": draft_v1,
                "reflection": reflection,
                "ideas_raw": ideas,
                "ideas_accepted": accepted_ideas,
            }
        )
        return context

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _find_similar_papers(self, paper_url: str) -> List[Dict[str, Any]]:
        try:
            return await self.similar_tool.find_similar(paper_url=paper_url, limit=10)
        except Exception as e:
            self.logger.error(f"SimilarPapersTool failed: {e}")
            return []

    async def _ingest_papers(
        self, paper_url: str, similar: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        # Let your existing ingest agent handle PDF download, parsing, MemCubes, etc.
        main_res = await self.doc_ingestor.run({"paper_url": paper_url})
        main_doc_id = main_res["doc_id"]

        similar_doc_ids: List[str] = []
        for sp in similar:
            try:
                res = await self.doc_ingestor.run({"paper_url": sp["url"]})
                similar_doc_ids.append(res["doc_id"])
            except Exception as e:
                self.logger.warning(f"Ingest failed for similar paper {sp.get('url')}: {e}")

        return {
            "main_doc_id": main_doc_id,
            "similar_doc_ids": similar_doc_ids,
        }

    async def _generate_blog_draft(
        self,
        main_doc_id: str,
        similar_doc_ids: List[str],
        audience: str,
        reflection_hint: Optional[List[str]],
    ) -> str:
        # Build a prompt that:
        # - Explains the main paper
        # - Draws on similar docs for context/contrast
        # - Applies any reflection hints if provided
        main_summary = await self._get_doc_summary(main_doc_id)
        similar_summaries = await self._get_similar_summaries(similar_doc_ids)

        prompt = self._build_blog_prompt(
            main_summary=main_summary,
            similar_summaries=similar_summaries,
            audience=audience,
            reflection_hint=reflection_hint,
        )

        out = await self.prompt.run_prompt(
            prompt_text=prompt,
            context=None,
            role=LLMRole.EXPLAINER,
        )
        return out or ""

    async def _score_draft(self, draft: str, main_doc_id: str) -> float:
        try:
            res = await self.scoring_agent.run(
                {
                    "doc_id": main_doc_id,
                    "generated_text": draft,
                }
            )
            return float(res.get("info_quality_score", 0.0))
        except Exception as e:
            self.logger.error(f"Draft scoring failed: {e}")
            return 0.0

    async def _reflect_on_draft(
        self,
        topic: str,
        draft: str,
        score: float,
        main_doc_id: str,
    ) -> Dict[str, Any]:
        # Minimal micro reflection – no trace dependency required for v1
        ref = await self.info_reflection.reflect_on_run(
            task_id=topic,
            trace_id=-1,
            draft_text=draft,
            reference_text=await self._get_doc_summary(main_doc_id),
            score=score,
        )
        # Expect ref["structured_output"]["action_plan"] from agent
        structured = ref.get("structured_output", {})
        return {
            "raw": ref.get("reflection_text", ""),
            "problems": structured.get("problems", []),
            "action_plan": structured.get("action_plan", []),
        }

    async def _generate_and_score_ideas(self) -> tuple[list[Idea], list[Idea]]:
        # Use the already-implemented IdeaGenerationHead + IdeaCriticHead
        ideas: List[Idea] = await self.idea_gen.generate_frontier_ideas()
        if not ideas:
            return [], []

        scored: List[Idea] = await asyncio.gather(
            *[self.idea_critic.evaluate(i) for i in ideas],
            return_exceptions=False,
        )
        accepted = [i for i in scored if i.r_final >= self.min_idea_score]
        return scored, accepted

    async def _store_ideas(self, ideas: List[Idea]) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: self.idea_store.upsert_ideas(ideas))

    async def _get_doc_summary(self, doc_id: str) -> str:
        # Thin wrapper around your existing summary retrieval
        summary_store = self.memory.summaries
        summary = await summary_store.get_summary(doc_id)
        return summary or ""

    async def _get_similar_summaries(self, doc_ids: List[str]) -> List[str]:
        summary_store = self.memory.summaries
        outs: List[str] = []
        for did in doc_ids:
            try:
                s = await summary_store.get_summary(did)
                if s:
                    outs.append(s)
            except Exception:
                continue
        return outs

    def _build_blog_prompt(
        self,
        main_summary: str,
        similar_summaries: List[str],
        audience: str,
        reflection_hint: Optional[List[str]],
    ) -> str:
        similar_block = ""
        if similar_summaries:
            joined = "\n\n".join(similar_summaries[:3])
            similar_block = f"""
Related papers (for context / contrast):
{joined}
"""

        reflection_block = ""
        if reflection_hint:
            fixes = "\n".join(f"- {f}" for f in reflection_hint)
            reflection_block = f"""
Critical feedback from previous draft:

{fixes}

Please explicitly apply these improvements.
"""

        return f"""
You are an expert AI explainer writing for a {audience} audience.

Main paper summary:
{main_summary}

{similar_block}

Write a clear, engaging blog-style explanation that covers:
- What problem this paper addresses
- Why it matters in practice
- The core idea, explained simply
- How it compares to related work (if relevant)
- Limitations and caveats

Use headings, short paragraphs, and examples where helpful.
Avoid heavy notation; focus on intuition.

{reflection_block}
"""
