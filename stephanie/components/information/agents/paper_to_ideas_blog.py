# stephanie/components/information/agents/paper_to_ideas_blog.py
from __future__ import annotations

import asyncio
import logging
from typing import Dict, Any, List

from stephanie.agents.base_agent import BaseAgent
from stephanie.memory.idea_store import IdeaStore
from stephanie.types.idea import Idea 
from stephanie.components.information.models import ReasonedBlogResult

log = logging.getLogger(__name__)


class PaperToIdeasAndBlogAgent(BaseAgent):
    """
    High-level orchestrator for demo:

      PDF(s) -> CaseBook/MemCube
             -> Section evidence
             -> Ideas (gen + critic)
             -> Blog draft that explains paper + 2–3 best ideas
             -> Scores & artefacts saved for inspection/demo.
    """
    def __init__(self, cfg, memory, container, logger):
        super().__init__(
            cfg=cfg, memory=memory, container=container, logger=logger
        )

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        docs = context.get("documents", [])
        if not docs:
            self.logger.warning("No documents provided to PaperToIdeasAndBlogAgent")
            return context

        ingest = self.container.get("information_ingest")
        section_agent = self.container.get("section_research")
        fast_ency = self.container.get("fast_encyclopedia")
        idea_gen = self.container.get("idea_generation_head")
        idea_critic = self.container.get("idea_critic_head")
        idea_store: IdeaStore = self.memory.ideas

        all_results = []

        for doc in docs:
            # 1) Ingest
            ingest_ctx = {"document": doc}
            ingest_out = await ingest.run(ingest_ctx)

            # 2) Section research
            sec_ctx = {"casebook_id": ingest_out["casebook_id"]}
            sec_out = await section_agent.run(sec_ctx)

            # 3) Fast encyclopedia view (summary + bullets + sections)
            enc_ctx = {
                "casebook_id": ingest_out["casebook_id"],
                "sections": sec_out.get("section_research", {}),
            }
            enc_out = await fast_ency.run(enc_ctx)
            reasoned: ReasonedBlogResult = enc_out["reasoned_blog"]

            # 4) Ideas: generate + critic
            ideas = await idea_gen.generate_frontier_ideas()
            scored = await asyncio.gather(*[idea_critic.evaluate(i) for i in ideas])
            min_final = self.cfg.get("min_r_final", 0.65)
            accepted = [i for i in scored if i.r_final >= min_final]
            if accepted:
                idea_store.upsert_ideas(accepted)

            # 5) Attach top ideas back into the blog narrative
            enriched_blog = self._inject_ideas_into_blog(
                reasoned.blog_text,
                sorted(accepted, key=lambda x: x.r_final, reverse=True)[:3],
            )

            all_results.append(
                {
                    "doc_id": doc.id,
                    "blog_enriched": enriched_blog,
                    "raw_blog": reasoned.blog_text,
                    "ideas_raw": scored,
                    "ideas_accepted": accepted,
                }
            )

        context["information_demo_results"] = all_results
        return context

    def _inject_ideas_into_blog(self, blog_text: str, ideas: List[Idea]) -> str:
        # Simple v1: append a "Future directions" section.
        if not ideas:
            return blog_text
        bullets = "\n".join(
            f"- **{i.title}** — {i.impact_summary} (score: {i.r_final:.2f})"
            for i in ideas
        )
        return blog_text + "\n\n## Future Directions\n\n" + bullets
