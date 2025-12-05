from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from stephanie.agents.base_agent import BaseAgent
from stephanie.components.information.agents.critic import IdeaCriticHead
from stephanie.components.information.agents.idea import IdeaGenerationHead
from stephanie.memory.idea_store import IdeaStore
from stephanie.types.idea import Idea

import logging
log = logging.getLogger(__name__)


class CreativeAssociationAgent(BaseAgent):
    """
    Coordinates:
      - IdeaGenerationHead
      - IdeaCriticHead
      - IdeaStore

    Produces:
      - context["ideas_raw"]    : all generated ideas with scores
      - context["ideas_accepted"]: filtered high-potential ideas
    """
    def __init__(self, cfg, memory, container, logger):
        super().__init__(
            cfg=cfg, memory=memory, container=container, logger=logger
        )
        self.gen = IdeaGenerationHead(cfg, memory, container, logger)
        self.critic = IdeaCriticHead(cfg, memory, container, logger)
        self.idea_store: IdeaStore = self.memory.ideas

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:

        # 1) Generate frontier ideas
        ideas: List[Idea] = await self.gen.generate_frontier_ideas()
        if not ideas:
            log.warning("CreativeAssociationAgent: no ideas generated")
            context["ideas_raw"] = []
            context["ideas_accepted"] = []
            return context

        # 2) Critique / score in parallel
        scored: List[Idea] = await asyncio.gather(
            *[self.critic.evaluate(idea) for idea in ideas],
            return_exceptions=False,
        )

        # 3) Filter and store
        min_final = self.cfg.get("min_r_final", 0.65)
        accepted = [i for i in scored if i.r_final >= min_final]

        if accepted:
            self.idea_store.upsert_ideas(accepted)

        context["ideas_raw"] = scored
        context["ideas_accepted"] = accepted
        return context
