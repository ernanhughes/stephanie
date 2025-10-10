# stephanie/agents/expository/ranker.py
from __future__ import annotations

from typing import List

from sqlalchemy import select

from stephanie.agents.base_agent import BaseAgent
from stephanie.models.expository import ExpositoryBuffer, ExpositorySnippet


class ExpositoryRanker(BaseAgent):
    async def run(self, topic: str, k: int = 12, min_exp: float = 0.45, min_blog: float = 0.5):
        async with self.session_maker() as s:
            rows = (await s.execute(
                select(ExpositorySnippet).where(
                    ExpositorySnippet.expository_score >= min_exp,
                    ExpositorySnippet.bloggability_score >= min_blog
                ).order_by(ExpositorySnippet.expository_score.desc())
            )).scalars().all()
            pick = rows[:k]
            buf = ExpositoryBuffer(topic=topic, snippet_ids=[r.id for r in pick])
            s.add(buf); await s.commit()
            return buf
