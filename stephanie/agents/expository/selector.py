# stephanie/agents/expository/selector.py
from __future__ import annotations

from stephanie.agents.base_agent import BaseAgent
from stephanie.models.expository import BlogDraft


class DraftSelector(BaseAgent):
    """
    Computes r_solve (doc-native) and marks kept drafts. No citations used.
    """
    async def run(self, draft_id: int, cfg):
        async with self.session_maker() as s:
            d = await s.get(BlogDraft, draft_id)
            ok_read = float(d.readability >= cfg.quality.min_readability)
            ok_coh  = float(d.local_coherence >= cfg.quality.min_adjacent_coherence)
            rep_pen = d.repetition_penalty
            r_solve = max(0.0, min(1.0, 0.5*ok_read + 0.5*ok_coh - 0.1*rep_pen))
            d.kept = r_solve >= cfg.quality.keep_threshold
            await s.commit()
        return r_solve
