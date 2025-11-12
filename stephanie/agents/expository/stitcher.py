# stephanie/agents/expository/stitcher.py
from __future__ import annotations

from sqlalchemy import select

from stephanie.agents.base_agent import BaseAgent
from stephanie.models.expository import (BlogDraft, ExpositoryBuffer,
                                         ExpositorySnippet)

BRIDGE_TMPL = "In summary, {nxt} builds upon the preceding idea by {link}."
INTRO_TMPL  = "This section walks through the standard approach to {topic}, focusing on the parts practitioners already use."
OUTRO_TMPL  = "That’s the core recipe used across papers; the rest of the literature layers variations on these steps."

def simple_coherence(a: str, b: str, embed) -> float:
    # cosine between mean embeddings (reuse your EmbeddingStore)
    va = embed(a); vb = embed(b)
    num = (va * vb).sum()
    den = (va.norm() * vb.norm()) + 1e-9
    return float(num / den)

class BlogStitcher(BaseAgent):
    async def run(self, buffer_id: int, topic: str, target_words: int = 700):
        async with self.session_maker() as s:
            buf = await s.get(ExpositoryBuffer, buffer_id)
            snips = (await s.execute(
                select(ExpositorySnippet).where(ExpositorySnippet.id.in_(buf.snippet_ids))
            )).scalars().all()

        # order by section priority then by doc order
        order = sorted(snips, key=lambda r: (r.section.lower(), r.order_idx))
        paras = [p.text.strip() for p in order]

        # add a tiny intro & bridges (lightweight, no refs)
        blocks = [INTRO_TMPL.format(topic=topic)]
        for i, p in enumerate(paras):
            blocks.append(p)
            if i < len(paras) - 1:
                blocks.append(f"*{BRIDGE_TMPL.format(nxt='the next piece', link='expanding the pipeline')}*")
        blocks.append(OUTRO_TMPL)

        # crop/expand to target length (very light touch)
        md = "\n\n".join(blocks)
        words = len(md.split())
        if words > target_words * 1.2:
            md = " ".join(md.split()[: int(target_words * 1.2)])

        # quality metrics (internal)
        from stephanie.services.embedding_store import EmbeddingStore
        E = EmbeddingStore.get_default()
        local = []
        for a, b in zip(paras[:-1], paras[1:]):
            local.append(simple_coherence(a, b, E.embed_text))
        local_coh = sum(local)/max(1, len(local))
        read = 50.0  # optional: call your readability util

        draft = BlogDraft(
            topic=topic, source_snippet_ids=[s.id for s in order],
            draft_md=md, readability=read, local_coherence=local_coh
        )
        async with self.session_maker() as s:
            s.add(draft); await s.commit()
            return draft
