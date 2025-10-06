# stephanie/agents/expository/extractor.py
from __future__ import annotations

from typing import List

from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.expository.heuristics import (bloggability_score,
                                                    compute_features,
                                                    expository_score)
from stephanie.models.expository import ExpositorySnippet


class ExpositoryExtractorAgent(BaseAgent):
    """
    Input: parsed doc (sections -> paragraphs) from your DocumentProfilerAgent
    Output: ExpositorySnippet rows with features and scores (no fact checks)
    """
    async def run(self, doc, cfg) -> List[ExpositorySnippet]:
        out = []
        for sec in doc.sections:
            section_name = (sec.title or "").strip()
            for idx, para in enumerate(sec.paragraphs):
                f = compute_features(para.text, section_name)
                s_exp = expository_score(f)
                s_blog = bloggability_score(para.text, f, cfg.extractor)
                if s_exp <= 0: 
                    continue
                out.append(ExpositorySnippet(
                    doc_id=doc.id, section=section_name, order_idx=idx,
                    text=para.text, features=f,
                    expository_score=s_exp, bloggability_score=s_blog
                ))
        # persist via session; return kept snippets
        async with self.session_maker() as s:
            s.add_all(out); await s.commit()
        return out
