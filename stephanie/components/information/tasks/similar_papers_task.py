# stephanie/components/information/tasks/hf_similar_paper_task.py
from __future__ import annotations

import asyncio
import logging
from typing import List, Optional

from stephanie.tools.huggingface_tool import recommend_similar_papers

log = logging.getLogger(__name__)


class HFSimilarPaperTask:
    """
    Backwards-compatible wrapper around HFSimilarPaperTool.

    Agents should call:
        task = HFSimilarPaperTask(cfg=..., memory=..., container=..., logger=...)
        recs = await task.run(arxiv_id="2506.21734", limit=10)

    This stays thin; all logic lives in the tool.
    """

    def __init__(self, cfg, memory, container, logger=None):
        # Pull your config however you like; keep defaults safe.
        # Example: cfg.information.similar.hf_max_limit
        max_limit = 16
        try:
            max_limit = int(getattr(getattr(getattr(cfg, "information", None), "similar", None), "hf_max_limit", 16))
        except Exception:
            pass

        # Import here to avoid import-time coupling if your HF tool loads lazily
        from stephanie.tools.huggingface_tool import recommend_similar_papers  # <-- adjust to your real import path

        tool_cfg = HFSimilarPaperConfig(max_limit=max_limit)
        self.tool = HFSimilarPaperTool(cfg=tool_cfg, recommender=recommend_similar_papers, logger=logger or log)

    async def run(self, *, arxiv_id: str, limit: int = 10) -> List["PaperReferenceRecord"]:
        # HF call is sync; don’t block the event loop.
        return await asyncio.to_thread(self.tool.get_similar_for_arxiv, arxiv_id=arxiv_id, limit=limit)
