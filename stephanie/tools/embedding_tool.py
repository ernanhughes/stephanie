# stephanie/tools/embedding_tool.py
from __future__ import annotations

import logging
from stephanie.tools.base_tool import BaseTool
from stephanie.utils.embed_utils import as_list_floats

log = logging.getLogger(__name__)

class EmbeddingTool(BaseTool):
    """
    Compute + store embeddings for scorables.
    """

    name = "embedding"

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)

        self.embedding_service = memory.embedding
        self.persist = bool(cfg.get("persist", True))
        self.force = bool(cfg.get("force", False))  # recompute even if exists

    # ------------------------------------------------------------------    
    async def apply(self, scorable, context: dict):
        """
        Compute embedding for scorable.text and persist it.
        """
        text = scorable.text or ""
        if not text.strip():
            return scorable

        # Try loading existing embeddings if allowed
        embed_id = self.memory.scorable_embeddings.get_or_create(scorable)
        log.debug(f"[EmbeddingTool] embedding id={embed_id}")
        vec = self.embedding_service.get_or_create(text)

        floats = as_list_floats(vec)
        # Attach to scorable meta
        scorable.meta.setdefault("embeddings", {})["global"] = floats

        return scorable

