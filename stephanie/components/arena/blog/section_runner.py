# stephanie/components/arena/blog/section_runner.py
from __future__ import annotations
from typing import Any, Dict, List, Optional

from stephanie.components.arena.knowledge_arena import KnowledgeArena  # your existing arena :contentReference[oaicite:1]{index=1}


class ArenaBlogSectionRunner:
    """
    Single responsibility:
      run KnowledgeArena for ONE blog section.
    """

    def __init__(
        self,
        *,
        arena: KnowledgeArena,
        topk_support: int = 3,
    ):
        self.arena = arena
        self.topk_support = int(topk_support)

    async def select_supports(
        self,
        *,
        problem_text: str,
        candidates: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
        emit: Optional[Any] = None,
        run_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        result = await self.arena.run(
            problem_text,
            candidates,
            context=context or {},
            emit=emit,
            run_meta=run_meta or {},
        )

        scored_pool = list(result.get("scored_pool") or [])
        scored_pool.sort(key=lambda x: float((x.get("score") or {}).get("overall", 0.0)), reverse=True)

        supports = scored_pool[: max(1, self.topk_support)]

        return {
            "winner": result.get("winner") or {"text": "", "score": {"overall": 0.0}},
            "supports": supports,
            "scored_pool": scored_pool,
        }
