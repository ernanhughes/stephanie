# stephanie/components/ssp/retrievers/prompt_history.py
from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from .base import BaseRetriever


def _maybe_sync(x):
    # await coroutine or return sync result
    if asyncio.iscoroutine(x):
        return x
    async def _wrap(): return x
    return _wrap()

class PromptHistoryRetriever(BaseRetriever):
    """
    Pull short evidence snippets from local history stores (CaseBooks/Chats/Traces).
    Tries a few common repos; degrades gracefully if unavailable.
    """
    def __init__(self, memory: Any, max_chars: int = 400):
        self.memory = memory
        self.max_chars = int(max_chars)

    async def retrieve(self, query: str, seed_answer: str, context: Dict[str, Any], k: int) -> List[str]:
        repos = [
            ("casebooks", "search"),   # preferred
            ("chat_repo", "search"),
            ("traces", "search"),
            ("docs", "search"),
        ]
        out: List[str] = []
        for name, meth in repos:
            repo = getattr(self.memory, name, None)
            if not repo: 
                continue
            fn = getattr(repo, meth, None)
            if not fn:
                continue
            try:
                # try async first (await) then sync
                hits = await _maybe_sync(fn(query, top_k=k))
                for h in hits or []:
                    # tolerate different shapes
                    txt = (h.get("assistant_text")
                           or h.get("text")
                           or h.get("content")
                           or str(h))[: self.max_chars]
                    txt = (txt or "").strip()
                    if txt:
                        out.append(txt)
                        if len(out) >= k:
                            return out
            except Exception:
                continue
        return out[:k]
