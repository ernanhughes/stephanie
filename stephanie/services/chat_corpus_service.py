# stephanie/services/chat_corpus_service.py
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

from stephanie.services.service_protocol import Service

EmbedFn = Callable[[List[str]], List[List[float]]]

import logging

_logger = logging.getLogger(__name__)

class ChatCorpusService(Service):
    """
    Facade for chat threads/messages + optional embeddings + search.
    """
    def __init__(self, cfg, memory, container, logger):
        self.cfg = cfg
        self.memory = memory
        self.container = container
        self.logger = logger

    def initialize(self, **kwargs) -> None:
        self._ok = True
        _logger.info("ChatCorpusServiceInit", {})

    def shutdown(self) -> None:
        self._ok = False

    @property
    def name(self) -> str:
        return "chat-corpus-service-v1"

    def health_check(self) -> Dict[str, Any]:
        return {"status": "healthy" if self._ok else "uninitialized"}

    # --- API ---
    def ensure_thread(self, *, casebook_name: str, topic: str | None = None, paper_id: str | None = None, post_slug: str | None = None, tags: Optional[List[str]] = None, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        t = self.store.ensure_thread(casebook_name=casebook_name, topic=topic, paper_id=paper_id, post_slug=post_slug, tags=tags, meta=meta)
        return t.to_dict(include_messages=False)

    def append(self, *, thread_id: int, role: str, text: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        emb = None
        if self.embedder:
            try:
                emb = self.embedder([text])[0]
            except Exception:
                emb = None
        m = self.store.add_message(thread_id=thread_id, role=role, text=text, meta=meta, embedding=emb)
        return m.to_dict()

    def snapshot_for_casebook(self, *, casebook_name: str, limit: int = 300, include_summaries: bool = False) -> List[Dict[str, Any]]:
        msgs = self.store.snapshot_for_casebook(casebook_name, limit)
        # optional: add a tiny head/tail summary block
        if include_summaries and msgs:
            head = " ".join(m["text"] for m in msgs[:5])
            tail = " ".join(m["text"] for m in msgs[-5:])
            msgs = [{"role":"system","text":f"Conversation summary (head): {head[:800]}", "meta":{"synthetic":True}}] + msgs + [{"role":"system","text":f"Conversation summary (tail): {tail[:800]}", "meta":{"synthetic":True}}]
        return msgs

    def search(self, *, casebook_name: str, query: str, k: int = 20) -> List[Dict[str, Any]]:
        return self.store.search(casebook_name, query, k)
