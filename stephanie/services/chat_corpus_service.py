# stephanie/services/chat_corpus_service.py
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

# 🔑 Embedding search target enum (adjust import if your path differs)
from stephanie.scoring.scorable import \
    ScorableType  # expects CONVERSATION_TURN
from stephanie.services.service_protocol import Service
from stephanie.utils.time_utils import now_ms

EmbedFn = Callable[[List[str]], List[List[float]]]
log = logging.getLogger(__name__)



class ChatCorpusService(Service):
    """
    Facade for chat threads/messages + optional embeddings + search.

    New:
      - build_corpus_for_document(doc=...) → {near, domain, embedding, all}
      - Uses memory.embedding.search_related_scorables(..., TargetType.CONVERSATION_TURN)
      - Graceful fallbacks if embedding/store are missing
    """

    def __init__(self, cfg, memory, container, logger):
        self.cfg = cfg
        self.memory = memory
        self.container = container
        self.logger = logger
        self._ok = False

        self.store = container.get_service("chat_store")

    # --- lifecycle ---------------------------------------------------------
    def initialize(self, **kwargs) -> None:
        self._ok = True
        log.debug("ChatCorpusServiceInit ts:  " + str(now_ms()))

    def shutdown(self) -> None:
        self._ok = False

    @property
    def name(self) -> str:
        return "chat-corpus-service-v1"

    def health_check(self) -> Dict[str, Any]:
        return {"status": "healthy" if self._ok else "uninitialized"}

    # --- basic thread ops ---------------------------------------------------
    def ensure_thread(
        self,
        *,
        casebook_name: str,
        topic: str | None = None,
        paper_id: str | None = None,
        post_slug: str | None = None,
        tags: Optional[List[str]] = None,
        meta: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if not self.store:
            raise RuntimeError("chat_corpus.store not available")
        t = self.store.ensure_thread(
            casebook_name=casebook_name,
            topic=topic,
            paper_id=paper_id,
            post_slug=post_slug,
            tags=tags,
            meta=meta,
        )
        return t.to_dict(include_messages=False)

    def append(
        self,
        *,
        thread_id: int,
        role: str,
        text: str,
        meta: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if not self.store:
            raise RuntimeError("chat_corpus.store not available")
        emb = self.memory.embedding.get_or_create(text)
        m = self.store.add_message(thread_id=thread_id, role=role, text=text, meta=meta, embedding=emb)
        return m.to_dict()

    def snapshot_for_casebook(
        self, *, casebook_name: str, limit: int = 300, include_summaries: bool = False
    ) -> List[Dict[str, Any]]:
        if not self.store:
            return []
        msgs = self.store.snapshot_for_casebook(casebook_name, limit)
        if include_summaries and msgs:
            head = " ".join(m["text"] for m in msgs[:5])
            tail = " ".join(m["text"] for m in msgs[-5:])
            msgs = (
                [{"role": "system", "text": f"Conversation summary (head): {head[:800]}", "meta": {"synthetic": True}}]
                + msgs
                + [{"role": "system", "text": f"Conversation summary (tail): {tail[:800]}", "meta": {"synthetic": True}}]
            )
        return msgs

    # --- simple lexical search passthrough (optional) ----------------------
    def search(self, *, casebook_name: str, query: str, k: int = 20) -> List[Dict[str, Any]]:
        if not self.store:
            return []
        return self.store.search(casebook_name, query, k)

    # --- NEW: build a task-specific chat corpus for a paper/section --------
    def build_corpus_for_document(
        self,
        *,
        doc: Dict[str, Any],
        k_near: int = 40,
        k_domain: int = 30,
        k_embed: int = 60,
        include_text: bool = True,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Given a document (paper or section dict), find related chats using:
          1) NEAR: messages near the same casebook/thread or sharing paper_id
          2) DOMAIN: entity/domain-aware embedding search (include_ner=True)
          3) EMBEDDING: pure semantic matches (include_ner=False)

        Returns:
          {
            "near":      [messages...],
            "domain":    [messages...],
            "embedding": [messages...],
            "all":       [deduped union...]
          }

        Expected doc keys (best effort):
          - id or doc_id (str)
          - title (str), abstract (str)
          - section_name (str) and section_text (str)   # optional
          - casebook_name (str)                          # optional
        """
        text = self._doc_text(doc)
        if not text.strip():
            return {"near": [], "domain": [], "embedding": [], "all": []}

        near = self._near_messages_for_doc(doc, limit=k_near)
        domain = self._embedding_related(text, include_ner=True, k=k_domain)
        embed = self._embedding_related(text, include_ner=False, k=k_embed)

        # Normalize & dedupe (preserve best score per id)
        near_n = [self._normalize_msg(m, source="near") for m in near]
        dom_n  = [self._normalize_msg(m, source="domain") for m in domain]
        emb_n  = [self._normalize_msg(m, source="embedding") for m in embed]

        all_union = self._dedupe_by_id(near_n + dom_n + emb_n)

        # Optionally strip text to be light-weight
        if not include_text:
            for arr in (near_n, dom_n, emb_n, all_union):
                for m in arr:
                    m.pop("text", None)

        return {"near": near_n, "domain": dom_n, "embedding": emb_n, "all": all_union}

    # --- internals ---------------------------------------------------------
    def _doc_text(self, doc: Dict[str, Any]) -> str:
        # Prefer a section if present; else title + abstract
        sec_text = (doc.get("section_text") or "").strip()
        if sec_text:
            return sec_text
        title = (doc.get("title") or "").strip()
        abstract = (doc.get("abstract") or "").strip()
        if title or abstract:
            return f"{title}\n\n{abstract}"
        # Final fallback: any 'text' field
        return (doc.get("text") or "").strip()

    def _near_messages_for_doc(self, doc: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        """
        Heuristics for "near":
          - if doc.casebook_name → snapshot recent messages from that casebook
          - else, if doc.id/doc_id → ask store for recent messages tagged with that paper_id
          - else → empty
        """
        if not self.store:
            return []
        casebook_name = doc.get("casebook_name")
        if casebook_name:
            try:
                return self.store.snapshot_for_casebook(casebook_name, limit)
            except Exception:
                pass

        paper_id = doc.get("id") or doc.get("doc_id")
        if paper_id:
            try:
                return self.store.recent_messages_for_paper(paper_id=paper_id, limit=limit)
            except Exception:
                # not all stores implement this; fall back empty
                return []
        return []

    def _embedding_related(self, text: str, *, include_ner: bool, k: int) -> List[Dict[str, Any]]:
        """
        Use the embeddings index to retrieve related conversation turns.
        """
        try:
            emb = self.memory.embedding
        except Exception:
            emb = None

        if not emb:
            log.warning("Embedding index unavailable; embedding search skipped")
            return []

        try:
            # This is the call you specified:
            candidates = emb.search_related_scorables(
                text, target_type=ScorableType.CONVERSATION_TURN, include_ner=bool(include_ner), top_k=int(k)
            )
            # Expect a list of scorables; each must be turned into a message dict
            out: List[Dict[str, Any]] = []
            for c in candidates:
                # Typical scorable shape: id, case_id, role, text, meta, score, etc.
                out.append({
                    "id": c.get("id") or c.get("scorable_id"),
                    "thread_id": (c.get("meta") or {}).get("thread_id"),
                    "case_id": c.get("case_id"),
                    "role": (c.get("meta") or {}).get("role") or "assistant",
                    "text": c.get("text") or "",
                    "score": float(c.get("score") or 0.0),
                    "meta": c.get("meta") or {},
                })
            return out
        except Exception as e:
            log.warning(f"Embedding search failed: {e}")
            return []

    def _normalize_msg(self, m: Dict[str, Any], *, source: str) -> Dict[str, Any]:
        """
        Normalize heterogeneous message dicts (store snapshots vs embedding returns).
        """
        return {
            "id": m.get("id") or m.get("message_id") or m.get("scorable_id"),
            "thread_id": m.get("thread_id"),
            "case_id": m.get("case_id"),
            "role": m.get("role") or (m.get("meta") or {}).get("role") or "assistant",
            "text": m.get("text") or "",
            "score": float(m.get("score") or 0.0),
            "meta": m.get("meta") or {},
            "source": source,
        }

    def _dedupe_by_id(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Keep the highest-scoring entry per id, preserve overall ranking by max score.
        """
        best: Dict[Any, Dict[str, Any]] = {}
        for it in items:
            _id = it.get("id")
            if _id is None:
                continue
            prev = best.get(_id)
            if (prev is None) or (it.get("score", 0) > prev.get("score", 0)):
                best[_id] = it
        # sort by score desc, then by source preference (near > domain > embedding)
        source_rank = {"near": 0, "domain": 1, "embedding": 2}
        return sorted(
            best.values(),
            key=lambda x: (-float(x.get("score", 0.0)), source_rank.get(x.get("source"), 9))
        )
