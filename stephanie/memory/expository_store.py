# stephanie/memory/expository_store.py
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from sqlalchemy import asc, desc, func
from sqlalchemy.orm import Query

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.models.expository import (BlogDraft, ExpositoryBuffer,
                                         ExpositorySnippet,
                                         PaperSourceQueueORM)


# -----------------------------
# ExpositorySnippetStore
# -----------------------------
class ExpositorySnippetStore(BaseSQLAlchemyStore):
    """
    Store for expository snippets extracted from papers.
    Provides add/bulk_add, dedup, and common retrievals.
    """
    orm_model = ExpositorySnippet
    default_order_by = desc(ExpositorySnippet.created_at)

    def __init__(self, session_or_maker, logger=None):
        super().__init__(session_or_maker, logger)
        self._name = "expository_snippets"

    def name(self) -> str:
        return self._name

    # -------- Create / Update --------
    def add(
        self,
        *,
        doc_id: int,
        section: str,
        order_idx: int,
        text: str,
        features: Dict[str, Any],
        expository_score: float,
        bloggability_score: float,
        picked: bool = False,
    ) -> ExpositorySnippet:
        def op(s):
            row = ExpositorySnippet(
                doc_id=doc_id,
                section=section,
                order_idx=order_idx,
                text=text,
                features=features or {},
                expository_score=float(expository_score),
                bloggability_score=float(bloggability_score),
                picked=bool(picked),
            )
            s.add(row)
            s.flush()
            if self.logger:
                self.logger.log("ExpositorySnippetInserted", {"id": row.id, "doc_id": doc_id, "section": section})
            return row
        return self._run(op)

    def bulk_add(self, items: Iterable[Dict[str, Any]], dedup_on_text: bool = True) -> List[int]:
        """
        items keys: doc_id, section, order_idx, text, features, expository_score, bloggability_score, picked?
        Dedup: (doc_id, section, text) exact match.
        """
        ids: List[int] = []

        def op(s):
            nonlocal ids
            rows: List[ExpositorySnippet] = []
            for it in items:
                doc_id = int(it["doc_id"])
                section = (it.get("section") or "").strip()
                text = (it.get("text") or "").strip()
                if not text:
                    continue

                if dedup_on_text:
                    exists = (
                        s.query(ExpositorySnippet.id)
                        .filter(
                            ExpositorySnippet.doc_id == doc_id,
                            ExpositorySnippet.section == section,
                            ExpositorySnippet.text == text,
                        )
                        .first()
                    )
                    if exists:
                        continue

                rows.append(
                    ExpositorySnippet(
                        doc_id=doc_id,
                        section=section,
                        order_idx=int(it.get("order_idx") or 0),
                        text=text,
                        features=it.get("features") or {},
                        expository_score=float(it.get("expository_score") or 0.0),
                        bloggability_score=float(it.get("bloggability_score") or 0.0),
                        picked=bool(it.get("picked") or False),
                    )
                )
            if not rows:
                ids = []
                return []
            s.add_all(rows)
            s.flush()
            ids = [r.id for r in rows]
            if self.logger:
                self.logger.log("ExpositorySnippetsBulkInserted", {"count": len(ids)})
            return ids
        return self._run(op)

    def mark_picked(self, snippet_id: int, picked: bool = True) -> Optional[ExpositorySnippet]:
        def op(s):
            row = s.get(ExpositorySnippet, snippet_id)
            if not row:
                return None
            row.picked = bool(picked)
            s.add(row); s.flush()
            if self.logger:
                self.logger.log("ExpositorySnippetPicked", {"id": row.id, "picked": row.picked})
            return row
        return self._run(op)

    # -------- Retrieval --------
    def get(self, snippet_id: int) -> Optional[ExpositorySnippet]:
        def op(s): return s.get(ExpositorySnippet, snippet_id)
        return self._run(op)

    def list_by_doc(self, doc_id: int, limit: int = 500) -> List[ExpositorySnippet]:
        def op(s):
            return (
                s.query(ExpositorySnippet)
                .filter(ExpositorySnippet.doc_id == doc_id)
                .order_by(asc(ExpositorySnippet.section), asc(ExpositorySnippet.order_idx))
                .limit(int(limit)).all()
            )
        return self._run(op)

    def list_top(
        self,
        *,
        min_exp: float = 0.45,
        min_blog: float = 0.50,
        section_allow: Optional[List[str]] = None,
        limit: int = 50,
    ) -> List[ExpositorySnippet]:
        def op(s):
            q: Query = s.query(ExpositorySnippet).filter(
                ExpositorySnippet.expository_score >= float(min_exp),
                ExpositorySnippet.bloggability_score >= float(min_blog),
            )
            if section_allow:
                q = q.filter(ExpositorySnippet.section.in_(section_allow))
            return q.order_by(
                desc(ExpositorySnippet.expository_score),
                desc(ExpositorySnippet.bloggability_score),
            ).limit(int(limit)).all()
        return self._run(op)

    def stats_summary(self) -> Dict[str, Any]:
        def op(s):
            total = s.query(func.count(ExpositorySnippet.id)).scalar() or 0
            avg_exp = s.query(func.avg(ExpositorySnippet.expository_score)).scalar()
            avg_blog = s.query(func.avg(ExpositorySnippet.bloggability_score)).scalar()
            return {"total": int(total), "avg_expository": float(avg_exp or 0.0), "avg_bloggability": float(avg_blog or 0.0)}
        return self._run(op)


# -----------------------------
# ExpositoryBufferStore
# -----------------------------
class ExpositoryBufferStore(BaseSQLAlchemyStore):
    """
    Store for buffers (a selected set of snippet ids for a topic).
    Includes helper to create a buffer from top-k snippets directly.
    """
    orm_model = ExpositoryBuffer
    default_order_by = desc(ExpositoryBuffer.created_at)

    def __init__(self, session_or_maker, logger=None):
        super().__init__(session_or_maker, logger)
        self._name = "expository_buffers"

    def name(self) -> str:
        return self._name

    def create(self, *, topic: str, snippet_ids: List[int], meta: Optional[Dict[str, Any]] = None) -> ExpositoryBuffer:
        def op(s):
            row = ExpositoryBuffer(
                topic=(topic or "general").strip(),
                snippet_ids=list(snippet_ids or []),
                meta=meta or {},
                created_at=datetime.now(),
            )
            s.add(row); s.flush()
            if self.logger:
                self.logger.log("ExpositoryBufferCreated", {"id": row.id, "topic": row.topic, "n_snippets": len(row.snippet_ids)})
            return row
        return self._run(op)

    def create_from_topk(
        self,
        *,
        topic: str,
        snippet_store: ExpositorySnippetStore,
        k: int = 12,
        min_exp: float = 0.45,
        min_blog: float = 0.50,
        section_allow: Optional[List[str]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> ExpositoryBuffer:
        top = snippet_store.list_top(min_exp=min_exp, min_blog=min_blog, section_allow=section_allow, limit=k)
        return self.create(topic=topic, snippet_ids=[t.id for t in top], meta=meta)

    def get(self, buffer_id: int) -> Optional[ExpositoryBuffer]:
        def op(s): return s.get(ExpositoryBuffer, buffer_id)
        return self._run(op)

    def list_recent(self, *, limit: int = 50) -> List[ExpositoryBuffer]:
        def op(s):
            return (
                s.query(ExpositoryBuffer)
                .order_by(self.default_order_by)
                .limit(int(limit)).all()
            )
        return self._run(op)


# -----------------------------
# BlogDraftStore
# -----------------------------
class BlogDraftStore(BaseSQLAlchemyStore):
    """
    Store for drafts assembled from expository buffers.
    Provides create/update quality, list_kept, and quick filters.
    """
    orm_model = BlogDraft
    default_order_by = desc(BlogDraft.created_at)

    def __init__(self, session_or_maker, logger=None):
        super().__init__(session_or_maker, logger)
        self._name = "blog_drafts"

    def name(self) -> str:
        return self._name

    # -------- Create / Update --------
    def create(
        self,
        *,
        topic: str,
        source_snippet_ids: List[int],
        draft_md: str,
        readability: float = 0.0,
        local_coherence: float = 0.0,
        arena_passes: int = 0,
        repetition_penalty: float = 0.0,
        kept: bool = False,
    ) -> BlogDraft:
        def op(s):
            row = BlogDraft(
                topic=(topic or "general").strip(),
                source_snippet_ids=list(source_snippet_ids or []),
                draft_md=draft_md or "",
                readability=float(readability),
                local_coherence=float(local_coherence),
                arena_passes=int(arena_passes or 0),
                repetition_penalty=float(repetition_penalty or 0.0),
                kept=bool(kept),
                created_at=datetime.now(),
            )
            s.add(row); s.flush()
            if self.logger:
                self.logger.log("BlogDraftCreated", {"id": row.id, "topic": row.topic, "kept": row.kept})
            return row
        return self._run(op)

    def update_quality(
        self,
        draft_id: int,
        *,
        readability: Optional[float] = None,
        local_coherence: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        kept: Optional[bool] = None,
        arena_passes_inc: int = 0,
    ) -> Optional[BlogDraft]:
        def op(s):
            row = s.get(BlogDraft, draft_id)
            if not row:
                return None
            if readability is not None:
                row.readability = float(readability)
            if local_coherence is not None:
                row.local_coherence = float(local_coherence)
            if repetition_penalty is not None:
                row.repetition_penalty = float(repetition_penalty)
            if kept is not None:
                row.kept = bool(kept)
            if arena_passes_inc:
                row.arena_passes = int(row.arena_passes or 0) + int(arena_passes_inc)
            s.add(row); s.flush()
            if self.logger:
                self.logger.log("BlogDraftUpdated", {"id": row.id, "kept": row.kept})
            return row
        return self._run(op)

    # -------- Retrieval --------
    def get(self, draft_id: int) -> Optional[BlogDraft]:
        def op(s): return s.get(BlogDraft, draft_id)
        return self._run(op)

    def list_recent(self, *, topic: Optional[str] = None, limit: int = 50) -> List[BlogDraft]:
        def op(s):
            q: Query = s.query(BlogDraft)
            if topic:
                q = q.filter(BlogDraft.topic == topic)
            return q.order_by(self.default_order_by).limit(int(limit)).all()
        return self._run(op)

    def list_kept(self, *, topic: Optional[str] = None, limit: int = 100) -> List[BlogDraft]:
        def op(s):
            q: Query = s.query(BlogDraft).filter(BlogDraft.kept == True)  # noqa: E712
            if topic:
                q = q.filter(BlogDraft.topic == topic)
            return q.order_by(self.default_order_by).limit(int(limit)).all()
        return self._run(op)

    def delete_draft(self, draft_id: int) -> bool:
        def op(s):
            row = s.get(BlogDraft, draft_id)
            if not row: return False
            s.delete(row); return True
        return self._run(op)

    def rank_and_create_buffer(
        self,
        *,
        topic: str,
        k: int = 12,
        min_exp: float = 0.45,
        min_blog: float = 0.50,
        section_allow: Optional[List[str]] = None,
    ) -> ExpositoryBuffer:
        """
        Rank by (expository_score desc, bloggability_score desc) and create a buffer of top-k snippet ids.
        This folds the agent's DB work into the store.
        """
        from sqlalchemy import desc  # local import to avoid top-level clutter
        def op(s):
            q = s.query(ExpositorySnippet).filter(
                ExpositorySnippet.expository_score >= float(min_exp),
                ExpositorySnippet.bloggability_score >= float(min_blog),
            )
            if section_allow:
                q = q.filter(ExpositorySnippet.section.in_(section_allow))
            rows = (
                q.order_by(
                    desc(ExpositorySnippet.expository_score),
                    desc(ExpositorySnippet.bloggability_score),
                )
                .limit(int(k))
                .all()
            )
            buf = ExpositoryBuffer(
                topic=(topic or "general").strip(),
                snippet_ids=[r.id for r in rows],
            )
            s.add(buf); s.flush()
            if self.logger:
                self.logger.log("ExpositoryBufferRankedCreated", {
                    "id": buf.id, "topic": buf.topic, "k": len(buf.snippet_ids)
                })
            return buf
        return self._run(op)


    def evaluate_and_mark(
        self,
        draft_id: int,
        *,
        min_readability: float,
        min_adjacent_coherence: float,
        keep_threshold: float = 0.6,
        repetition_penalty_weight: float = 0.1,
    ) -> Optional[float]:
        """
        Computes r_solve from doc-native metrics and updates 'kept'.
        r_solve = clamp01(0.5*ok_read + 0.5*ok_coh - w*rep_pen)
        Returns r_solve (or None if draft not found).
        """
        def clamp01(x: float) -> float:
            return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

        def op(s):
            d = s.get(BlogDraft, draft_id)
            if not d:
                return None
            ok_read = 1.0 if (d.readability or 0.0) >= float(min_readability) else 0.0
            ok_coh  = 1.0 if (d.local_coherence or 0.0) >= float(min_adjacent_coherence) else 0.0
            rep_pen = float(d.repetition_penalty or 0.0)
            r_solve = clamp01(0.5*ok_read + 0.5*ok_coh - float(repetition_penalty_weight)*rep_pen)
            d.kept = bool(r_solve >= float(keep_threshold))
            s.add(d); s.flush()
            if self.logger:
                self.logger.log("BlogDraftEvaluated", {
                    "id": d.id, "r_solve": r_solve, "kept": d.kept
                })
            return r_solve
        return self._run(op)


    def enqueue(self, *, url: str, topic: str, source: str = "manual", meta: Optional[Dict]=None) -> Optional[PaperSourceQueueORM]:
        url = (url or "").strip()
        if not url: return None
        def op(s):
            # soft dedup by url
            existing = s.query(PaperSourceQueueORM).filter(PaperSourceQueueORM.url == url).first()
            if existing: return existing
            row = PaperSourceQueueORM(topic=topic, url=url, source=source, meta=meta or {})
            s.add(row); s.flush()
            if self.logger: self.logger.log("PaperEnqueued", {"id": row.id, "topic": topic, "url": url, "source": source})
            return row
        return self._run(op)

    def bulk_enqueue(self, *, topic: str, items: List[Dict[str, Any]]) -> int:
        count = 0
        for it in items:
            if self.enqueue(url=it["url"], topic=topic, source=it.get("source","hf"), meta=it): count += 1
        return count

    def claim_pending(self, *, topic: Optional[str] = None, limit: int = 10) -> List[PaperSourceQueueORM]:
        def op(s):
            q = s.query(PaperSourceQueueORM).filter(PaperSourceQueueORM.status == "pending")
            if topic: q = q.filter(PaperSourceQueueORM.topic == topic)
            rows = q.order_by(PaperSourceQueueORM.created_at.asc()).limit(int(limit)).all()
            for r in rows:
                r.status = "fetched"  # move to fetched to avoid double work; adjust if you want separate fetch step
                r.updated_at = datetime.now()
            s.flush()
            return rows
        return self._run(op)

    def mark(self, queue_id: int, status: str, meta_update: Optional[Dict]=None) -> None:
        def op(s):
            r = s.get(PaperSourceQueueORM, queue_id)
            if not r: return
            r.status = status
            if meta_update:
                r.meta = {**(r.meta or {}), **meta_update}
            r.updated_at = datetime.now()
            s.flush()
        return self._run(op)

    def recent(self, *, topic: Optional[str]=None, limit: int=50) -> List[PaperSourceQueueORM]:
        def op(s):
            q = s.query(PaperSourceQueueORM)
            if topic: q = q.filter(PaperSourceQueueORM.topic == topic)
            return q.order_by(self.default_order_by).limit(int(limit)).all()
        return self._run(op)
