from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.exc import IntegrityError

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.orm.source import (SourceCandidateORM, SourceORM,
                                  SourceQualityORM)


def _sha256_hex(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


class SourceStore(BaseSQLAlchemyStore):
    orm_model = SourceORM
    default_order_by = SourceORM.created_at.desc()

    def __init__(self, session_or_maker, logger=None):
        super().__init__(session_or_maker, logger)
        self.name = "sources"

    # -------------------------
    # Sources
    # -------------------------

    def get_or_create_source(
        self,
        *,
        source_type: str,
        source_uri: str,
        canonical_uri: Optional[str] = None,
        title: Optional[str] = None,
        snippet: Optional[str] = None,
        mime_type: Optional[str] = None,
        content_hash: Optional[str] = None,
        verification: Optional[str] = None,
        trust_score: Optional[float] = None,
        quality_score: Optional[float] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> int:
        meta = meta or {}

        def op(s):
            existing = (
                s.query(SourceORM)
                .filter_by(source_type=source_type, source_uri=source_uri)
                .first()
            )
            if existing:
                # best-effort patch
                patch = False
                if canonical_uri and not existing.canonical_uri:
                    existing.canonical_uri = canonical_uri
                    patch = True
                if title and not existing.title:
                    existing.title = title
                    patch = True
                if snippet and not existing.snippet:
                    existing.snippet = snippet
                    patch = True
                if mime_type and not existing.mime_type:
                    existing.mime_type = mime_type
                    patch = True
                if content_hash and not existing.content_hash:
                    existing.content_hash = content_hash
                    patch = True
                if verification and not existing.verification:
                    existing.verification = verification
                    patch = True
                if trust_score is not None and existing.trust_score is None:
                    existing.trust_score = float(trust_score)
                    patch = True
                if quality_score is not None and existing.quality_score is None:
                    existing.quality_score = float(quality_score)
                    patch = True
                if meta:
                    merged = dict(existing.meta or {})
                    merged.update(meta)
                    existing.meta = merged
                    patch = True
                if patch:
                    s.add(existing)
                return int(existing.id)

            obj = SourceORM(
                source_type=source_type,
                source_uri=source_uri,
                canonical_uri=canonical_uri,
                title=title,
                snippet=snippet,
                mime_type=mime_type,
                content_hash=content_hash,
                verification=verification,
                trust_score=trust_score,
                quality_score=quality_score,
                meta=meta,
            )
            s.add(obj)
            s.flush()
            return int(obj.id)

        try:
            return self._run(op)
        except IntegrityError:
            # race: read back
            def op_retry(s):
                row = (
                    s.query(SourceORM)
                    .filter_by(source_type=source_type, source_uri=source_uri)
                    .first()
                )
                return int(row.id) if row else None

            out = self._run(op_retry)
            if out is None:
                raise
            return out

    # -------------------------
    # Candidates (search hits)
    # -------------------------

    def add_candidate(
        self,
        *,
        pipeline_run_id: int,
        query_text: str,
        source_id: int,
        provider: Optional[str] = None,
        result_type: Optional[str] = None,
        rank: Optional[int] = None,
        score: Optional[float] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> int:
        meta = meta or {}
        qhash = _sha256_hex(query_text)[:16]

        def op(s):
            existing = (
                s.query(SourceCandidateORM)
                .filter_by(
                    pipeline_run_id=int(pipeline_run_id),
                    query_hash=qhash,
                    source_id=int(source_id),
                )
                .first()
            )
            if existing:
                # patch rank/score/meta if better
                patch = False
                if rank is not None and existing.rank is None:
                    existing.rank = int(rank)
                    patch = True
                if score is not None and existing.score is None:
                    existing.score = float(score)
                    patch = True
                if provider and not existing.provider:
                    existing.provider = provider
                    patch = True
                if result_type and not existing.result_type:
                    existing.result_type = result_type
                    patch = True
                if meta:
                    merged = dict(existing.meta or {})
                    merged.update(meta)
                    existing.meta = merged
                    patch = True
                if patch:
                    s.add(existing)
                return int(existing.id)

            obj = SourceCandidateORM(
                pipeline_run_id=int(pipeline_run_id),
                query_text=query_text,
                query_hash=qhash,
                source_id=int(source_id),
                provider=provider,
                result_type=result_type,
                rank=rank,
                score=score,
                meta=meta,
            )
            s.add(obj)
            s.flush()
            return int(obj.id)

        try:
            return self._run(op)
        except IntegrityError:
            def op_retry(s):
                row = (
                    s.query(SourceCandidateORM)
                    .filter_by(
                        pipeline_run_id=int(pipeline_run_id),
                        query_hash=qhash,
                        source_id=int(source_id),
                    )
                    .first()
                )
                return int(row.id) if row else None
            out = self._run(op_retry)
            if out is None:
                raise
            return out

    # -------------------------
    # Quality (goal-conditioned)
    # -------------------------

    def upsert_quality(
        self,
        *,
        pipeline_run_id: Optional[int],
        goal_type: str,
        source_id: int,
        trust_score: Optional[float],
        quality_score: Optional[float],
        verification: Optional[str],
        method: str = "heuristic",
        rationale: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> int:
        meta = meta or {}

        def op(s):
            q = (
                s.query(SourceQualityORM)
                .filter_by(
                    pipeline_run_id=pipeline_run_id,
                    goal_type=goal_type,
                    source_id=int(source_id),
                    method=method,
                )
                .first()
            )
            if q:
                q.trust_score = trust_score if trust_score is None else float(trust_score)
                q.quality_score = quality_score if quality_score is None else float(quality_score)
                q.verification = verification
                q.rationale = rationale
                if meta:
                    merged = dict(q.meta or {})
                    merged.update(meta)
                    q.meta = merged
                s.add(q)
                s.flush()
                return int(q.id)

            obj = SourceQualityORM(
                pipeline_run_id=pipeline_run_id,
                goal_type=goal_type,
                source_id=int(source_id),
                trust_score=trust_score,
                quality_score=quality_score,
                verification=verification,
                method=method,
                rationale=rationale,
                meta=meta,
            )
            s.add(obj)
            s.flush()
            return int(obj.id)

        try:
            return self._run(op)
        except IntegrityError:
            # read back
            def op_retry(s):
                row = (
                    s.query(SourceQualityORM)
                    .filter_by(
                        pipeline_run_id=pipeline_run_id,
                        goal_type=goal_type,
                        source_id=int(source_id),
                        method=method,
                    )
                    .first()
                )
                return int(row.id) if row else None
            out = self._run(op_retry)
            if out is None:
                raise
            return out
