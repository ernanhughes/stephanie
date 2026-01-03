from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.exc import IntegrityError

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.orm.source import SourceORM, ScorableSourceLinkORM, SourceCandidateORM, SourceQualityORM

from stephanie.utils.hash_utils import hash_text

class SourceStore(BaseSQLAlchemyStore):
    """
    One-stop provenance store:
      - get/create SourceORM (source_type, locator)
      - link scorable_type+scorable_id -> source_id with a role

    Constraints expected:
      - sources: UNIQUE(source_type, locator)
      - scorable_sources: UNIQUE(scorable_type, scorable_id, source_id, role)
    """

    # Base store wants a primary orm_model. We'll treat links as primary.
    orm_model = ScorableSourceLinkORM
    default_order_by = ScorableSourceLinkORM.created_at.asc()

    def __init__(self, session_or_maker, logger=None):
        super().__init__(session_or_maker, logger)
        self.name = "sources"

    # -------------------------
    # Internal helpers (same session)
    # -------------------------

    def _get_source(self, s, *, source_type: str, locator: str) -> Optional[SourceORM]:
        return (
            s.query(SourceORM)
            .filter_by(source_type=source_type, locator=locator)
            .first()
        )

    def _get_link(
        self,
        s,
        *,
        scorable_type: str,
        scorable_id: str,
        source_id: int,
        role: str,
    ) -> Optional[ScorableSourceLinkORM]:
        return (
            s.query(ScorableSourceLinkORM)
            .filter_by(
                scorable_type=scorable_type,
                scorable_id=str(scorable_id),
                source_id=source_id,
                role=role,
            )
            .first()
        )

    # -------------------------
    # Public API
    # -------------------------

    def get_or_create_source_id(
        self,
        *,
        source_type: str,
        locator: str,
        canonical_locator: Optional[str] = None,
        content_hash: Optional[str] = None,
        mime_type: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> int:
        meta = meta or {}

        def op(s):
            existing = self._get_source(s, source_type=source_type, locator=locator)
            if existing:
                return existing.id

            obj = SourceORM(
                source_type=source_type,
                locator=locator,
                canonical_locator=canonical_locator,
                content_hash=content_hash,
                mime_type=mime_type,
                name=name,
                description=description,
                meta=meta,
            )
            s.add(obj)
            s.flush()
            return obj.id

        try:
            return self._run(op)
        except IntegrityError:
            # race: someone inserted first
            def op_retry(s):
                ex = self._get_source(s, source_type=source_type, locator=locator)
                return ex.id if ex else None

            out = self._run(op_retry)
            if out is None:
                raise
            return out

    def link_source_id(
        self,
        *,
        scorable_type: str,
        scorable_id: str,
        source_id: int,
        role: str = "origin",
        confidence: Optional[float] = None,
        pipeline_run_id: Optional[int] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> int:
        meta = meta or {}

        def op(s):
            existing = self._get_link(
                s,
                scorable_type=scorable_type,
                scorable_id=str(scorable_id),
                source_id=source_id,
                role=role,
            )
            if existing:
                return existing.id

            obj = ScorableSourceLinkORM(
                scorable_type=scorable_type,
                scorable_id=str(scorable_id),
                source_id=source_id,
                role=role,
                confidence=confidence,
                pipeline_run_id=pipeline_run_id,
                meta=meta,
            )
            s.add(obj)
            s.flush()
            return obj.id

        try:
            return self._run(op)
        except IntegrityError:
            def op_retry(s):
                ex = self._get_link(
                    s,
                    scorable_type=scorable_type,
                    scorable_id=str(scorable_id),
                    source_id=source_id,
                    role=role,
                )
                return ex.id if ex else None

            out = self._run(op_retry)
            if out is None:
                raise
            return out

    def get_or_create_link(
        self,
        *,
        scorable_type: str,
        scorable_id: str,
        source_type: str,
        locator: str,
        role: str = "origin",
        pipeline_run_id: Optional[int] = None,
        source_meta: Optional[Dict[str, Any]] = None,
        link_meta: Optional[Dict[str, Any]] = None,
        **source_kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Convenience: create/get source + create/get link in one transaction.

        Returns:
          {"source_id": ..., "link_id": ...}
        """
        source_meta = source_meta or {}
        link_meta = link_meta or {}

        def op(s):
            src = self._get_source(s, source_type=source_type, locator=locator)
            if not src:
                src = SourceORM(
                    source_type=source_type,
                    locator=locator,
                    meta=source_meta,
                    **source_kwargs,
                )
                s.add(src)
                s.flush()

            link = self._get_link(
                s,
                scorable_type=scorable_type,
                scorable_id=str(scorable_id),
                source_id=src.id,
                role=role,
            )
            if not link:
                link = ScorableSourceLinkORM(
                    scorable_type=scorable_type,
                    scorable_id=str(scorable_id),
                    source_id=src.id,
                    role=role,
                    pipeline_run_id=pipeline_run_id,
                    meta=link_meta,
                )
                s.add(link)
                s.flush()

            return {"source_id": src.id, "link_id": link.id}

        try:
            return self._run(op)
        except IntegrityError:
            # race safety
            def op_retry(s):
                src = self._get_source(s, source_type=source_type, locator=locator)
                if not src:
                    return None
                link = self._get_link(
                    s,
                    scorable_type=scorable_type,
                    scorable_id=str(scorable_id),
                    source_id=src.id,
                    role=role,
                )
                return {"source_id": src.id, "link_id": (link.id if link else None)}

            out = self._run(op_retry)
            if not out or out.get("link_id") is None:
                raise
            return out

    def list_links_for_scorable(self, *, scorable_type: str, scorable_id: str) -> List[ScorableSourceLinkORM]:
        def op(s):
            return (
                s.query(ScorableSourceLinkORM)
                .filter_by(scorable_type=scorable_type, scorable_id=str(scorable_id))
                .all()
            )
        return self._run(op)

    def list_sources_for_scorable(self, *, scorable_type: str, scorable_id: str) -> List[SourceORM]:
        def op(s):
            return (
                s.query(SourceORM)
                .join(ScorableSourceLinkORM, ScorableSourceLinkORM.source_id == SourceORM.id)
                .filter(
                    ScorableSourceLinkORM.scorable_type == scorable_type,
                    ScorableSourceLinkORM.scorable_id == str(scorable_id),
                )
                .all()
            )
        return self._run(op)


    def upsert_candidate(
        self,
        *,
        pipeline_run_id: int,
        goal_type: str,
        query_text: str,
        source_id: int,
        rank: Optional[int] = None,
        title: Optional[str] = None,
        snippet: Optional[str] = None,
        result_type: str = "unknown",
        provider: Optional[str] = None,
        status: str = "pending",
        quality_total: Optional[float] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> int:
        meta = meta or {}
        query_hash = hash_text(query_text)

        def op(s):
            existing = (
                s.query(SourceCandidateORM)
                .filter_by(
                    pipeline_run_id=pipeline_run_id,
                    goal_type=goal_type,
                    query_hash=query_hash,
                    source_id=source_id,
                )
                .first()
            )
            if existing:
                # update “latest view” fields
                if rank is not None:
                    existing.rank = rank
                if title is not None:
                    existing.title = title
                if snippet is not None:
                    existing.snippet = snippet
                if result_type:
                    existing.result_type = result_type
                if provider is not None:
                    existing.provider = provider
                if status:
                    existing.status = status
                if quality_total is not None:
                    existing.quality_total = float(quality_total)
                if meta:
                    existing.meta = {**(existing.meta or {}), **meta}
                s.flush()
                return existing.id

            obj = SourceCandidateORM(
                pipeline_run_id=pipeline_run_id,
                goal_type=goal_type,
                query_text=query_text,
                query_hash=query_hash,
                source_id=source_id,
                rank=rank,
                title=title,
                snippet=snippet,
                result_type=result_type or "unknown",
                provider=provider,
                status=status or "pending",
                quality_total=quality_total,
                meta=meta,
            )
            s.add(obj)
            s.flush()
            return obj.id

        try:
            return self._run(op)
        except IntegrityError:
            # race retry
            def op_retry(s):
                ex = (
                    s.query(SourceCandidateORM)
                    .filter_by(
                        pipeline_run_id=pipeline_run_id,
                        goal_type=goal_type,
                        query_hash=query_hash,
                        source_id=source_id,
                    )
                    .first()
                )
                return ex.id if ex else None

            out = self._run(op_retry)
            if out is None:
                raise
            return out

    def list_candidates_for_run(
        self,
        *,
        pipeline_run_id: int,
        goal_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 200,
    ) -> List[SourceCandidateORM]:
        def op(s):
            q = s.query(SourceCandidateORM).filter_by(pipeline_run_id=pipeline_run_id)
            if goal_type:
                q = q.filter_by(goal_type=goal_type)
            if status:
                q = q.filter_by(status=status)
            return q.order_by(SourceCandidateORM.rank.asc().nullslast()).limit(limit).all()

        return self._run(op)

    def upsert_quality(
        self,
        *,
        source_id: int,
        goal_type: str,
        dimension: str,
        score: float,
        judge_type: str = "heuristic_v1",
        judge_version: Optional[str] = None,
        weight: Optional[float] = None,
        rationale: Optional[str] = None,
        pipeline_run_id: Optional[int] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> int:
        meta = meta or {}
        score = float(max(0.0, min(1.0, score)))

        def op(s):
            existing = (
                s.query(SourceQualityORM)
                .filter_by(
                    source_id=source_id,
                    goal_type=goal_type,
                    dimension=dimension,
                    judge_type=judge_type,
                )
                .first()
            )
            if existing:
                existing.score = score
                existing.weight = weight
                existing.judge_version = judge_version
                existing.rationale = rationale
                existing.pipeline_run_id = pipeline_run_id
                if meta:
                    existing.meta = {**(existing.meta or {}), **meta}
                s.flush()
                return existing.id

            obj = SourceQualityORM(
                source_id=source_id,
                goal_type=goal_type,
                dimension=dimension,
                score=score,
                weight=weight,
                judge_type=judge_type,
                judge_version=judge_version,
                rationale=rationale,
                pipeline_run_id=pipeline_run_id,
                meta=meta,
            )
            s.add(obj)
            s.flush()
            return obj.id

        try:
            return self._run(op)
        except IntegrityError:
            def op_retry(s):
                ex = (
                    s.query(SourceQualityORM)
                    .filter_by(
                        source_id=source_id,
                        goal_type=goal_type,
                        dimension=dimension,
                        judge_type=judge_type,
                    )
                    .first()
                )
                return ex.id if ex else None

            out = self._run(op_retry)
            if out is None:
                raise
            return out

    def list_quality(
        self,
        *,
        source_id: int,
        goal_type: Optional[str] = None,
        judge_type: Optional[str] = None,
    ) -> List[SourceQualityORM]:
        def op(s):
            q = s.query(SourceQualityORM).filter_by(source_id=source_id)
            if goal_type:
                q = q.filter_by(goal_type=goal_type)
            if judge_type:
                q = q.filter_by(judge_type=judge_type)
            return q.order_by(SourceQualityORM.created_at.desc()).all()
        return self._run(op)

    def compute_total(
        self,
        *,
        source_id: int,
        goal_type: str,
        weights: Dict[str, float],
        judge_type: str = "heuristic_v1",
    ) -> Tuple[float, Dict[str, float]]:
        """
        Weighted mean over latest dimension scores for (source_id, goal_type, judge_type).
        Returns (total, breakdown_dict).
        """
        rows = self.list_quality(source_id=source_id, goal_type=goal_type, judge_type=judge_type)
        # keep latest per dimension
        latest: Dict[str, float] = {}
        for r in rows:
            if r.dimension not in latest:
                latest[r.dimension] = float(r.score)

        num = 0.0
        den = 0.0
        for dim, w in (weights or {}).items():
            if dim in latest:
                num += float(w) * latest[dim]
                den += float(w)

        total = (num / den) if den > 0 else 0.0
        return float(max(0.0, min(1.0, total))), latest
