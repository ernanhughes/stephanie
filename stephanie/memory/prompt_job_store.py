from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import or_, func

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.models.prompt_job import PromptJobORM


PRIORITY_ORDER = {"high": 0, "normal": 1, "low": 2}


class PromptJobStore(BaseSQLAlchemyStore):
    orm_model = PromptJobORM
    default_order_by = "created_at"

    def __init__(self, session_maker, logger=None):
        super().__init__(session_maker, logger)
        self.name = "prompt_jobs"

    # ── Create / enqueue ──────────────────────────────────────────────────────

    def enqueue(self, job_dict: Dict[str, Any]) -> PromptJobORM:
        """Insert a new queued job (no dedupe)."""

        def op(s):
            job = PromptJobORM(**job_dict)
            s.add(job)
            s.flush()
            if self.logger:
                self.logger.log("PromptJobEnqueued", job.to_dict())
            return job

        return self._run(op)

    def upsert_by_dedupe(
        self, job_dict: Dict[str, Any]
    ) -> Tuple[PromptJobORM, bool]:
        """
        Find existing by dedupe_key (if provided), else create.
        Returns (job, created_new_flag).
        """

        def op(s):
            dedupe_key = job_dict.get("dedupe_key")
            if dedupe_key:
                existing = (
                    s.query(PromptJobORM)
                    .filter_by(dedupe_key=dedupe_key, status="queued")
                    .order_by(PromptJobORM.created_at.desc())
                    .first()
                )
                if existing:
                    # merge metadata/gen_params if desired
                    if job_dict.get("metadata"):
                        md = dict(existing.metadata or {})
                        md.update(job_dict["metadata"])
                        existing.metadata = md
                    if self.logger:
                        self.logger.log("PromptJobDeduped", existing.to_dict())
                    return existing, False

            job = PromptJobORM(**job_dict)
            s.add(job)
            s.flush()
            if self.logger:
                self.logger.log("PromptJobEnqueued", job.to_dict())
            return job, True

        return self._run(op)

    # ── Fetch next batch for workers ──────────────────────────────────────────

    def fetch_next_batch(
        self,
        max_n: int = 32,
        priorities: Optional[List[str]] = None,
        statuses: Optional[List[str]] = None,
        before_deadline: Optional[float] = None,
    ) -> List[PromptJobORM]:
        """
        Pull the next N jobs ordered by (priority, created_at). Defaults to queued only.
        """

        def op(s):
            prios = priorities or ["high", "normal", "low"]
            stats = statuses or ["queued"]

            q = s.query(PromptJobORM).filter(
                PromptJobORM.status.in_(stats),
                PromptJobORM.priority.in_(prios),
            )

            if before_deadline is not None:
                q = q.filter(
                    or_(
                        PromptJobORM.deadline_ts == None,
                        PromptJobORM.deadline_ts >= before_deadline,
                    )
                )

            # stable order: priority -> created_at
            # Do priority sorting in Python if DB doesn't support custom ordering map
            rows = (
                q.order_by(PromptJobORM.created_at.asc())
                .limit(max_n * 5)  # oversample then sort in app
                .all()
            )

            rows.sort(
                key=lambda r: (
                    PRIORITY_ORDER.get(r.priority, 1),
                    r.created_at or datetime.now(tz=timezone.utc),
                )
            )
            return rows[:max_n]

        return self._run(op)

    # ── Lifecycle updates ─────────────────────────────────────────────────────

    def mark_started(
        self, job_id: str, provider: Optional[str] = None
    ) -> Optional[PromptJobORM]:
        def op(s):
            job = s.query(PromptJobORM).filter_by(job_id=job_id).first()
            if not job:
                return None
            job.status = "running"
            job.started_at = datetime.now(tz=timezone.utc)
            if provider:
                job.provider = provider
            if self.logger:
                self.logger.log(
                    "PromptJobStarted",
                    {"job_id": job.job_id, "provider": provider},
                )
            return job

        return self._run(op)

    def append_partial(
        self, job_id: str, fragment: Dict[str, Any]
    ) -> Optional[PromptJobORM]:
        """Store streaming partials (optional)."""

        def op(s):
            job = s.query(PromptJobORM).filter_by(job_id=job_id).first()
            if not job:
                return None
            parts = list(job.partial or [])
            parts.append(fragment)
            job.partial = parts
            if self.logger:
                self.logger.log("PromptJobPartial", {"job_id": job.job_id})
            return job

        return self._run(op)

    def mark_completed(
        self,
        job_id: str,
        *,
        result_text: Optional[str] = None,
        result_json: Optional[Dict[str, Any]] = None,
        cost: Optional[Dict[str, Any]] = None,
        latency_ms: Optional[float] = None,
        cache_hit: Optional[bool] = None,
    ) -> Optional[PromptJobORM]:
        def op(s):
            job = s.query(PromptJobORM).filter_by(job_id=job_id).first()
            if not job:
                return None
            job.status = "succeeded"
            job.completed_at = datetime.now(tz=timezone.utc)
            job.result_text = result_text
            job.result_json = result_json
            job.cost = cost
            job.latency_ms = latency_ms
            if cache_hit is not None:
                job.cache_hit = cache_hit
            if self.logger:
                self.logger.log("PromptJobCompleted", {"job_id": job.job_id})
            return job

        return self._run(op)

    def mark_failed(self, job_id: str, error: str) -> Optional[PromptJobORM]:
        def op(s):
            job = s.query(PromptJobORM).filter_by(job_id=job_id).first()
            if not job:
                return None
            job.status = "failed"
            job.completed_at = datetime.now(tz=timezone.utc)
            job.error = error[:20_000]  # keep it sane
            if self.logger:
                self.logger.log(
                    "PromptJobFailed", {"job_id": job.job_id, "error": error}
                )
            return job

        return self._run(op)

    def cancel(self, job_id: str) -> Optional[PromptJobORM]:
        def op(s):
            job = s.query(PromptJobORM).filter_by(job_id=job_id).first()
            if not job:
                return None
            job.status = "canceled"
            job.completed_at = datetime.now(tz=timezone.utc)
            if self.logger:
                self.logger.log("PromptJobCanceled", {"job_id": job.job_id})
            return job

        return self._run(op)

    # ── Queries & housekeeping ────────────────────────────────────────────────

    def get_by_job_id(self, job_id: str) -> Optional[PromptJobORM]:
        def op(s):
            return s.query(PromptJobORM).filter_by(job_id=job_id).first()

        return self._run(op)

    def get_by_dedupe_key(self, dedupe_key: str) -> List[PromptJobORM]:
        def op(s):
            return (
                s.query(PromptJobORM)
                .filter_by(dedupe_key=dedupe_key)
                .order_by(PromptJobORM.created_at.desc())
                .all()
            )

        return self._run(op)

    def list_recent_for_scorable(
        self, scorable_id: str, limit: int = 50
    ) -> List[PromptJobORM]:
        def op(s):
            return (
                s.query(PromptJobORM)
                .filter_by(scorable_id=scorable_id)
                .order_by(PromptJobORM.created_at.desc())
                .limit(limit)
                .all()
            )

        return self._run(op)

    def count_by_status(self) -> Dict[str, int]:
        def op(s):
            rows = (
                s.query(PromptJobORM.status, func.count(PromptJobORM.id))
                .group_by(PromptJobORM.status)
                .all()
            )
            return {k: int(v) for k, v in rows}

        return self._run(op)

    def purge_expired(
        self, now_ts: Optional[float] = None, batch: int = 1000
    ) -> int:
        """
        Delete completed/failed jobs whose created_at + ttl_s < now.
        Returns rows deleted.
        """

        def op(s):
            now_dt = (
                datetime.fromtimestamp(now_ts, tz=timezone.utc)
                if now_ts
                else datetime.now(tz=timezone.utc)
            )
            q = s.query(PromptJobORM).filter(
                PromptJobORM.ttl_s != None,
                PromptJobORM.created_at != None,
                PromptJobORM.status.in_(["succeeded", "failed", "canceled"]),
            )
            rows = 0
            for j in q.limit(batch).all():
                if j.ttl_s is None or j.created_at is None:
                    continue
                if j.created_at + timedelta(seconds=int(j.ttl_s)) < now_dt:
                    s.delete(j)
                    rows += 1
            if rows and self.logger:
                self.logger.log("PromptJobPurged", {"count": rows})
            return rows

        return self._run(op)
