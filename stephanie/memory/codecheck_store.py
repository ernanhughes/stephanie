# stephanie/stores/codecheck_store.py
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from sqlalchemy.orm import Session, selectinload

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.models.codecheck import (
    CodeCheckRunORM,
    CodeCheckFileORM,
    CodeCheckFileMetricsORM,
    CodeCheckIssueORM,
    CodeCheckSuggestionORM,
)


class CodeCheckStore(BaseSQLAlchemyStore):
    """
    SQLAlchemy data access layer for the CodeCheck component.

    Responsibilities:
      - Manage analysis runs (CodeCheckRunORM)
      - Attach files to runs (CodeCheckFileORM)
      - Store per-file metric vectors (CodeCheckFileMetricsORM)
      - Store individual issues (CodeCheckIssueORM)

    This mirrors NexusStore in structure but is scoped to code quality data.
    """

    orm_model = CodeCheckRunORM
    default_order_by = "created_ts"

    def __init__(self, session_or_maker, logger=None):
        super().__init__(session_or_maker, logger)
        self.name = "codecheck"

    # ------------------------------------------------------------------ Runs

    def create_run(
        self,
        run_id: str,
        repo_root: str,
        *,
        rel_root: Optional[str] = None,
        branch: Optional[str] = None,
        commit_hash: Optional[str] = None,
        language: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> CodeCheckRunORM:
        """Create a new analysis run. Fails if run_id already exists."""

        def op(s: Session):
            existing = s.get(CodeCheckRunORM, run_id)
            if existing is not None:
                return existing
            obj = CodeCheckRunORM(
                id=run_id,
                repo_root=repo_root,
                rel_root=rel_root,
                branch=branch,
                commit_hash=commit_hash,
                language=language,
                config=config or {},
                status="pending",
            )
            s.add(obj)
            s.flush()
            return obj

        return self._run(op)

    def update_run_status(
        self,
        run_id: str,
        *,
        status: Optional[str] = None,
        status_message: Optional[str] = None,
        summary_metrics: Optional[Dict[str, float]] = None,
        finished: bool = False,
    ) -> Optional[CodeCheckRunORM]:
        """Update status/summary for a run."""

        def op(s: Session):
            obj = s.get(CodeCheckRunORM, run_id)
            if obj is None:
                return None
            if status is not None:
                obj.status = status
            if status_message is not None:
                obj.status_message = status_message
            if summary_metrics is not None:
                obj.summary_metrics = dict(summary_metrics)
            if finished and obj.finished_ts is None:
                from datetime import datetime as _dt
                obj.finished_ts = _dt.utcnow()
            s.flush()
            return obj

        return self._run(op)

    def get_run(self, run_id: str) -> Optional[CodeCheckRunORM]:
        def op(s: Session):
            return s.get(CodeCheckRunORM, run_id)
        return self._run(op)

    def list_runs(self, limit: int = 50) -> List[CodeCheckRunORM]:
        def op(s: Session):
            q = s.query(CodeCheckRunORM).order_by(CodeCheckRunORM.created_ts.desc())
            return q.limit(limit).all()
        return self._run(op) or []

    # ------------------------------------------------------------------ Files

    def upsert_file(
        self,
        run_id: str,
        path: str,
        *,
        language: Optional[str] = None,
        module_name: Optional[str] = None,
        loc: Optional[int] = None,
        num_classes: Optional[int] = None,
        num_functions: Optional[int] = None,
        num_methods: Optional[int] = None,
        num_inner_functions: Optional[int] = None,
        is_test: bool = False,
        content_hash: Optional[str] = None,
        labels: Optional[List[str]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> CodeCheckFileORM:
        """
        Insert or update a file row for a given run.

        We treat (run_id, path) as the logical key.
        """

        def op(s: Session):
            q = (
                s.query(CodeCheckFileORM)
                .filter(
                    CodeCheckFileORM.run_id == run_id,
                    CodeCheckFileORM.path == path,
                )
            )
            obj = q.one_or_none()
            if obj is None:
                obj = CodeCheckFileORM(
                    run_id=run_id,
                    path=path,
                    language=language,
                    module_name=module_name,
                    loc=loc,
                    num_classes=num_classes,
                    num_functions=num_functions,
                    num_methods=num_methods,
                    num_inner_functions=num_inner_functions,
                    is_test=bool(is_test),
                    content_hash=content_hash,
                    labels=list(labels or []),
                    meta=dict(meta or {}),
                )
                s.add(obj)
            else:
                # Update basic stats
                if language is not None:
                    obj.language = language
                if module_name is not None:
                    obj.module_name = module_name
                if loc is not None:
                    obj.loc = loc
                if num_classes is not None:
                    obj.num_classes = num_classes
                if num_functions is not None:
                    obj.num_functions = num_functions
                if num_methods is not None:
                    obj.num_methods = num_methods
                if num_inner_functions is not None:
                    obj.num_inner_functions = num_inner_functions
                obj.is_test = bool(is_test)
                if content_hash is not None:
                    obj.content_hash = content_hash
                if labels is not None:
                    obj.labels = list(labels)
                if meta is not None:
                    obj.meta = dict(meta)
            s.flush()
            return obj

        return self._run(op)

    def list_files_for_run(
        self,
        run_id: str,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[CodeCheckFileORM]:
        """
        List files for a run, eagerly loading metrics to avoid DetachedInstanceError.

        NOTE:
        - For very large runs, keep `limit` small and paginate in the UI.
        """
        def op(s: Session) -> list[CodeCheckFileORM]:
            q = (
                s.query(CodeCheckFileORM)
                .options(selectinload(CodeCheckFileORM.metrics))
                .filter(CodeCheckFileORM.run_id == run_id)
                .order_by(CodeCheckFileORM.path.asc())
                .offset(offset)
                .limit(limit)
            )
            return q.all()

        return self._run(op)


    # ------------------------------------------------------------------ Metrics

    def upsert_file_metrics(
        self,
        file_id: int,
        columns: List[str],
        values: List[float],
        *,
        vector: Optional[Dict[str, float]] = None,
    ) -> CodeCheckFileMetricsORM:
        """
        Insert or update per-file metrics. Patterned after NexusMetricsORM.upsert_metrics.
        """

        if vector is None:
            vector = {k: float(v) for k, v in zip(columns, values)}

        def op(s: Session):
            m = s.get(CodeCheckFileMetricsORM, file_id)
            if m is None:
                m = CodeCheckFileMetricsORM(
                    file_id=file_id,
                    columns=list(columns),
                    values=[float(v) for v in values],
                    vector=dict(vector),
                )
                s.add(m)
            else:
                m.columns = list(columns)
                m.values = [float(v) for v in values]
                m.vector = dict(vector)
            s.flush()
            return m

        return self._run(op)

    # ------------------------------------------------------------------ Issues

    def add_issue(
        self,
        run_id: str,
        file_id: int,
        *,
        source: str,
        message: str,
        code: Optional[str] = None,
        issue_type: Optional[str] = None,
        severity: Optional[str] = None,
        line: Optional[int] = None,
        col: Optional[int] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> CodeCheckIssueORM:
        """Create a single issue."""

        def op(s: Session):
            obj = CodeCheckIssueORM(
                run_id=run_id,
                file_id=file_id,
                source=source,
                message=message,
                code=code,
                type=issue_type,
                severity=severity,
                line=line,
                col=col,
                meta=meta or {},
            )
            s.add(obj)
            s.flush()
            return obj

        return self._run(op)

    def add_issues_bulk(self, issues: Iterable[Dict[str, Any]]) -> int:
        """
        Bulk-insert issues.

        Each dict should have keys compatible with CodeCheckIssueORM constructor:
          run_id, file_id, source, message, code, issue_type, severity, line, col, meta
        """

        issues = list(issues)
        if not issues:
            return 0

        def op(s: Session):
            count = 0
            for row in issues:
                obj = CodeCheckIssueORM(
                    run_id=row["run_id"],
                    file_id=row["file_id"],
                    source=row["source"],
                    message=row["message"],
                    code=row.get("code"),
                    type=row.get("issue_type"),
                    severity=row.get("severity"),
                    line=row.get("line"),
                    col=row.get("col"),
                    meta=row.get("meta") or {},
                )
                s.add(obj)
                count += 1
            s.flush()
            return count

        return int(self._run(op) or 0)

    def list_issues_for_run(
        self,
        run_id: str,
        *,
        min_severity: Optional[str] = None,
        source: Optional[str] = None,
        limit: int = 5000,
    ) -> List[CodeCheckIssueORM]:
        def op(s: Session):
            q = s.query(CodeCheckIssueORM).filter(CodeCheckIssueORM.run_id == run_id)
            if source:
                q = q.filter(CodeCheckIssueORM.source == source)
            if min_severity:
                # Simple lexicographic filter; you can replace with explicit ordering if needed
                q = q.filter(CodeCheckIssueORM.severity >= min_severity)
            return (
                q.order_by(CodeCheckIssueORM.severity.desc(), CodeCheckIssueORM.line.asc())
                .limit(limit)
                .all()
            )
        return self._run(op) or []

    def list_issues_for_file(self, file_id: int, limit: int = 500) -> List[CodeCheckIssueORM]:
        def op(s: Session):
            q = (
                s.query(CodeCheckIssueORM)
                .filter(CodeCheckIssueORM.file_id == file_id)
                .order_by(CodeCheckIssueORM.line.asc())
            )
            return q.limit(limit).all()
        return self._run(op) or []



    def add_suggestions(
        self,
        run_id: str,
        file_id: int,
        suggestions: list[dict],
    ) -> list[CodeCheckSuggestionORM]:
        """
        suggestions: list of dicts with keys:
          - kind, title, summary, detail, patch, patch_type, meta
        """
        def op(session: Session):
            rows = []
            for s in suggestions:
                row = CodeCheckSuggestionORM(
                    run_id=run_id,
                    file_id=file_id,
                    kind=s["kind"],
                    title=s["title"],
                    summary=s["summary"],
                    detail=s.get("detail"),
                    patch=s.get("patch"),
                    patch_type=s.get("patch_type"),
                    status="pending",
                    meta=s.get("meta") or {},
                )
                session.add(row)
                rows.append(row)
            session.flush()
            return rows

        return self._run(op)

    def list_suggestions_for_file(self, file_id: int) -> list[CodeCheckSuggestionORM]:
        def op(session: Session):
            return (
                session.query(CodeCheckSuggestionORM)
                .filter(CodeCheckSuggestionORM.file_id == file_id)
                .order_by(CodeCheckSuggestionORM.created_ts.asc())
                .all()
            )
        return self._run(op)
