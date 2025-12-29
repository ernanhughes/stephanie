# stephanie/memory/report_store.py
from __future__ import annotations

from typing import List, Optional

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.orm.report import ReportORM


class ReportStore(BaseSQLAlchemyStore):
    orm_model = ReportORM
    default_order_by = ReportORM.created_at.desc()

    def __init__(self, session_or_maker, logger=None):
        super().__init__(session_or_maker, logger)
        self.name = "reports"

    def insert(self, run_id: int, goal: str, summary: str, path: str = None, content: str = None) -> ReportORM:
        def op(s):
            
            report = ReportORM(
                run_id=run_id,
                goal=goal,
                summary=summary,
                path=path,
                content=content,
            )
            s.add(report)
            s.flush()
            if self.logger:
                self.logger.log("ReportInserted", {
                    "report_id": report.id,
                    "run_id": run_id,
                    "goal": goal[:100],
                })
            return report
        return self._run(op)

    def get_content(self, run_id: int) -> Optional[str]:
        """Fetch the latest report content for a given pipeline run ID."""
        def op(s):
            
            report = (
                s.query(ReportORM)
                .filter(ReportORM.run_id == run_id)
                .order_by(ReportORM.created_at.desc())
                .first()
            )
            return report.content if report else None
        return self._run(op)

    def get_path(self, run_id: int) -> Optional[str]:
        """Fetch the latest report path for a given pipeline run ID."""
        def op(s):
            
            report = (
                s.query(ReportORM)
                .filter(ReportORM.run_id == run_id)
                .order_by(ReportORM.created_at.desc())
                .first()
            )
            return report.path if report else None
        return self._run(op)

    def get_all(self, run_id: int) -> List[ReportORM]:
        """Fetch all reports for a given pipeline run ID."""
        def op(s):
            
            return (
                s.query(ReportORM)
                .filter(ReportORM.run_id == run_id)
                .order_by(ReportORM.created_at.asc())
                .all()
            )
        return self._run(op)
