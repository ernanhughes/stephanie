from typing import Optional, List
from stephanie.models.report import ReportORM
from sqlalchemy.orm import Session


class ReportStore:
    def __init__(self, session: Session, logger=None):
        self.session = session
        self.logger = logger
        self.name = "reports"

    def insert(self, run_id, goal, summary, path=None, content=None) -> ReportORM:
        try:
            report = ReportORM(
                run_id=run_id,
                goal=goal,
                summary=summary,
                path=path,
                content=content,
            )
            self.session.add(report)
            self.session.commit()
            return report
        except Exception as e:
            self.session.rollback()
            if self.logger:
                self.logger.log("ReportInsertFailed", {"error": str(e)})
            raise

    def get_content(self, run_id: int) -> Optional[str]:
        """Fetch the report content for a given pipeline run ID (latest if multiple)."""
        try:
            report = (
                self.session.query(ReportORM)
                .filter(ReportORM.run_id == run_id)
                .order_by(ReportORM.created_at.desc())
                .first()
            )
            return report.content if report else None
        except Exception as e:
            if self.logger:
                self.logger.log("ReportFetchFailed", {"error": str(e), "run_id": run_id})
            raise

    def get_path(self, run_id: int) -> Optional[str]:
        """Fetch the report path for a given pipeline run ID (latest if multiple)."""
        try:
            report = (
                self.session.query(ReportORM)
                .filter(ReportORM.run_id == run_id)
                .order_by(ReportORM.created_at.desc())
                .first()
            )
            return report.path if report else None
        except Exception as e:
            if self.logger:
                self.logger.log("ReportPathFetchFailed", {"error": str(e), "run_id": run_id})
            raise

    def get_all(self, run_id: str) -> List[ReportORM]:
        """Fetch all reports for a given pipeline run ID."""
        try:
            return (
                self.session.query(ReportORM)
                .filter(ReportORM.run_id == run_id)
                .order_by(ReportORM.created_at.asc())
                .all()
            )
        except Exception as e:
            if self.logger:
                self.logger.log("ReportFetchAllFailed", {"error": str(e), "run_id": run_id})
            raise
