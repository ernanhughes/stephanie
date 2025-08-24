from sqlalchemy.orm import Session

from stephanie.models.report import ReportORM


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
