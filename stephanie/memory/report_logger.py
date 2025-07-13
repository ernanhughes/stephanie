# stephanie/memory/report_logger.py
from stephanie.memory import BaseStore


class ReportLogger(BaseStore):
    def __init__(self, db, logger=None):
        super().__init__(db, logger)
        self.name = "report"

    def __repr__(self):
        return f"<{self.name} connected={self.db is not None}>"

    def name(self) -> str:
        return "report"

    def log(self, run_id, goal, summary, path):
        try:
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO reports (run_id, goal, summary, path)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (run_id, goal, summary, path),
                )
        except Exception as e:
            print(f"❌ Exception: {type(e).__name__}: {e}")
            if self.logger:
                self.logger.log("ReportLogFailed", {"error": str(e)})
