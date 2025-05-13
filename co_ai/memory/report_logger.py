from co_ai.memory import BaseStore


class ReportLogger(BaseStore):
    def __init__(self, db, logger=None):
        self.db = db
        self.logger = logger
        self.name = "report"

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
                    (run_id, goal, summary, path)
                )
        except Exception as e:
            if self.logger:
                self.logger.log("ReportLogFailed", {"error": str(e)})