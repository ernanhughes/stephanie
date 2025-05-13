import json

from co_ai.memory import BaseStore


class ContextStore(BaseStore):
    def __init__(self, db, logger=None):
        self.db = db
        self.logger = logger
        self.name = "context"

    def name(self) -> str:
        return "context"

    def save(self, run_id, stage, context, preferences=None, metadata=None):
        try:
            with self.db.cursor() as cur:
                cur.execute("UPDATE context_states SET is_current = FALSE WHERE run_id = %s AND stage_name = %s",
                            (run_id, stage))
                cur.execute("SELECT MAX(version) FROM context_states WHERE run_id = %s AND stage_name = %s",
                            (run_id, stage))
                latest = cur.fetchone()[0] or 0
                version = latest + 1
                cur.execute(
                    """
                    INSERT INTO context_states (run_id, stage_name, version, context, preferences, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (run_id, stage, version, json.dumps(context), json.dumps(preferences), json.dumps(metadata or {}))
                )
        except Exception as e:
            if self.logger:
                self.logger.log("ContextSaveFailed", {"error": str(e)})

    def has_completed(self, run_id: str, stage_name: str) -> bool:
        """Check if this stage has already been run"""
        if not run_id or not stage_name:
            return False
        with self.db.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) FROM context_states
                WHERE run_id = %s AND stage_name = %s
            """, (run_id, stage_name))
            return cur.fetchone()[0] > 0

    def load(self, run_id: str, stage: str = None) -> dict:
        """
        Load the latest saved context for a given run and optional stage.

        Args:
            run_id: Unique ID for the pipeline run
            stage: Optional stage name to resume from

        Returns:
            dict: The deserialized context
        """
        try:

            if stage:
                with self.db.cursor() as cur:
                    cur.execute("""
                        SELECT context FROM context_states
                        WHERE run_id = %s AND stage_name = %s
                        ORDER BY timestamp DESC LIMIT 1
                    """, (run_id, stage))
            else:
                with self.db.cursor() as cur:
                    cur.execute("""
                        SELECT context FROM context_states
                        WHERE run_id = %s
                        ORDER BY timestamp ASC
                    """, (run_id,))

            rows = cur.fetchall()
            if not rows:
                return {}

            # Reconstruct context by merging all stages up to this point
            result = {}
            for row in rows:
                partial_context = row[0]
                result.update(partial_context)

            return result
        except Exception as e:
            self.logger.log("ContextLoadFailed", {"error": str(e)})
            return {}

