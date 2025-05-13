import json
from co_ai.memory.base_store import BaseStore

class ContextStore(BaseStore):
    def __init__(self, db, logger=None):
        self.db = db
        self.logger = logger

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