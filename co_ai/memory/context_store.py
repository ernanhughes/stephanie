import json
import os
from datetime import datetime, timezone
import yaml

from co_ai.memory import BaseStore


class ContextStore(BaseStore):
    def __init__(self, db, logger=None):
        self.db = db
        self.logger = logger
        self.name = "context"
        self.dump_dir = os.path.dirname(self.logger.log_path)

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
            if self.dump_dir:
                self._dump_to_yaml(run_id, stage, context)
        except Exception as e:
            if self.logger:
                for k, v in context.items():
                    try:
                        json.dumps(v)
                    except Exception as ex:
                        print(f"❌ Key '{k}' failed: {ex}")
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
            rows = []
            if stage:
                with self.db.cursor() as cur:
                    cur.execute("""
                        SELECT context FROM context_states
                        WHERE run_id = %s AND stage_name = %s
                        ORDER BY timestamp DESC LIMIT 1
                    """, (run_id, stage))
                    rows = cur.fetchall()
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

    def _dump_to_yaml(self, run_id, stage, context):
        os.makedirs(self.dump_dir, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
        filename = f"{run_id}_{stage}_{timestamp}.yaml"
        path = os.path.join(self.dump_dir, filename)

        try:
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(context, f, allow_unicode=True, sort_keys=False)
            if self.logger:
                self.logger.log("ContextYAMLDumpSaved", {"path": path})
        except Exception as e:
            if self.logger:
                self.logger.log("ContextYAMLDumpFailed", {"error": str(e)})