import json
from co_ai.memory.base_store import BaseStore

class PromptLogger(BaseStore):
    def __init__(self, db, logger=None):
        self.db = db
        self.logger = logger

    def log(self, agent_name, prompt_key, prompt_text, response=None, strategy="default", version=1):
        try:
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO prompts (
                        agent_name, prompt_key, prompt_text, response_text,
                        source, version, is_current, strategy, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s, TRUE, %s, %s)
                    """,
                    (agent_name, prompt_key, prompt_text, response, "manual", version, strategy, json.dumps({}))
                )
                cur.execute(
                    """
                    UPDATE prompts SET is_current = FALSE
                    WHERE agent_name = %s AND prompt_key = %s AND is_current IS TRUE
                    """,
                    (agent_name, prompt_key)
                )
        except Exception as e:
            if self.logger:
                self.logger.log("PromptLogFailed", {"error": str(e)})