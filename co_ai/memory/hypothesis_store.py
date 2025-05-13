from co_ai.memory.base_store import BaseStore

class HypothesisStore(BaseStore):
    def __init__(self, db, embeddings, logger=None):
        self.db = db
        self.embeddings = embeddings
        self.logger = logger

    def store(self, goal, text, confidence, review, features):
        embedding = self.embeddings.get_or_create(text)
        try:
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO hypotheses (goal, text, confidence, review, features, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (goal, text, confidence, review, features, embedding)
                )
            if self.logger:
                self.logger.log("HypothesisStored", {
                    "goal": goal,
                    "text": text[:100],
                    "confidence": confidence
                })
        except Exception as e:
            if self.logger:
                self.logger.log("HypothesisStoreFailed", {
                    "error": str(e),
                    "text": text[:100]
                })