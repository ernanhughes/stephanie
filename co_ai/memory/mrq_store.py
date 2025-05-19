from co_ai.memory import BaseStore
import json


class MRQStoreBaseStore(BaseStore):
    def __init__(self, db, embeddings, logger=None):
        self.db = db
        self.embeddings = embeddings
        self.logger = logger
        self.name = "mrq"

    def add(
        self,
        goal: str,
        strategy: str,
        prompt: str,
        response: str,
        reward: float,
        metadata: dict = None,
    ):
        """Add a new (prompt, response, reward) to MRQ memory"""
        emb_prompt = self.embeddings.get_or_create(prompt)

        try:
            with self.db.cursor() as cur:
                cur.execute(
                    """
            INSERT INTO mrq_memory (goal, strategy, prompt, response, reward, embedding, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
                    (
                        goal,
                        strategy,
                        prompt,
                        response,
                        reward,
                        emb_prompt,
                        json.dumps(metadata or {}),
                    ),
                )
        except Exception as e:
            if self.logger:
                self.logger.log(
                    "PromptLookupFailed",
                    {"error": str(e), "prompt_snippet": prompt[:100]},
                )

    def get_similar_prompt(self, prompt: str, top_k=5):
        """Retrieve most similar prompt-response pairs"""
        emb_query = self.embeddings.get_or_create(prompt)

        try:
            with self.db.cursor() as cur:
                cur.execute(
                    """
            SELECT strategy, prompt, response, reward
            FROM mrq_memory
            ORDER BY embedding <-> %s
            LIMIT %s
            """,
                    (str(emb_query), top_k),
                )

            return [
                {"strategy": r[0], "prompt": r[1], "response": r[2], "reward": r[3]}
                for r in cur.fetchall()
            ]
        except Exception as e:
            if self.logger:
                self.logger.log(
                    "PromptLookupFailed",
                    {"error": str(e), "prompt_snippet": prompt[:100]},
                )

    def get_similar_prompts(self, prompt: str, top_k=5):
        emb = self.encoder.encode(prompt)

        try:
            with self.db.cursor() as cur:
                cur.execute(
                    """
            SELECT prompt, response, reward, strategy
            FROM mrq_memory
            ORDER BY prompt_embedding <-> %s
            LIMIT %s
            """,
                    (str(emb), top_k),
                )

            return cur.fetchall()
        except Exception as e:
            if self.logger:
                self.logger.log(
                    "PromptLookupFailed",
                    {"error": str(e), "prompt_snippet": prompt[:100]},
                )
