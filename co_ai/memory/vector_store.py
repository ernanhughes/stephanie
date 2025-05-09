# co_ai/memory/vector_store.py
import json

import psycopg2

from co_ai.memory.base_memory import BaseMemory
from co_ai.memory.embedding_tool import get_embedding
from co_ai.memory.hypothesis_model import Hypothesis


class VectorMemory(BaseMemory):
    def __init__(self, logger=None):
        self.conn = psycopg2.connect(
            dbname="co", user="co", password="co", host="localhost"
        )
        self.conn.autocommit = True
        self.logger = logger

    def store_hypothesis(self, hypothesis: Hypothesis):
        embedding = get_embedding(hypothesis.text)
        hypothesis.embedding = embedding
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO hypotheses (goal, text, confidence, review, embedding, features, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    hypothesis.goal,
                    hypothesis.text,
                    hypothesis.confidence,
                    hypothesis.review,
                    embedding,
                    json.dumps(hypothesis.features or []),
                    hypothesis.created_at,
                )
            )

    def search_related(self, query: str, top_k: int = 5):
        embedding = get_embedding(query)
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT text, goal, confidence, review
                FROM hypotheses
                ORDER BY embedding <-> %s
                LIMIT %s;
                """,
                (embedding, top_k)
            )
            return cur.fetchall()

    def store_review(self, hypothesis_text: str, review: str):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE hypotheses
                SET review = %s
                WHERE text = %s
                """,
                (review, hypothesis_text)
            )

    def store_ranking(self, hypothesis: str, score: float):
        """
        Store or update the ELO ranking score for a hypothesis.
        Assumes there is a 'rankings' table or collection in the backing store.

        Args:
            hypothesis (str): The hypothesis text.
            score (float): ELO score for the hypothesis.
        """
        print(f"[VectorMemory] Storing ranking: '{hypothesis[:60]}...' with score {score}")
        # Example: Store in PostgreSQL, SQLite, or in-memory dict
        if hasattr(self, "db"):
            self.db.execute(
                "INSERT INTO rankings (hypothesis, score) VALUES (%s, %s) "
                "ON CONFLICT (hypothesis) DO UPDATE SET score = EXCLUDED.score;",
                (hypothesis, score)
            )
            self.db.commit()
        else:
            # fallback if db isn't configured
            if not hasattr(self, "_rankings"):
                self._rankings = {}
            self._rankings[hypothesis] = score


    def log_summary(self, summary: str):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO summaries (text)
                VALUES (%s);
                """,
                (summary,)
            )
