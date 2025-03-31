# ai_co_scientist/memory/vector_store.py
from memory.base_memory import BaseMemory
from memory.hypothesis_model import Hypothesis
from tools.embedding_tool import get_embedding
import psycopg2
import json

class VectorMemory(BaseMemory):
    def __init__(self):
        self.conn = psycopg2.connect(
            dbname="co", user="co", password="co", host="localhost"
        )
        self.conn.autocommit = True

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

    def log_summary(self, summary: str):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO summaries (text)
                VALUES (%s);
                """,
                (summary,)
            )
