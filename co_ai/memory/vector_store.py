import json

import psycopg2

from co_ai.memory.base_memory import BaseMemory
from co_ai.memory.embedding_tool import get_embedding
from co_ai.memory.hypothesis_model import Hypothesis


class VectorMemory(BaseMemory):
    def __init__(self, cfg, logger=None):
        """
        Initialize PostgreSQL connection using Hydra config.
        
        Args:
            cfg: Hydra config object with database settings under cfg.db
            logger: JSONLogger instance for structured logging
        """
        db_config = cfg.db  # Load DB config from Hydra
        self.conn = psycopg2.connect(
            dbname=db_config.database,
            user=db_config.user,
            password=db_config.password,
            host=db_config.host,
            port=db_config.port,
        )
        self.conn.autocommit = True
        self.logger = logger
        self.cfg = cfg  # Store cfg if needed later

    def store_hypothesis(self, hypothesis: Hypothesis):
        try:
            embedding = get_embedding(hypothesis.text, self.cfg)
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
            # Log the operation
            if self.logger:
                self.logger.log("HypothesisStored", {
                    "goal": hypothesis.goal,
                    "hypothesis_text": hypothesis.text[:200],
                    "confidence": hypothesis.confidence,
                    "created_at": hypothesis.created_at
                })
        except Exception as e:
            if self.logger:
                self.logger.log("HypothesisStoreFailed", {
                    "error": str(e),
                    "hypothesis_text": hypothesis.text[:200]
                })
            else:
                print(f"[VectorMemory] Failed to store hypothesis: {e}")

    def search_related(self, query: str, top_k: int = 5):
        try:
            embedding = get_embedding(query, self.cfg)
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
                results = cur.fetchall()

            if self.logger:
                self.logger.log("HypothesesSearched", {
                    "query": query,
                    "top_k": top_k,
                    "result_count": len(results)
                })

            return results
        except Exception as e:
            if self.logger:
                self.logger.log("HypothesesSearchFailed", {
                    "error": str(e),
                    "query": query
                })
            else:
                print(f"[VectorMemory] Search failed: {e}")
            return []

    def store_review(self, hypothesis_text: str, review: str):
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE hypotheses
                    SET review = %s
                    WHERE text = %s
                    """,
                    (review, hypothesis_text)
                )

            if self.logger:
                self.logger.log("ReviewStored", {
                    "hypothesis_text": hypothesis_text[:200],
                    "review_snippet": review[:100]
                })
        except Exception as e:
            if self.logger:
                self.logger.log("ReviewStoreFailed", {
                    "error": str(e),
                    "hypothesis_text": hypothesis_text[:200]
                })
            else:
                print(f"[VectorMemory] Failed to store review: {e}")

    def store_ranking(self, hypothesis: str, score: float):
        try:
            print(f"[VectorMemory] Storing ranking: '{hypothesis[:60]}...' with score {score}")
            if hasattr(self, "db"):
                self.db.execute(
                    "INSERT INTO rankings (hypothesis, score) VALUES (%s, %s) "
                    "ON CONFLICT (hypothesis) DO UPDATE SET score = EXCLUDED.score;",
                    (hypothesis, score)
                )
                self.db.commit()
            else:
                if not hasattr(self, "_rankings"):
                    self._rankings = {}
                self._rankings[hypothesis] = score

            if self.logger:
                self.logger.log("RankingStored", {
                    "hypothesis": hypothesis[:200],
                    "score": score
                })
        except Exception as e:
            if self.logger:
                self.logger.log("RankingStoreFailed", {
                    "error": str(e),
                    "hypothesis": hypothesis[:200]
                })
            else:
                print(f"[VectorMemory] Failed to store ranking: {e}")

    def log_summary(self, summary: str):
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO summaries (text)
                    VALUES (%s);
                    """,
                    (summary,)
                )

            if self.logger:
                self.logger.log("SummaryLogged", {
                    "summary_snippet": summary[:100]
                })
        except Exception as e:
            if self.logger:
                self.logger.log("SummaryLogFailed", {
                    "error": str(e),
                    "summary_snippet": summary[:100]
                })
            else:
                print(f"[VectorMemory] Failed to log summary: {e}")

    def store_prompt(self, agent_name: str, prompt_text: str, response: str = None):  
        """Insert a prompt into the prompts table."""
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO prompts (agent_name, prompt_text, response_text)
                    VALUES (%s, %s, %s)
                    """,
                    (agent_name, prompt_text, response)
                )
            if self.logger:
                self.logger.log("Prompt", {
                    "agent_name": agent_name,
                    "prompt": prompt_text,
                    "response": response
                })
        except Exception as e:
            if self.logger:
                self.logger.log("PromptLogFailed", {
                    "error": str(e),
                    "prompt_snippet": prompt_text[:100]
                })
            else:
                print(f"[VectorMemory] Failed to log Prompt: {e}")

    def store_report(self, run_id: str, goal: str, summary: str, path: str):
        """Insert a report into the reports table."""
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO reports (run_id, goal, summary, path)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (run_id, goal, summary, path)
                )
        except Exception as e:
            if self.logger:
                self.logger.log("ReportLogFailed", {
                    "error": str(e),
                    "summary_snippet": summary[:100]
                })
            else:
                print(f"[VectorMemory] Failed to log Report: {e}")
