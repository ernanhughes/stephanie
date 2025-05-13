import json

from co_ai.memory import BaseStore
from co_ai.tools.embedding_tool import get_embedding


class HypothesesStore(BaseStore):
    def __init__(self, db, embeddings, logger=None):
        self.db = db
        self.embeddings = embeddings
        self.logger = logger
        self.name = "hypotheses"
        self._rankings = {}

    def name(self) -> str:
        return "hypotheses"

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

    def store_review(self, hypothesis_text: str, review: dict):
        try:
            with self.db.cursor() as cur:
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
            if self.db:
                with self.db.cursor() as cur:
                    cur.execute(
                        "INSERT INTO rankings (hypothesis, score) VALUES (%s, %s) "
                        "ON CONFLICT (hypothesis) DO UPDATE SET score = EXCLUDED.score;",
                        (hypothesis, score)
                    )
                    self.db.commit()
            else:
                self._rankings[hypothesis] = score

            if self.logger:
                self.logger.log("RankingStored", {
                    "hypotheses": hypothesis[:200],
                    "score": score
                })
        except Exception as e:
            if self.logger:
                self.logger.log("RankingStoreFailed", {
                    "error": str(e),
                    "hypotheses": hypothesis[:200]
                })
            else:
                print(f"[VectorMemory] Failed to store ranking: {e}")

    def get_similar(self, goal: str, top_k: int = 5) -> list[dict[str, any]]:
        """
        Retrieve most similar hypotheses based on current goal or hypothesis.
        
        Uses embeddings to compute cosine similarity.
        """
        try:
            # Get embedding for current goal
            goal_embedding = get_embedding(goal, self.cfg)
            with self.db.cursor() as cur:
                cur.execute("""
                    SELECT text, embedding <-> %s AS distance, source, elo_rating
                    FROM hypotheses
                    WHERE enabled IS NOT FALSE
                    ORDER BY distance ASC
                    LIMIT %s
                """, (str(goal_embedding), top_k))

                results = []
                for row in cur.fetchall():
                    results.append({
                        "text": row[0],
                        "similarity": 1 - float(row[1]),  # Convert distance → similarity
                        "source": row[2],
                        "elo_rating": row[3] or 1000
                    })
                return results
        
        except Exception as e:
            print(f"[VectorMemory] Failed to fetch similar hypotheses: {e}")
            return []
        
    def get_ranked(self, goal: str, limit: int = 5) -> list[dict[str, any]]:
        """
        Get top-ranked hypotheses for the given goal.
        
        Args:
            goal: Research objective used to generate hypotheses
            limit: Number of hypotheses to retrieve
            
        Returns:
            List of dicts containing hypothesis + review + score
        """
        try:
            # Fetch top hypotheses sorted by Elo rating
            with self.db.cursor() as cur:
                cur.execute("""
                    SELECT 
                        hypothesis, 
                        reflection, 
                        elo_rating,
                        metadata
                    FROM hypothesis
                    WHERE goal = %s
                    ORDER BY elo_rating DESC
                    LIMIT %s
                """, (goal[:200], limit))

            rows = cur.fetchall()
            result = []

            for hyp, reflection, score, meta in rows:
                reflection_dict = json.loads(reflection) if isinstance(reflection, str) else reflection or {}
                
                result.append({
                    "goal": goal,
                    "hypotheses": hyp,
                    "review": reflection_dict.get("full_review", ""),
                    "score": score,
                    "elo_rating": score,
                    "metadata": meta
                })
            return result
        except Exception as e:
            print(f"[Memory] Failed to load ranked hypotheses: {e}")
            return []
